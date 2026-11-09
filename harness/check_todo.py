"""TODO.yaml against its schema, its pins, its acceptance tests and its stubs.

    1. TODO.yaml validates against schemas/todo.schema.json; ids are unique; depends_on names existing ids, acyclically.
    2. Every context statement is pinned by something that exists — a test of harness/test_claims.py, an action or a
       static check of harness/run.py (check_todo itself pins what an item's failing acceptance test shows) — or is a
       hypothesis.
    3. Every acceptance test names a function of harness/test_acceptance.py, each used by one item.  The tests are run:
       an item whose status is not `done` must FAIL its test (a pass means the item is done and the file is stale), an
       item marked `done` must PASS it.
    4. Every stub class an item names exists in stubs.py, an item marked `done` names none, every stub class is named by
       some item that is not done, and every public method of every stub class was exercised by the acceptance run.

    uv run python harness/check_todo.py [--root DIR] [--corpus DIR]
"""
import sys, os, re, json, argparse, importlib.util, inspect, io, contextlib
import yaml
from jsonschema import Draft202012Validator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def pins_available(root):
    """the names a context statement may be pinned by"""
    names = set()
    src = open(os.path.join(root, "harness", "test_claims.py"), encoding="utf-8").read()
    names |= set(re.findall(r"^def (test_\w+)\(", src, flags=re.M))
    run_src = open(os.path.join(root, "harness", "run.py"), encoding="utf-8").read()
    names |= set(re.findall(r'^\s+"(\w+)":\s*\{"version"', run_src, flags=re.M))                       # actions
    names |= {os.path.basename(p)[:-3] for p in re.findall(r'\("(harness/check_\w+\.py)"', run_src)}    # static checks
    return names

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=ROOT); ap.add_argument("--corpus"); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(); root = os.path.abspath(a.root); errors = []
    todo = yaml.safe_load(open(os.path.join(root, "TODO.yaml"), encoding="utf-8"))
    schema = json.load(open(os.path.join(root, "schemas", "todo.schema.json"), encoding="utf-8"))
    errors += [f"TODO.yaml: {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:140]}" for e in Draft202012Validator(schema).iter_errors(todo)]
    items = todo.get("items", []) if isinstance(todo, dict) else []
    ids = [it.get("id") for it in items]
    for i in set(ids):
        if ids.count(i) > 1: errors.append(f"TODO.yaml: duplicate id {i!r}")
    by_id = {it["id"]: it for it in items if "id" in it}
    for it in items:
        for d in it.get("depends_on", []):
            if d not in by_id: errors.append(f"TODO.yaml: {it['id']}: depends_on unknown id {d!r}")
    def reaches(start, target, seen=()):
        return any(d == target or (d in by_id and d not in seen and reaches(d, target, seen + (d,))) for d in by_id.get(start, {}).get("depends_on", []))
    for i in by_id:
        if reaches(i, i): errors.append(f"TODO.yaml: {i}: depends_on cycle")
    # 2. pins
    pins = pins_available(root)
    for it in items:
        for s in it.get("context", []):
            if "pinned_by" in s and s["pinned_by"] not in pins: errors.append(f"TODO.yaml: {it['id']}: pinned_by {s['pinned_by']!r} is not a test, an action or a check")
    # 3. acceptance tests
    T = load_module(os.path.join(root, "harness", "test_acceptance.py"), "test_acceptance_for_check")
    tests = {n for n, f in vars(T).items() if n.startswith("test_") and callable(f)}
    named = [it["acceptance"] for it in items if "acceptance" in it]
    for n in named:
        if n not in tests: errors.append(f"TODO.yaml: acceptance {n!r} is not a function of harness/test_acceptance.py")
        if named.count(n) > 1: errors.append(f"TODO.yaml: acceptance {n!r} is used by more than one item")
    for n in sorted(tests - set(named)): errors.append(f"harness/test_acceptance.py: {n} belongs to no item of TODO.yaml")
    results = {}
    if not errors:
        with contextlib.redirect_stdout(io.StringIO()):
            T.setup(root, a.corpus); results = T.run(root)
        for it in items:
            ok, msg = results.get(it["acceptance"], (None, "not run"))
            if it["status"] == "done" and ok is not True: errors.append(f"{it['id']} is marked done but its acceptance test fails: {msg}")
            if it["status"] != "done" and ok is True: errors.append(f"{it['id']} is {it['status']} but its acceptance test passes: mark it done (and retire its stubs)")
    # 4. stubs
    ST = load_module(os.path.join(root, "stubs.py"), "stubs_for_check")
    classes = {n: c for n, c in vars(ST).items() if inspect.isclass(c) and issubclass(c, ST.Stub) and c is not ST.Stub}
    named_stubs = {s for it in items for s in it.get("stubs", []) if it["status"] != "done"}
    for it in items:
        for s in it.get("stubs", []):
            if s not in classes: errors.append(f"TODO.yaml: {it['id']}: stub {s!r} is not a class of stubs.py")
            if it["status"] == "done": errors.append(f"TODO.yaml: {it['id']} is done but still names stub {s!r}")
        for s in it.get("stubs", []):
            if s in classes and classes[s].item != it["id"]: errors.append(f"stubs.py: {s}.item is {classes[s].item!r}, the item naming it is {it['id']!r}")
    for n in sorted(set(classes) - named_stubs): errors.append(f"stubs.py: {n} is named by no open item of TODO.yaml")
    if results:
        methods = {f"{n}.{m}" for n, c in classes.items() for m, f in vars(c).items() if callable(f) and not m.startswith("_")}
        missed = sorted(methods - set(T.ST.CALLS))
        for m in missed: errors.append(f"stubs.py: {m} is not exercised by any acceptance test")
    for e in errors: print("error:", e)
    done = sum(1 for it in items if it.get("status") == "done")
    print(f"todo: {len(items)} items ({done} done), {sum(1 for it in items for s in it.get('context', []) if 'hypothesis' in s)} hypotheses, "
          f"{len(results)} acceptance tests run, {len(set(T.ST.CALLS)) if results else 0} stub methods exercised, {len(errors)} error(s)")
    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
