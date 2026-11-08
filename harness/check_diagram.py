"""The rendered diagrams must match their lock: docs/architecture.lock.json records the sha256 of each .puml source and
of the .svg rendered from it with the pinned PlantUML jar.  A source edited without a re-render, an SVG edited by hand, or
an SVG the README does not reference fails.  No Java is needed here; rendering is harness/render_diagram.py.

    uv run python harness/check_diagram.py [--root DIR]
"""
import sys, os, json, hashlib, argparse
from jsonschema import Draft202012Validator

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args(); root = os.path.abspath(a.root); errors = []
    lock = json.load(open(os.path.join(root, "docs", "architecture.lock.json"), encoding="utf-8"))
    schema = json.load(open(os.path.join(root, "schemas", "diagram-lock.schema.json"), encoding="utf-8"))
    errors += [f"docs/architecture.lock.json: {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:120]}" for e in Draft202012Validator(schema).iter_errors(lock)]
    readme = open(os.path.join(root, "README.md"), encoding="utf-8").read()
    for d in lock.get("diagrams", []):
        for key, fld in (("source", "source_sha256"), ("output", "output_sha256")):
            p = os.path.join(root, d[key])
            if not os.path.exists(p): errors.append(f"{d[key]}: missing"); continue
            got = sha256_file(p)
            if got != d.get(fld): errors.append(f"{d[key]}: sha256 {got[:12]}… differs from the lock ({str(d.get(fld))[:12]}…): " + ("re-render with harness/render_diagram.py" if key == "source" else "the SVG is not the pinned render"))
        if d["output"] not in readme: errors.append(f"README.md does not reference {d['output']}")
        svg = open(os.path.join(root, d["output"]), "rb").read() if os.path.exists(os.path.join(root, d["output"])) else b""
        if svg and not svg.startswith(b"<svg"): errors.append(f"{d['output']} is not an SVG")
    for e in errors: print("error:", e)
    print(f"diagrams: {len(lock.get('diagrams', []))} pinned (PlantUML {lock.get('plantuml', {}).get('version')}), {len(errors)} error(s)")
    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
