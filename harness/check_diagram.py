"""The rendered diagrams must match their lock: docs/architecture.lock.json records the sha256 of each .puml source and
of the .svg rendered from it with the pinned PlantUML jar, and of each drawing script under harness/ and the .svg it
draws with nothing but Python.  A source edited without a re-render, an SVG edited by hand, or an SVG the README does
not reference fails; a rendered diagram must also show the labels of its source — matching hashes do not make a picture:
PlantUML without a layout engine draws its error message and exits 0, and such an image was once committed and pinned;
a drawing is re-run and must reproduce the committed SVG byte for byte.  No Java is needed here; rendering is
harness/render_diagram.py and harness/render_neurons.py.

    uv run python harness/check_diagram.py [--root DIR]
"""
import sys, os, json, hashlib, argparse, importlib.util
from jsonschema import Draft202012Validator

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

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
    RD = load_module(os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_diagram.py"), "render_diagram_for_check")
    for kind, entries in (("diagram", lock.get("diagrams", [])), ("drawing", lock.get("drawings", []))):
        for d in entries:
            for key, fld in (("source", "source_sha256"), ("output", "output_sha256")):
                p = os.path.join(root, d[key])
                if not os.path.exists(p): errors.append(f"{d[key]}: missing"); continue
                got = sha256_file(p)
                if got != d.get(fld): errors.append(f"{d[key]}: sha256 {got[:12]}… differs from the lock ({str(d.get(fld))[:12]}…): " + (f"re-render with {'harness/render_diagram.py' if kind == 'diagram' else d['source']}" if key == "source" else "the SVG is not the pinned render"))
            if d["output"] not in readme: errors.append(f"README.md does not reference {d['output']}")
            out = os.path.join(root, d["output"])
            svg = open(out, "rb").read() if os.path.exists(out) else b""
            if svg and not svg.startswith(b"<svg"): errors.append(f"{d['output']} is not an SVG")
            if kind == "diagram" and svg and os.path.exists(os.path.join(root, d["source"])):
                missing = RD.shows_labels(open(os.path.join(root, d["source"]), encoding="utf-8").read(), svg.decode("utf-8", "replace"))
                if missing: errors.append(f"{d['output']} is not a picture of {d['source']}: {len(missing)} of its label lines are not in the SVG, e.g. {missing[0]!r}")
            if kind == "drawing" and svg and os.path.exists(os.path.join(root, d["source"])):
                try:
                    drawn = load_module(os.path.join(root, d["source"]), "drawing_" + os.path.basename(d["source"])[:-3]).render().encode("utf-8")
                    if drawn != svg: errors.append(f"{d['output']} is not what {d['source']} draws: re-run it")
                except Exception as ex: errors.append(f"{d['source']}: render() failed: {ex}")
    for e in errors: print("error:", e)
    print(f"diagrams: {len(lock.get('diagrams', []))} pinned (PlantUML {lock.get('plantuml', {}).get('version')}), {len(lock.get('drawings', []))} drawing(s) reproduced, {len(errors)} error(s)")
    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
