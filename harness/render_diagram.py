"""Render docs/*.puml to SVG hermetically, with the renderer and the sources pinned by hash.

docs/architecture.lock.json (schema: schemas/diagram-lock.schema.json) names the PlantUML release jar, its URL and
sha256, and for each diagram the sha256 of its .puml source and of the .svg it produced.  Rendering fetches the jar
into .cache/tools/ if missing, refuses it unless its hash matches, runs it headless in PlantUML's SANDBOX security
profile (no file or network access from the diagram), writes the SVG and updates the lock.  The layout is the jar's own
engine (Smetana, the lock's `layout`), so the render needs no Graphviz and no tool that is not pinned: PlantUML's default
is an external `dot`, and where there is none it draws its error message as the picture and exits 0 — which is why the
output is refused unless it shows the labels of its source (shows_labels, shared with the check).  The check
(harness/check_diagram.py) needs no Java: it compares the hashes in the lock with the files in the tree and looks for
the source's labels in the SVG, so a source edited without a re-render, an SVG edited by hand, or an SVG that is not a
picture of its source fails the hook.

    uv run python harness/render_diagram.py            render every diagram in the lock and update the lock
"""
import sys, os, re, json, html, hashlib, subprocess, urllib.request, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "docs", "architecture.lock.json")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def label_lines(puml):
    """the lines a diagram's source puts on the picture: every line of every quoted label, without the bold markup"""
    out = []
    for label in re.findall(r'"((?:[^"\\]|\\.)*)"\s+as\s+\w+', puml):
        for line in label.split("\\n"):
            line = re.sub(r"\s+", " ", line.replace("**", " ")).strip()
            if line: out.append(line)
    return out

def shows_labels(puml, svg):
    """-> the label lines of the source that the SVG does not show (empty when the SVG is a picture of the source).  PlantUML
    writes a label line as one or more <text> elements (a bold run is its own); the comparison ignores spacing"""
    texts = "".join(html.unescape(t) for t in re.findall(r"<text[^>]*>([^<]*)</text>", svg))
    squeeze = lambda t: re.sub(r"[\s\u00a0]+", "", t)
    shown = squeeze(texts)
    return [line for line in label_lines(puml) if squeeze(line) not in shown]

def ensure_jar(lock):
    tool = lock["plantuml"]; d = os.path.join(ROOT, ".cache", "tools"); os.makedirs(d, exist_ok=True)
    jar = os.path.join(d, f"plantuml-{tool['version']}.jar")
    if not os.path.exists(jar):
        print(f"fetching {tool['url']}")
        tmp = jar + ".part"
        with urllib.request.urlopen(tool["url"]) as r, open(tmp, "wb") as f: shutil.copyfileobj(r, f)
        os.rename(tmp, jar)
    got = sha256_file(jar)
    if got != tool["sha256"]:
        os.remove(jar); sys.exit(f"plantuml jar hash {got} does not match the lock ({tool['sha256']}); removed")
    return jar

def render(lock, jar):
    java = shutil.which("java")
    if not java: sys.exit("java not found: rendering needs a JRE (the check does not)")
    env = {"PATH": "/usr/bin:/bin", "HOME": os.path.join(ROOT, ".cache", "tools"), "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
    for d in lock["diagrams"]:
        src = os.path.join(ROOT, d["source"]); out = os.path.join(ROOT, d["output"])
        with open(src, "rb") as fin:
            proc = subprocess.run([java, "-Djava.awt.headless=true", "-Dfile.encoding=UTF-8", "-DPLANTUML_SECURITY_PROFILE=SANDBOX", "-jar", jar,
                                   f"-Playout={lock['plantuml']['layout']}", "-tsvg", "-pipe"], stdin=fin, capture_output=True, env=env, cwd=os.path.dirname(src))
        if proc.returncode != 0 or not proc.stdout.startswith(b"<svg"):
            sys.exit(f"plantuml failed on {d['source']}:\n{proc.stderr.decode('utf-8', 'replace')[-2000:]}")
        missing = shows_labels(open(src, encoding="utf-8").read(), proc.stdout.decode("utf-8"))
        if missing: sys.exit(f"plantuml did not draw {d['source']}: its output lacks {len(missing)} label line(s) of the source, e.g. {missing[0]!r} (an error message drawn as the picture?)")
        open(out, "wb").write(proc.stdout)
        d["source_sha256"] = sha256_file(src); d["output_sha256"] = sha256_file(out)
        print(f"rendered {d['source']} -> {d['output']} ({len(proc.stdout)} bytes)")
    json.dump(lock, open(LOCK, "w", encoding="utf-8"), indent=1)
    print(f"updated {os.path.relpath(LOCK, ROOT)}")

def main():
    lock = json.load(open(LOCK, encoding="utf-8"))
    render(lock, ensure_jar(lock))

if __name__ == "__main__":
    main()
