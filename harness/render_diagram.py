"""Render docs/*.puml to SVG hermetically, with the renderer and the sources pinned by hash.

docs/architecture.lock.json (schema: schemas/diagram-lock.schema.json) names the PlantUML release jar, its URL and
sha256, and for each diagram the sha256 of its .puml source and of the .svg it produced.  Rendering fetches the jar
into .cache/tools/ if missing, refuses it unless its hash matches, runs it headless in PlantUML's SANDBOX security
profile (no file or network access from the diagram), writes the SVG and updates the lock.  The check
(harness/check_diagram.py) needs no Java: it compares the hashes in the lock with the files in the tree, so a source
edited without a re-render, or an SVG edited by hand, fails the hook.

    uv run python harness/render_diagram.py            render every diagram in the lock and update the lock
"""
import sys, os, json, hashlib, subprocess, urllib.request, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "docs", "architecture.lock.json")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

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
            proc = subprocess.run([java, "-Djava.awt.headless=true", "-Dfile.encoding=UTF-8", "-DPLANTUML_SECURITY_PROFILE=SANDBOX", "-jar", jar, "-tsvg", "-pipe"],
                                  stdin=fin, capture_output=True, env=env, cwd=os.path.dirname(src))
        if proc.returncode != 0 or not proc.stdout.startswith(b"<svg"):
            sys.exit(f"plantuml failed on {d['source']}:\n{proc.stderr.decode('utf-8', 'replace')[-2000:]}")
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
