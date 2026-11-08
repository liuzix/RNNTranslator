"""README.md is rendered from docs/README.tmpl.md; this check keeps it honest.

1. README.md must equal the rendering of the template with the facts in harness/facts.py ({{f:name}} placeholders), so
   every number in it is a named fact — none is typed by hand.  Any digit left in the template's prose outside a
   placeholder, code, a URL, the file tree or the appendix is an ungrounded number and fails.
2. The appendix "Claims and their pins" lists every claim as a verbatim quote with its pin: `Pinned by X` where X is a
   test in harness/test_claims.py, an action or a static check in harness/run.py (all run by the pre-commit hook),
   or `Hypothesis`.  Each quote must occur verbatim in the README body, each pin must exist, and the hook must run it.
3. Every sentence of the README body (outside code, tables, headings, the tree and the appendix) must be one of the
   appendix quotes — pinned or declared a hypothesis — so no claim goes unpinned.

    uv run python harness/check_readme.py [--root DIR]      check
    uv run python harness/check_readme.py --write           render README.md from the template
"""
import sys, os, re, ast, argparse, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPENDIX = "## Appendix: claims and their pins"
ENTRY = re.compile(r'^- "(.+)": (?:Pinned by `([\w.-]+)`|(Hypothesis))\.?\s*$')

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def strip_code(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    text = re.sub(r"\(https?://[^)]*\)", "()", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return text

def body_and_appendix(text):
    if APPENDIX not in text: return text, ""
    body, app = text.split(APPENDIX, 1)
    return body, app

def sentences(body):
    """prose sentences of the body: not headings, tables, images, the tree block, code"""
    body = re.sub(r"<!-- tree:begin -->.*?<!-- tree:end -->", "", body, flags=re.S)
    body = re.sub(r"```.*?```", "", body, flags=re.S)
    out = []
    for para in re.split(r"\n\s*\n", body):
        lines = [l for l in para.split("\n") if l.strip()]
        if not lines: continue
        if all(l.lstrip().startswith(("#", "|", "![", "<!--")) for l in lines): continue
        txt = " ".join(l.strip().lstrip("-* ").strip() for l in lines if not l.lstrip().startswith(("|", "![", "#")))
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"`(*_\[])", txt):
            s = s.strip()
            if s: out.append(s)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=ROOT); ap.add_argument("--write", action="store_true")
    a = ap.parse_args(); root = os.path.abspath(a.root)
    F = load_module(os.path.join(root, "harness", "facts.py"), "facts")
    tmpl = open(os.path.join(root, "docs", "README.tmpl.md"), encoding="utf-8").read()
    facts = F.facts(root)
    rendered = F.render(tmpl, facts)
    readme_path = os.path.join(root, "README.md")
    if a.write:
        open(readme_path, "w", encoding="utf-8").write(rendered); print("README.md rendered from docs/README.tmpl.md"); return
    errors = []
    current = open(readme_path, encoding="utf-8").read()
    if current != rendered: errors.append("README.md differs from the rendering of docs/README.tmpl.md (run `uv run python harness/check_readme.py --write`)")
    # 1. ungrounded numbers in the template
    tbody, tapp = body_and_appendix(tmpl)
    prose = strip_code(re.sub(r"<!-- tree:begin -->.*?<!-- tree:end -->", "", tbody, flags=re.S))
    prose = re.sub(r"\{\{f:[^}]+\}\}", "", prose)
    prose = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", prose)
    for m in re.finditer(r"(?<![A-Za-z_\d])\d+(?![A-Za-z_\d])", prose):          # digits inside a word (sha256) are a name, not a number
        ctx = prose[max(0, m.start() - 40):m.end() + 20].replace("\n", " ")
        errors.append(f"ungrounded number {m.group()!r} in the template: …{ctx}…")
    # 2. appendix entries
    body, app = body_and_appendix(rendered)
    entries = []
    for line in app.split("\n"):
        if not line.strip() or line.startswith(("#", "<!--")) or not line.startswith("-"): continue
        m = ENTRY.match(line)
        if not m: errors.append(f"appendix line not of the form - \"quote\": Pinned by `name` | Hypothesis: {line[:90]}"); continue
        entries.append((m.group(1), m.group(2), bool(m.group(3))))
    runner = load_module(os.path.join(root, "harness", "run.py"), "runner")
    tests = {n.name for n in ast.walk(ast.parse(open(os.path.join(root, "harness", "test_claims.py"), encoding="utf-8").read())) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    checks = {os.path.basename(s)[:-3] for s, _ in runner.STATIC}
    actions = set(runner.ACTIONS)
    hook = open(os.path.join(root, ".githooks", "pre-commit"), encoding="utf-8").read()
    if "harness/run.py check --staged" not in hook: errors.append(".githooks/pre-commit does not run harness/run.py check --staged")
    if "test_claims" not in checks: errors.append("harness/run.py does not run test_claims among its static checks")
    body_text = body
    quotes = set()
    for quote, pin, hyp in entries:
        quotes.add(quote)
        if quote not in body_text: errors.append(f"quote not found verbatim in the README body: {quote[:80]!r}")
        if hyp: continue
        if pin in tests or pin in actions or pin in checks: continue
        errors.append(f"pin {pin!r} is not a test in harness/test_claims.py, an action or a static check run by the hook (for {quote[:50]!r})")
    # 3. every body sentence is a claim entry
    for s in sentences(body):
        if s not in quotes: errors.append(f"sentence neither pinned nor declared a hypothesis in the appendix: {s[:100]!r}")
    for e in errors: print("error:", e)
    print(f"readme: {len(entries)} claims ({sum(1 for _, _, h in entries if h)} hypotheses), {len(sentences(body))} sentences, {len(errors)} error(s)")
    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
