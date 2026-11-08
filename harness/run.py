"""The harness: hermetic, content-addressed checks — a small Bazel-style action cache without Bazel.

An action is (name, version, declared inputs, command).  Its key is the sha256 of the action's name and version,
the content hash of every declared input, the dependency lock (uv.lock) and the interpreter version.  The action
runs in a sandbox directory holding only its declared inputs (hard links), with a minimal environment and a fixed
hash seed, and writes result.json.  The cache keeps result.json and stdout under .cache/actions/<key>/; a hit
replays without running.  Nothing an action did not declare can reach it, so a hit is sound.

    python harness/run.py fetch-corpus            clone the pinned corpus commit into .cache/corpus/ and verify its hashes
    python harness/run.py check [--staged] [--full]
                                                  the static checks, then the actions (on the index when --staged), then the
                                                  lattice comparison with harness/baseline.json: no item may move down
    python harness/run.py baseline [--staged]     run the checks and write harness/baseline.json (commit it with the change)
    python harness/run.py install-hooks           point git at .githooks/ so pre-commit runs `check --staged`
    python harness/run.py frontier [--list]       the items not at the top of their chain, grouped by blocker (harness/lattice.py)
    python harness/run.py readme                  render README.md from docs/README.tmpl.md (numbers from harness/facts.py)
    python harness/run.py render-diagram          render docs/*.puml with the pinned PlantUML jar and update its lock

Static checks (fast, run every time, not cached): the schema, file-tree, diagram, README and claim checks (STATIC below).
Actions (inputs listed in ACTIONS below):
    rules    coverage and char-BLEU of the rule transducer on dev and test, and every output
    matrix   the hybrid matrix parser compiled on dev+test and cross-tabulated against the bounded rule parser
    np       np_module against Parser.parse_np over all windows of length <= 2 (<= 3 with --full)
    negatives  the ungrammatical sentences of harness/negatives.json refused by both parsers, their controls accepted
"""
import sys, os, json, hashlib, subprocess, shutil, tempfile, glob, time, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".cache")
CORPUS_FILES = ("dev.en", "dev.ja", "test.en", "test.ja")
CORPUS_REL = os.path.join("data", "small_parallel_enja")

SOURCES = ["rules_ja.py", "tables.py", "tables/*.tbl", "parser.py", "parser_rules.py", "run_rules.py"]
ACTIONS = {
    "rules":  {"version": 1, "script": "harness/eval_rules.py",  "inputs": SOURCES, "corpus": True,  "args": []},
    "matrix": {"version": 1, "script": "harness/eval_matrix.py", "inputs": SOURCES + ["parser_matrix.py", "np_module.py"], "corpus": True, "args": []},
    "np":     {"version": 1, "script": "harness/eval_np.py",     "inputs": SOURCES + ["parser_matrix.py", "np_module.py"], "corpus": False, "args": ["--max-len", "2"]},
    "negatives": {"version": 1, "script": "harness/eval_negatives.py", "inputs": SOURCES + ["parser_matrix.py", "np_module.py", "harness/negatives.json"], "corpus": True, "args": []},
}
FULL_NP_LEN = 3            # `check --full`: the np action over windows up to this length (~4 minutes)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def read_corpus_lock(root=ROOT):
    """harness/corpus.lock.json (schema: schemas/corpus-lock.schema.json) -> {repo, commit, sha256: {file: hash}}"""
    doc = json.load(open(os.path.join(root, "harness", "corpus.lock.json"), encoding="utf-8"))
    return {"repo": doc["repo"], "commit": doc["commit"], "sha256": {f: spec["sha256"] for f, spec in doc["files"].items()}}

def corpus_dir(lock): return os.path.join(CACHE, "corpus", lock["commit"])

def verify_corpus(lock, d):
    bad = []
    for name, want in lock["sha256"].items():
        p = os.path.join(d, name)
        if not os.path.exists(p): bad.append((name, "missing")); continue
        got = sha256_file(p)
        if got != want: bad.append((name, got))
    return bad

def fetch_corpus(src=None):
    lock = read_corpus_lock(); d = corpus_dir(lock)
    if os.path.isdir(d) and not verify_corpus(lock, d):
        print(f"corpus: {lock['commit'][:12]} already present and verified"); return d
    tmp = d + ".tmp"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    if src:
        for name in CORPUS_FILES: shutil.copy2(os.path.join(src, name), tmp)
    else:
        run = lambda *a: subprocess.run(a, cwd=tmp, check=True, capture_output=True, text=True)
        run("git", "init", "-q"); run("git", "remote", "add", "origin", lock["repo"])
        run("git", "fetch", "-q", "--depth", "1", "origin", lock["commit"]); run("git", "checkout", "-q", "FETCH_HEAD")
    bad = verify_corpus(lock, tmp)
    if bad:
        shutil.rmtree(tmp); sys.exit("corpus: hash mismatch — " + ", ".join(f"{n}: {g}" for n, g in bad))
    os.makedirs(os.path.dirname(d), exist_ok=True)
    if os.path.isdir(d): shutil.rmtree(d)
    os.rename(tmp, d); print(f"corpus: fetched {lock['commit'][:12]} and verified {len(lock['sha256'])} files"); return d

# ------------------------------------------------------------------------------------------------ the source tree under test
def source_tree(staged):
    """the working tree, or the index exported to a temporary directory (what the commit will contain)"""
    if not staged: return ROOT, None
    tmp = tempfile.mkdtemp(prefix="staged-", dir=ensure(os.path.join(CACHE, "tmp")))
    subprocess.run(["git", "checkout-index", "-a", "-f", f"--prefix={tmp}/"], cwd=ROOT, check=True)
    return tmp, tmp

def ensure(d): os.makedirs(d, exist_ok=True); return d

def expand_inputs(src_root, patterns):
    files = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(src_root, pat)))
        if not hits: sys.exit(f"harness: declared input {pat!r} matches nothing in {src_root}")
        files += [os.path.relpath(h, src_root) for h in hits]
    return files

def venv_python():
    """the interpreter of the locked environment (uv sync --locked keeps it in step with uv.lock)"""
    subprocess.run(["uv", "sync", "--locked", "--quiet"], cwd=ROOT, check=True)
    exe = os.path.join(ROOT, ".venv", "bin", "python") if os.name != "nt" else os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    ver = subprocess.run([exe, "-c", "import sys; print(sys.version)"], capture_output=True, text=True, check=True).stdout.strip()
    return exe, ver

# ------------------------------------------------------------------------------------------------ actions
def action_key(name, spec, src_root, inputs, corpus_hashes, lock_hash, pyver, args):
    rec = {"name": name, "version": spec["version"], "script": spec["script"], "args": args, "python": pyver, "uv.lock": lock_hash,
           "inputs": [(rel, sha256_file(os.path.join(src_root, rel))) for rel in inputs], "corpus": corpus_hashes}
    return hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest(), rec

def link_or_copy(src, dst):
    ensure(os.path.dirname(dst))
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)

def run_action(name, spec, src_root, exe, pyver, lock_hash, corpus, args, force=False):
    inputs = expand_inputs(src_root, spec["inputs"] + [spec["script"]])
    corpus_hashes = {}
    if spec["corpus"]:
        lock = read_corpus_lock(); corpus_hashes = lock["sha256"]
    key, rec = action_key(name, spec, src_root, inputs, corpus_hashes, lock_hash, pyver, args)
    cdir = os.path.join(CACHE, "actions", key)
    if not force and os.path.exists(os.path.join(cdir, "result.json")):
        res = json.load(open(os.path.join(cdir, "result.json"), encoding="utf-8"))
        print(f"[{name}] cached {key[:12]}  {res.get('summary', '')}"); return res
    sandbox = tempfile.mkdtemp(prefix=f"{name}-", dir=ensure(os.path.join(CACHE, "sandbox")))
    try:
        for rel in inputs: link_or_copy(os.path.join(src_root, rel), os.path.join(sandbox, rel))
        if spec["corpus"]:
            for fname in corpus_hashes: link_or_copy(os.path.join(corpus, fname), os.path.join(sandbox, CORPUS_REL, fname))
        env = {"PATH": "/usr/bin:/bin", "HOME": sandbox, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8",
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1"}
        t0 = time.time()
        proc = subprocess.run([exe, "-I", "-B", spec["script"], "--out", "result.json", *args], cwd=sandbox, env=env, capture_output=True, text=True)
        dt = time.time() - t0
        ensure(cdir)
        open(os.path.join(cdir, "stdout.txt"), "w", encoding="utf-8").write(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))
        json.dump(rec, open(os.path.join(cdir, "action.json"), "w", encoding="utf-8"), indent=1, sort_keys=True)
        if proc.returncode != 0 or not os.path.exists(os.path.join(sandbox, "result.json")):
            print(f"[{name}] FAILED (exit {proc.returncode}) in {dt:.0f}s; log: {os.path.join(cdir, 'stdout.txt')}")
            print(proc.stderr[-2000:]); sys.exit(1)
        shutil.copy2(os.path.join(sandbox, "result.json"), os.path.join(cdir, "result.json"))
        res = json.load(open(os.path.join(cdir, "result.json"), encoding="utf-8"))
        print(f"[{name}] ran {key[:12]} in {dt:.0f}s  {res.get('summary', '')}"); return res
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

# the static checks, in order: (script, extra arguments given the context); every one runs on the tree under test
STATIC = [
    ("harness/check_schemas.py", lambda c: ["--corpus", c["corpus"], "--quiet"] + (["--skip-baseline"] if c["skip_baseline"] else [])),
    ("harness/check_tree.py",    lambda c: ["--repo", ROOT] + (["--staged"] if c["staged"] else [])),
    ("harness/check_diagram.py", lambda c: []),
    ("harness/check_readme.py",  lambda c: []),
    ("harness/test_claims.py",   lambda c: ["--corpus", c["corpus"]]),
]
READS_BASELINE = {"harness/check_readme.py", "harness/test_claims.py", "harness/check_tree.py"}   # depend on the baseline (the README is rendered from it): skipped while it is rewritten; `check` runs them

def static_checks(exe, src_root, corpus, staged, skip_baseline=False):
    """the schema, tree, diagram, README and claim checks on the tree under test; exit 1 on an error"""
    ok = True; ctx = {"corpus": corpus, "staged": staged, "skip_baseline": skip_baseline}
    for script, extra in STATIC:
        if skip_baseline and script in READS_BASELINE: continue
        proc = subprocess.run([exe, os.path.join(src_root, script), "--root", src_root, *extra(ctx)], cwd=ROOT, capture_output=True, text=True)
        if proc.stdout.strip(): print(proc.stdout.rstrip())
        if proc.returncode != 0: ok = False; print(proc.stderr.rstrip()[-1500:])
    if not ok: sys.exit("static checks failed")

def run_all(staged, full, force=False, skip_baseline=False, keep_tree=False):
    exe, pyver = venv_python()
    lock_hash = sha256_file(os.path.join(ROOT, "uv.lock"))
    corpus = fetch_corpus()
    src_root, tmp = source_tree(staged)
    try:
        static_checks(exe, src_root, corpus, staged, skip_baseline)
        results = {}
        for name, spec in ACTIONS.items():
            args = list(spec["args"])
            if name == "np" and full: args = ["--max-len", str(FULL_NP_LEN)]
            results[name] = run_action(name, spec, src_root, exe, pyver, lock_hash, corpus, args, force)
        if keep_tree: return results, src_root, tmp
        return results
    finally:
        if tmp and not keep_tree: shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------------------------------------------------------ the baseline
BASELINE = os.path.join(ROOT, "harness", "baseline.json")

def metrics(results):
    r, m, n = results["rules"], results["matrix"], results["np"]
    out = {}
    for split in ("dev", "test"):
        out[f"rules.{split}.covered"] = r[split]["covered"]
        out[f"rules.{split}.bleu_all"] = round(r[split]["bleu_all"], 3)
    out["matrix.identical"] = m["identical"]
    out["matrix.silent_misfires"] = m["wrong_frame"] + m["matrix_parses_refused"]
    out["np.mismatches"] = n["mismatches"]
    g = results["negatives"]
    out["negatives.accepted"] = g["accepted"]; out["negatives.controls_refused"] = g["controls_refused_by_rules"]; out["negatives.matrix_accepted"] = g["matrix_accepted"]
    return out

# ------------------------------------------------------------------------------------------------ the lattice (harness/lattice.py)
import importlib.util as _ilu
def _lattice(src_root=ROOT):
    spec = _ilu.spec_from_file_location("lattice", os.path.join(src_root, "harness", "lattice.py")); mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def references(corpus):
    """{split: {citation: reference}} for the lines the harness keeps"""
    refs = {}
    for split in ("dev", "test"):
        en = [l.split() for l in open(os.path.join(corpus, split + ".en"), encoding="utf-8")]
        ja = ["".join(l.split()) for l in open(os.path.join(corpus, split + ".ja"), encoding="utf-8")]
        refs[split] = {f"{i}: {' '.join(e)}": j for i, (e, j) in enumerate(zip(en, ja), 1) if e and e[-1] in (".", "?", "!") and not any(t in (".", "?", "!") for t in e[:-1])}
    return refs

def item_levels(doc, corpus, src_root=ROOT):
    """the lattice levels of a baseline-shaped document (results or baseline.json), under the adjudications of src_root"""
    LT = _lattice(src_root)
    verdicts, ungrammatical = LT.load_adjudications(os.path.join(src_root, "harness", "adjudications.json"))
    return LT.levels(doc["outputs"]["rules"], doc["outputs"].get("matrix", {}), doc["outputs"].get("negatives", []),
                     doc["detail"]["np"]["mismatches"], references(corpus), verdicts, ungrammatical)

def as_document(results):
    """results of the actions -> the baseline's shape"""
    return {"metrics": metrics(results),
            "detail": {"rules": {s: {k: v for k, v in results["rules"][s].items() if k not in ("outputs", "refusals")} for s in ("dev", "test")},
                       "matrix": {k: v for k, v in results["matrix"].items() if k != "outputs"}, "np": results["np"],
                       "negatives": {k: v for k, v in results["negatives"].items() if k != "rows"}},
            "outputs": {"rules": {s: results["rules"][s]["outputs"] for s in ("dev", "test")}, "matrix": results["matrix"].get("outputs", {}),
                        "negatives": [{k: v for k, v in r.items() if not k.endswith(".reason")} for r in results["negatives"].get("rows", [])],
                        "refusals": {s: results["rules"][s].get("refusals", {}) for s in ("dev", "test")}}}

def compare(results, baseline_path, corpus, src_root=ROOT):
    """the gate: no item may move down or into an incomparable level; the aggregate metrics are printed as views"""
    LT = _lattice(src_root); cur = as_document(results)
    print("metrics: " + ", ".join(f"{k}={v}" for k, v in cur["metrics"].items()))
    if not os.path.exists(baseline_path):
        print("no baseline yet: run `python harness/run.py baseline` and commit harness/baseline.json"); return True
    base = json.load(open(baseline_path, encoding="utf-8"))
    if "rules" not in base.get("outputs", {}): print("the committed baseline predates the lattice: run `python harness/run.py baseline` and commit it"); return True
    new_L = item_levels(cur, corpus, src_root); old_L = item_levels(base, corpus, src_root)
    d = LT.compare_levels(old_L, new_L)
    print(f"lattice vs baseline {base.get('taken', '?')}: {len(d['up'])} item(s) up, {len(d['down'])} down, {len(d['incomparable'])} need a verdict, {len(d['added'])} added, {len(d['removed'])} removed")
    def show(label, rows, n=6):
        for item, olv, lv, det in rows[:n]:
            od, nd = det if isinstance(det, tuple) else (None, det)
            print(f"  {label} {item}: {olv} -> {lv}" + (f"   [{od or '(refused)'} -> {nd or '(refused)'}]" if od != nd else ""))
        if len(rows) > n: print(f"  … {len(rows) - n} more")
    show("up  ", d["up"]); show("DOWN", d["down"]); show("VERDICT", d["incomparable"]); show("REMOVED", d["removed"])
    if d["down"] or d["incomparable"] or d["removed"]:
        if d["incomparable"]:
            print("an output nobody has judged changed or disappeared: record a verdict on the OLD output in harness/adjudications.json")
            print("  (right / wrong / input_ungrammatical, with the reason), stage it, and run the check again")
        if d["down"] or d["removed"]: print("an item moved down: fix the cause; a baseline is never lowered to pass (CLAUDE.md §4)")
        return False
    if d["up"]: print("improvements: raise the baseline with `python harness/run.py baseline --staged` and stage it with the change")
    print("no item moved down"); return True

def write_baseline(results, staged):
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    doc = as_document(results); doc["taken"] = time.strftime("%Y-%m-%d") + f" at {head}" + (" (staged)" if staged else "")
    json.dump(doc, open(BASELINE, "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=True)
    print(f"wrote {os.path.relpath(BASELINE, ROOT)}: " + ", ".join(f"{k}={v}" for k, v in doc["metrics"].items()))
    print("now `python harness/run.py readme` if a cited number moved, then `python harness/run.py check`")

def show_frontier(listing=False, group=None):
    """the items of the committed baseline that are not at the top of their chain, grouped by what blocks them"""
    LT = _lattice(); corpus = corpus_dir(read_corpus_lock())
    if not os.path.isdir(corpus): corpus = fetch_corpus()
    base = json.load(open(BASELINE, encoding="utf-8"))
    L = item_levels(base, corpus)
    reasons = {f"{s} {k}": r for s, m in base["outputs"].get("refusals", {}).items() for k, r in m.items()}
    print(f"state ({base.get('taken', '?')}):")
    for kind, counts in LT.summary(L).items(): print(f"  {kind:16s} " + ", ".join(f"{lv} {n}" for lv, n in counts.items()))
    F = LT.frontier(L, reasons)
    print(f"frontier: {sum(len(v) for v in F.values())} item(s) below the top of their chain, in {len(F)} groups")
    for g, items in F.items():
        if group and group not in g: continue
        print(f"  {len(items):4d}  {g}")
        if listing or group:
            for it in items[:40]: print(f"          {it}")
            if len(items) > 40: print(f"          … {len(items) - 40} more")

def install_hooks():
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=ROOT, check=True)
    os.chmod(os.path.join(ROOT, ".githooks", "pre-commit"), 0o755)
    print("git core.hooksPath = .githooks; pre-commit runs `harness/run.py check --staged`")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch-corpus"); p.add_argument("--from", dest="src", help="copy the files from an existing clone instead of fetching")
    for c in ("check", "baseline"):
        p = sub.add_parser(c); p.add_argument("--staged", action="store_true", help="test the index (what the commit will contain)")
        p.add_argument("--full", action="store_true", help="np: windows up to length 3 (~4 min)"); p.add_argument("--force", action="store_true", help="ignore the cache")
    sub.add_parser("install-hooks")
    p = sub.add_parser("frontier", help="the items of the committed baseline not at the top of their chain, by blocker")
    p.add_argument("--list", action="store_true", help="list the items of every group"); p.add_argument("--group", help="list the items of the groups whose name contains this")
    sub.add_parser("readme", help="render README.md from docs/README.tmpl.md with the current facts")
    sub.add_parser("render-diagram", help="render docs/*.puml with the pinned PlantUML jar (needs Java)")
    a = ap.parse_args()
    if a.cmd == "fetch-corpus": fetch_corpus(a.src)
    elif a.cmd == "install-hooks": install_hooks()
    elif a.cmd == "frontier": show_frontier(a.list, a.group)
    elif a.cmd == "readme": subprocess.run([venv_python()[0], os.path.join(ROOT, "harness", "check_readme.py"), "--write"], check=True)
    elif a.cmd == "render-diagram": subprocess.run([venv_python()[0], os.path.join(ROOT, "harness", "render_diagram.py")], check=True)
    else:
        results, src_root, tmp = run_all(a.staged, a.full, a.force, skip_baseline=(a.cmd == "baseline"), keep_tree=True)
        try:
            if a.cmd == "baseline": write_baseline(results, a.staged)
            else:
                # compare with the baseline and the adjudications as staged, so an update commits together with its change
                base = os.path.join(src_root, "harness", "baseline.json")
                sys.exit(0 if compare(results, base, corpus_dir(read_corpus_lock()), src_root) else 1)
        finally:
            if tmp: shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
