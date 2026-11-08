"""The tests the README's claims are pinned to (see the README's appendix and harness/check_readme.py).

Each test_* function asserts one claim against the live tree.  Run as a script: every test runs, failures are listed,
exit 1 on any failure.  Claims about corpus-level numbers are pinned to the actions instead (rules, matrix, np), whose
results the README renders from harness/baseline.json.

    uv run python harness/test_claims.py [--root DIR]
"""
import sys, os, re, json, ast, glob, tarfile, argparse, subprocess, tempfile, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cell = {}

def setup(root, corpus=None):
    sys.path.insert(0, root)
    if corpus: os.environ["ENJA_DATA"] = corpus
    elif not os.path.isdir(os.path.join(root, "data", "small_parallel_enja")):
        lock = json.load(open(os.path.join(root, "harness", "corpus.lock.json"), encoding="utf-8"))
        for base in (root, ROOT):
            d = os.path.join(base, ".cache", "corpus", lock["commit"])
            if os.path.isdir(d): os.environ["ENJA_DATA"] = d; break
    global R, P, PR, PM, NPM, H
    import rules_ja as R, parser as P, parser_rules as PR, parser_matrix as PM, np_module as NPM, run_rules as H

def small_cell(root):
    """the hybrid cell compiled on a few dev sentences (seconds), shared by the cell tests"""
    if "comp" not in _cell:
        sents = [R.merge_multiword(en) for en, ja in H.load("dev")[:60]] + [R.merge_multiword(s.split()) for s in ("she took a rest .", "i like the big dog .")]
        _cell["comp"] = PM.build_hybrid(sents); _cell["mp"] = PM.MatrixParser(_cell["comp"])
    return _cell["comp"], _cell["mp"]

# ---------------------------------------------------------------- the circuit
def test_cell_weights_are_named_integers(root):
    """every nonzero of every layer is an integer weight between two named units, not a learned float"""
    comp, mp = small_cell(root)
    import numpy as np
    for (L, W, b), (M, bias) in zip(comp["layers"], mp.mats):
        assert np.all(np.round(M.data) == M.data), "non-integer weight"
        assert set(np.unique(M.data)) <= {-5.0, -2.0, -1.0, 1.0}, f"weights {set(np.unique(M.data))}"
        assert M.shape[0] == len(L.names), "an unnamed unit"
    assert all(isinstance(n, tuple) for n in comp["S"].names) and all(isinstance(n, tuple) for n in comp["X"].names)

def test_parser_interface_two_implementations(root):
    comp, mp = small_cell(root)
    rp = P.RuleParser()
    assert isinstance(rp, P.ParserInterface) and hasattr(mp, "parse") and mp.name == "matrix" and rp.name == "rules"
    toks = R.merge_multiword("i like the big dog .".split())
    assert R.generate(rp.parse(toks)) == R.generate(mp.parse(toks))

def test_rule_parser_bounds(root):
    assert (PR.BUFFER, PR.MAX_DEPTH, PR.WIDTH, PR.HOLD) == (4, 2, 2, 4)
    p = PR.Parser(["i", "like", "it", "."], [])
    try: p.peek(PR.BUFFER); assert False, "peek beyond the window did not raise"
    except RuntimeError: pass
    try:
        for _ in range(PR.MAX_DEPTH + 1): p.c = PR.Clause()
        assert False, "frame stack deeper than MAX_DEPTH was allowed"
    except PR.Overflow: pass
    src = open(os.path.join(root, "parser_rules.py"), encoding="utf-8").read()
    tree = ast.parse(src); rewinds = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "i" and isinstance(t.value, ast.Name) and t.value.id == "self": rewinds.append(node.lineno)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Attribute) and node.target.attr == "i" and isinstance(node.op, ast.Sub): rewinds.append(node.lineno)
    assert rewinds == [min(rewinds)] and "self.i = 0" in src.split("\n")[min(rewinds) - 1], f"the position is assigned at lines {rewinds}, not only initialised"

def test_matrix_cell_shape(root):
    comp, mp = small_cell(root)
    assert len(comp["layers"]) == 5
    kinds = {"module": 0, "outer": 0}
    for prev, lits, out in comp["rules"]:
        if isinstance(prev[0], str) and prev[0].startswith("NP"): kinds["module"] += 1
        elif not (prev[0] == "RET" or out[0][0] == "NP0" or out[0] == PM.FAIL): kinds["outer"] += 1
    assert kinds["module"] > 0 and kinds["outer"] > 0
    assert ("shift",) in comp["S"]

def test_layers_roundtrip(root):
    toks = R.merge_multiword("a lot of people came .".split())
    assert toks[0] == "a lot"
    assert {c for c, v in R.classify("rest")} >= {"verb", "noun"}
    assert R.translate("i like the big dog .".split()) == "私は大きい犬が好きだ。"

def test_feedback_edges(root):
    assert R.translate("she took a rest .".split()) == "彼女は休んだ。"            # the noun's use-verb rewrote the verb
    assert R.translate("i rest .".split()) == "私は休む。" and R.translate("i need a rest .".split()) == "私は休息が必要だ。"   # the same token read by the parser's state
    comp, mp = small_cell(root)
    assert any(n[0] == "usepair" for n in comp["layers"][0][0].names)

def test_corpus_pinned(root):
    lock = json.load(open(os.path.join(root, "harness", "corpus.lock.json"), encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", lock["commit"]) and set(lock["files"]) == {"dev.en", "dev.ja", "test.en", "test.ja"}
    base = json.load(open(os.path.join(root, "harness", "baseline.json"), encoding="utf-8"))
    assert {len(H.load(s)) for s in ("dev",)} == {base["detail"]["rules"]["dev"]["n"]} and len(H.load("test")) == base["detail"]["rules"]["test"]["n"]
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    assert not any(p.startswith("data/") for p in tracked), "the corpus is vendored"

def test_np_two_adjective_slots(root):
    toks = ["a lot", "absent", "absent", "bus stop", "."]
    p = PR.Parser(list(toks), []); ja, cls = p.parse_np()
    assert p.i == 4, "parse_np did not take the whole phrase"
    out = NPM.run_module("obj", toks, 0, {"verb": None, "cop_kind": None, "subj": None}, None)
    assert out[0] == "FAIL" and out[1] == "three adjectives", out
    assert NPM.run_module("obj", ["a lot", "absent", "bus stop", "."], 0, {"verb": None, "cop_kind": None, "subj": None}, None)[0] == "RET"

# ---------------------------------------------------------------- the harness
def test_precommit_hook_runs_check(root):
    hook = open(os.path.join(root, ".githooks", "pre-commit"), encoding="utf-8").read()
    assert "harness/run.py check --staged" in hook and "uv lock --check" in hook
    import importlib.util
    spec = importlib.util.spec_from_file_location("runner", os.path.join(root, "harness", "run.py")); runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    base = {"taken": "x", "metrics": {"rules.dev.covered": 10, "matrix.silent_misfires": 3}, "outputs": {}}
    tmp = tempfile.mkdtemp(); bp = os.path.join(tmp, "b.json"); json.dump(base, open(bp, "w"))
    def results(cov, mis): return {"rules": {"dev": {"covered": cov, "bleu_all": 1, "outputs": {}}, "test": {"covered": 1, "bleu_all": 1, "outputs": {}}}, "matrix": {"identical": 1, "wrong_frame": mis, "matrix_parses_refused": 0}, "np": {"mismatches": 0}}
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        assert runner.compare(results(10, 3), bp) and not runner.compare(results(9, 3), bp) and not runner.compare(results(10, 4), bp)
    shutil.rmtree(tmp)

def test_sandbox_holds_only_declared_inputs(root):
    import importlib.util
    spec = importlib.util.spec_from_file_location("runner", os.path.join(root, "harness", "run.py")); runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    probe = os.path.join(root, "harness", "probe_sandbox.py")
    spec_ = {"version": 1, "script": "harness/probe_sandbox.py", "inputs": ["pyproject.toml"], "corpus": False, "args": []}
    exe = sys.executable
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        res = runner.run_action("probe", spec_, root, exe, "test", "lock", None, [], force=True)
    assert set(res["files"]) == {"pyproject.toml", "harness/probe_sandbox.py"}, res["files"]
    assert set(res["env_keys"]) <= {"PATH", "HOME", "LC_ALL", "LANG", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING", "OMP_NUM_THREADS", "PWD", "OLDPWD", "SHLVL", "_"}, res["env_keys"]
    assert "rules_ja.py" not in res["files"]
    key1, _ = runner.action_key("probe", spec_, root, ["pyproject.toml", "harness/probe_sandbox.py"], {}, "lock", "py", [])
    key2, _ = runner.action_key("probe", spec_, root, ["pyproject.toml", "harness/probe_sandbox.py"], {}, "lock2", "py", [])
    assert key1 != key2

def test_dependencies_hash_pinned(root):
    text = open(os.path.join(root, "uv.lock"), encoding="utf-8").read()
    pkgs = [b for b in text.split("[[package]]")[1:] if "source = { registry" in b]
    assert pkgs, "no registry packages in uv.lock"
    for b in pkgs:
        assert re.search(r'hash = "sha256:[0-9a-f]{64}"', b), "a package without a sha256: " + b[:80]

def test_archive_is_opaque(root):
    blobs = glob.glob(os.path.join(root, "archive", "*.tar.xz"))
    assert blobs and os.path.exists(os.path.join(root, "archive", "README.md"))
    ignore = open(os.path.join(root, ".ignore"), encoding="utf-8").read()
    assert "archive/*.tar.*" in ignore
    for blob in blobs:
        with tarfile.open(blob) as t: names = t.getnames()
        assert any(n.endswith("MANIFEST.txt") for n in names)
        raw = open(blob, "rb").read()
        for n in names[:5]:
            base = os.path.basename(n).encode()
            if len(base) > 6: assert base not in raw, f"{base} readable in the blob"

def test_license_is_mit(root):
    assert open(os.path.join(root, "LICENSE"), encoding="utf-8").read().startswith("MIT License")

# ---------------------------------------------------------------- runner
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=ROOT); ap.add_argument("--corpus"); ap.add_argument("-k", help="only tests whose name contains this")
    a = ap.parse_args(); root = os.path.abspath(a.root)
    setup(root, a.corpus)
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f) and (not a.k or a.k in n)]
    failed = []
    for name, fn in tests:
        try: fn(root)
        except Exception as e:
            import traceback; failed.append(name); print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}"); traceback.print_exc(limit=2)
    print(f"claims: {len(tests) - len(failed)}/{len(tests)} tests pass" + (": " + ", ".join(failed) if failed else ""))
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
