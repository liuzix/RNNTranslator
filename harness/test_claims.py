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

def test_subject_case_and_agreement(root):
    for bad in ("me went to the store .", "him likes her .", "himself saw john .", "he have a car .", "she go to school .", "they is here .", "he are here .", "the dogs is here ."):
        try: R.translate(bad.split()); assert False, f"accepted {bad!r}"
        except PR.ParseFail as e: assert str(e).split(" at ")[0] in ("PRE: subject case", "VP0: agreement"), (bad, str(e))
    for good in ("i went to the store .", "he likes her .", "he saw john .", "he has a car .", "she goes to school .", "they are here .", "he put the book here .", "the dogs are here ."):
        R.translate(good.split())
    assert (R.number_of("he"), R.number_of("they"), R.number_of("dogs"), R.number_of("people"), R.number_of("news")) == ("3sg", "non3sg", "non3sg", None, None)

def test_np_two_adjective_slots(root):
    toks = ["a lot", "absent", "absent", "bus stop", "."]
    p = PR.Parser(list(toks), []); ja, cls = p.parse_np()
    assert p.i == 4, "parse_np did not take the whole phrase"
    out = NPM.run_module("obj", toks, 0, {"verb": None, "cop_kind": None, "subj": None}, None)
    assert out[0] == "FAIL" and out[1] == "three adjectives", out
    assert NPM.run_module("obj", ["a lot", "absent", "bus stop", "."], 0, {"verb": None, "cop_kind": None, "subj": None}, None)[0] == "RET"

# ---------------------------------------------------------------- the harness
def _runner(root):
    import importlib.util
    spec = importlib.util.spec_from_file_location("runner", os.path.join(root, "harness", "run.py")); runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner); return runner

def _fake_world(tmp, dev_outputs, matrix_outputs, negatives, adjudications=None):
    """a corpus of two dev lines, a results dict and an adjudications file, for the gate tests"""
    corpus = os.path.join(tmp, "corpus"); os.makedirs(corpus, exist_ok=True)
    open(os.path.join(corpus, "dev.en"), "w", encoding="utf-8").write("i like cats .\nhe came .\n"); open(os.path.join(corpus, "dev.ja"), "w", encoding="utf-8").write("私 は 猫 が 好き だ 。\n彼 は 来 た 。\n")
    open(os.path.join(corpus, "test.en"), "w", encoding="utf-8").write(""); open(os.path.join(corpus, "test.ja"), "w", encoding="utf-8").write("")
    src = os.path.join(tmp, "src", "harness"); os.makedirs(src, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "harness", "lattice.py"), src)
    json.dump({"sentences": adjudications or []}, open(os.path.join(src, "adjudications.json"), "w", encoding="utf-8"), ensure_ascii=False)
    results = {"rules": {"dev": {"n": 2, "covered": sum(1 for v in dev_outputs.values() if v), "bleu_all": 1.0, "outputs": dev_outputs, "refusals": {k: "x" for k, v in dev_outputs.items() if not v}},
                         "test": {"n": 0, "covered": 0, "bleu_all": 0.0, "outputs": {}, "refusals": {}}},
               "matrix": {"identical": 0, "wrong_frame": 0, "matrix_parses_refused": 0, "outputs": {"dev": matrix_outputs, "test": {}}},
               "np": {"mismatches": 0}, "negatives": {"accepted": 0, "controls_refused_by_rules": 0, "matrix_accepted": 0, "rows": negatives}}
    return corpus, os.path.join(tmp, "src"), results

def test_lattice_order(root):
    """the item order: WRONG < REFUSED < RIGHT, UNKNOWN below RIGHT and incomparable with the rest; the other chains total"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("lattice", os.path.join(root, "harness", "lattice.py")); LT = importlib.util.module_from_spec(spec); spec.loader.exec_module(LT)
    assert LT.cmp("sentence", "WRONG", "REFUSED") == "up" and LT.cmp("sentence", "REFUSED", "RIGHT") == "up" and LT.cmp("sentence", "WRONG", "RIGHT") == "up"
    assert LT.cmp("sentence", "RIGHT", "REFUSED") == "down" and LT.cmp("sentence", "RIGHT", "UNKNOWN") == "down"
    assert LT.cmp("sentence", "UNKNOWN", "REFUSED") == "incomparable" and LT.cmp("sentence", "REFUSED", "UNKNOWN") == "incomparable" and LT.cmp("sentence", "UNKNOWN", "WRONG") == "incomparable"
    assert LT.cmp("sentence", "UNKNOWN", "RIGHT") == "up" and LT.cmp("matrix", "GAP", "AGREE") == "up" and LT.cmp("matrix", "AGREE", "MISFIRE") == "down" and LT.cmp("negative", "REFUSED", "ACCEPTED") == "down"
    V = {"k": {"bad": "wrong", "ok": "right"}}; U = {"u": "ungrammatical"}
    assert LT.sentence_level("ref", "ref", "k", V, U) == "RIGHT" and LT.sentence_level("", "ref", "k", V, U) == "REFUSED" and LT.sentence_level("bad", "ref", "k", V, U) == "WRONG"
    assert LT.sentence_level("ok", "ref", "k", V, U) == "RIGHT" and LT.sentence_level("new", "ref", "k", V, U) == "UNKNOWN"
    assert LT.sentence_level("", "ref", "u", V, U) == "RIGHT" and LT.sentence_level("anything", "ref", "u", V, U) == "WRONG"
    assert LT.matrix_level("a", "a") == "AGREE" and LT.matrix_level("", "") == "AGREE" and LT.matrix_level("a", "") == "GAP" and LT.matrix_level("", "b") == "MISFIRE" and LT.matrix_level("a", "b") == "MISFIRE"

def test_precommit_hook_runs_check(root):
    """the hook runs the check on the staged tree, and the gate refuses a down move or an unjudged change"""
    hook = open(os.path.join(root, ".githooks", "pre-commit"), encoding="utf-8").read()
    assert "harness/run.py check --staged" in hook and "uv lock --check" in hook
    runner = _runner(root); import io, contextlib
    tmp = tempfile.mkdtemp()
    try:
        neg = [{"sentence": "me go .", "control": "i go .", "sentence.rules": None, "sentence.matrix": None, "control.rules": "私は行く。", "control.matrix": "私は行く。"}]
        k1, k2 = "1: i like cats .", "2: he came ."
        corpus, src, base_results = _fake_world(tmp, {k1: "私は猫が好きだ。", k2: "彼が来た。"}, {k1: "私は猫が好きだ。", k2: ""}, neg)
        bp = os.path.join(src, "harness", "baseline.json"); json.dump(runner.as_document(base_results), open(bp, "w", encoding="utf-8"), ensure_ascii=False)
        def gate(dev_outputs, matrix_outputs, negatives=neg, adj=None):
            _, src2, res = _fake_world(tmp, dev_outputs, matrix_outputs, negatives, adj)
            with contextlib.redirect_stdout(io.StringIO()): return runner.compare(res, bp, corpus, src2)
        assert gate({k1: "私は猫が好きだ。", k2: "彼が来た。"}, {k1: "私は猫が好きだ。", k2: ""})                   # same state passes
        assert gate({k1: "私は猫が好きだ。", k2: "彼が来た。"}, {k1: "私は猫が好きだ。", k2: "彼が来た。"})           # matrix GAP -> AGREE is up
        assert not gate({k1: "私は猫が好きだ。", k2: ""}, {k1: "私は猫が好きだ。", k2: ""})                          # UNKNOWN -> REFUSED needs a verdict
        assert gate({k1: "私は猫が好きだ。", k2: ""}, {k1: "私は猫が好きだ。", k2: ""}, adj=[{"citation": "dev 2: he came .", "verdict": "wrong", "output": "彼が来た。", "reason": "wrong particle"}])   # judged wrong: refusing it is up
        assert not gate({k1: "", k2: "彼が来た。"}, {k1: "", k2: ""})                                               # RIGHT -> REFUSED is down
        assert not gate({k1: "私は猫が好きだ。", k2: "彼が来た。"}, {k1: "私は猫が好きだ。", k2: ""}, negatives=[dict(neg[0], **{"sentence.rules": "私は行く。"})])   # a negative accepted is down
    finally: shutil.rmtree(tmp)

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

def test_frontier_command(root):
    """`run.py frontier` prints the state per chain and the groups of the frontier from the committed baseline"""
    out = subprocess.run([sys.executable, os.path.join(root, "harness", "run.py"), "frontier"], capture_output=True, text=True, cwd=root)
    assert out.returncode == 0, out.stderr[-500:]
    assert "state (" in out.stdout and "frontier:" in out.stdout and "sentence" in out.stdout and "matrix" in out.stdout

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

def test_claude_md_references_exist(root):
    """every path CLAUDE.md names in backticks exists in the tree, and every check it names is one the hook runs"""
    text = open(os.path.join(root, "CLAUDE.md"), encoding="utf-8").read()
    import importlib.util
    spec = importlib.util.spec_from_file_location("runner", os.path.join(root, "harness", "run.py")); runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    checks = {os.path.basename(s)[:-3] for s, _ in runner.STATIC} | set(runner.ACTIONS)
    missing = []
    for ref in re.findall(r"`([^`\n]+)`", text):
        if ref.startswith(("uv ", "git ", "{{", "# ", "hypothesis", "Pinned", "Hypothesis", "STATIC", "ACTIONS", "check_", "matrix", "rules", "np")):
            name = ref.split()[0]
            if name.startswith("check_") and name not in checks: missing.append(ref)
            continue
        if "/" in ref or "." in ref:
            if not os.path.exists(os.path.join(root, ref)): missing.append(ref)
    assert not missing, f"CLAUDE.md names things that do not exist: {missing}"
    for section in ("Read before you write", "Every claim is pinned", "reproducible and hermetic", "The hook is the gate", "Retirement and scope"):
        assert section in text

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
