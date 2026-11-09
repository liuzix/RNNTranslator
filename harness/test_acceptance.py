"""Acceptance tests for the items of TODO.yaml: one function per item, named by the item's `acceptance` field.

An acceptance test states what "done" means for its item and is written today, against the stubs in stubs.py where
the implementation does not exist yet.  harness/check_todo.py runs every test: an item that is not done must FAIL its
test (a passing test means the item is done and TODO.yaml is stale), an item marked done must PASS it, and every stub
method must have been exercised by the run (a stub returns TODO, an empty false value, so a test reaches its assertions
and fails there rather than at the first call).  Call every stub method you rely on before the first assertion, so the
whole interface is exercised even while the item is open.

    uv run python harness/test_acceptance.py [--corpus DIR] [-k name]       # run the acceptance tests and report pass/fail
"""
import sys, os, json, glob, argparse, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = {}

def setup(root, corpus=None):
    sys.path.insert(0, root)
    if corpus: os.environ["ENJA_DATA"] = corpus
    elif not os.path.isdir(os.path.join(root, "data", "small_parallel_enja")):
        lock = json.load(open(os.path.join(root, "harness", "corpus.lock.json"), encoding="utf-8"))
        for base in (root, ROOT):
            d = os.path.join(base, ".cache", "corpus", lock["commit"])
            if os.path.isdir(d): os.environ["ENJA_DATA"] = d; break
    global R, P, PR, PM, H, ST, LAT
    import rules_ja as R, parser as P, parser_rules as PR, parser_matrix as PM, run_rules as H, stubs as ST
    LAT = load_module(os.path.join(root, "harness", "lattice.py"), "lattice_for_acceptance")

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def small_cell(root):
    """the hybrid cell compiled on a few dev sentences plus the sentences the tests below need"""
    if "comp" not in _cache:
        extra = ("she took a rest .", "i like the big dog .", "he said that he was tired .", "he gave me a book .")
        sents = [R.merge_multiword(en) for en, ja in H.load("dev")[:60]] + [R.merge_multiword(s.split()) for s in extra]
        _cache["comp"] = PM.build_hybrid(sents); _cache["mp"] = PM.MatrixParser(_cache["comp"]); _cache["sents"] = sents
    return _cache["comp"], _cache["mp"], _cache["sents"]

def baseline(root): return json.load(open(os.path.join(root, "harness", "baseline.json"), encoding="utf-8"))

def suite_closed(root, cls):
    suite = json.load(open(os.path.join(root, "harness", "suites", f"{cls}.json"), encoding="utf-8"))
    wrong = []
    for case in suite["cases"]:
        try: out = R.translate(case["sentence"].split())
        except PR.ParseFail: out = ""
        if out != case["expect"]: wrong.append((case["sentence"], out, case["expect"]))
    assert not wrong, f"suite {cls}: {len(wrong)} of {len(suite['cases'])} cases not right: {wrong[0]}"

# ---------------------------------------------------------------- round 3
def test_torch_action_in_hook(root):
    """the cell runs under PyTorch and the hook pins it: a `torch` action exists and the baseline records zero divergence"""
    RUN = load_module(os.path.join(root, "harness", "run.py"), "run_for_acceptance")
    base = baseline(root)
    assert "torch" in RUN.ACTIONS, "no `torch` action in harness/run.py"
    assert base["detail"].get("torch", {}).get("diverging") == 0, "the baseline has no torch result with zero divergence"

def test_print_rules_match_generate(root):
    """the printing rules, stepped, emit exactly what generate() prints, for frames the rule parser fills"""
    pr = ST.PrintRules()
    positions = pr.positions(); rules = pr.rules_for(next(iter(positions), None)); derived = pr.derived()
    frames = [PR.parse_bounded(R.merge_multiword(s.split())) for s in ("i like the big dog .", "she did not read the book yesterday .", "he gave me a book .")]
    tokens = [pr.render(f) for f in frames]
    assert len(list(positions)) > 0 and rules and derived is not None
    for f, t in zip(frames, tokens): assert "".join(t) == R.generate(f), f"{t} != {R.generate(f)}"

def test_decoder_emits_japanese(root):
    """the compiled cell has an emit unit and y(t) units, and the print head's output, through the output lexical layer, is generate()'s string"""
    comp, mp, sents = small_cell(root)
    d = ST.Decoder()
    comp2 = d.compile(comp)
    mp2 = PM.MatrixParser(comp2) if comp2 else mp
    toks = R.merge_multiword("i like the big dog .".split()); frame = PR.parse_bounded(toks)
    units = d.decode(mp2, frame); first = d.output_lexicon(next(iter(units), None))
    assert ("emit",) in comp2["S"], "no emit unit in the state"
    assert first is not None and len(list(units)) > 0
    assert "".join(d.output_lexicon(u) for u in units) == R.generate(frame)

def test_hand_modules_agree(root):
    """with the PP machine, predicate and copula_predicate translated by hand, the matrix parses every single-clause sentence the rules parse"""
    hm = ST.HandModules()
    rules = list(hm.pp_rules()) + list(hm.predicate_rules()) + list(hm.copula_rules())
    assert rules, "no hand-translated rules for the PP machine, predicate or copula_predicate"
    comp, mp, sents = small_cell(root); gaps = []
    for toks in sents:
        try: frame = PR.parse_bounded(toks)
        except PR.ParseFail: continue
        if any(getattr(frame, r, None) for r in PM.UNSUPPORTED_REGS): continue
        try: out = R.generate(mp.parse(toks))
        except PR.ParseFail: out = ""
        if out != R.generate(frame): gaps.append((" ".join(toks), out))
    assert not gaps, f"{len(gaps)} single-clause sentences the matrix does not parse as the rules do: {gaps[0]}"

def test_inner_clause_sentence_agrees(root):
    """a sentence with an inner clause is parsed identically by the matrix, through flushing at clause end and tag slots"""
    cf = ST.ClauseFlush(); tags = cf.tag_slots(); flush = cf.flush_rules()
    assert tags and flush, "no tag slots or flush rules"
    comp, mp, sents = small_cell(root)
    toks = R.merge_multiword("he said that he was tired .".split())
    expect = R.generate(PR.parse_bounded(toks))
    assert R.generate(mp.parse(toks)) == expect

def test_lexical_hits_inside_cell(root):
    """the table hits are computed inside the cell from the window and the verb block, so the matrix no longer needs table_hits at run time"""
    comp, mp, sents = small_cell(root)
    units = ST.Layer0Lexical().hit_units(comp)
    assert units, "no hit units in the cell"
    saved = PM.table_hits
    try:
        PM.table_hits = lambda toks, i, verb: []
        toks = R.merge_multiword("she took a rest .".split())
        assert R.generate(mp.parse(toks)) == "彼女は休んだ。"
    finally: PM.table_hits = saved

def test_canonical_diagrams(root):
    """every control point's canonical diagram (canon.py) is complete and consistent: no gap and no conflict — which, for the
    induced clause-level rules and the noun-phrase entry state, needs the diagram built over the feasible inputs of the
    token model rather than over literals treated as independent"""
    import canon as C
    comp, mp, sents = small_cell(root)
    by_prev = {}
    for p, l, o in comp["rules"]: by_prev.setdefault(p, []).append((l, o))
    tab = C.Table(); bad = []
    for prev, rs in by_prev.items():
        d = C.diagram(rs, table=tab)
        assert C.diagram(C.shuffled(rs, 1), table=tab) is d
        if not (C.complete(d) and C.consistent(d)): bad.append(prev)
    assert not bad, f"{len(bad)} of {len(by_prev)} control points have a gap or a conflict, e.g. {bad[0]}"

# ---------------------------------------------------------------- carried over from round 2
def test_matrix_items_scope(root):
    """a matrix item exists only for sentences the rules parse, so both parsers refusing is not an AGREE that a rule gain can lose"""
    base = baseline(root); outs = base["outputs"]
    both_refused = [(split, key) for split in ("dev", "test") for key, out in outs["rules"][split].items() if not out and key in outs["matrix"].get(split, {})]
    assert not both_refused, f"{len(both_refused)} matrix items on sentences the rules refuse, e.g. {both_refused[0]}"

def test_suite_i_closed(root): suite_closed(root, "I")
def test_suite_d_closed(root): suite_closed(root, "D")
def test_suite_e_closed(root): suite_closed(root, "E")
def test_suite_g_closed(root): suite_closed(root, "G")
def test_suite_h_closed(root): suite_closed(root, "H")
def test_suite_l_closed(root): suite_closed(root, "L")
def test_suite_m_closed(root): suite_closed(root, "M")

# ---------------------------------------------------------------- runner
def run(root, only=None):
    """-> {name: (passed, message)} for every acceptance test"""
    results = {}
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)) or (only and only not in name): continue
        try: fn(root); results[name] = (True, "")
        except Exception as e: results[name] = (False, f"{type(e).__name__}: {str(e)[:200]}")
    return results

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=ROOT); ap.add_argument("--corpus"); ap.add_argument("-k")
    a = ap.parse_args(); root = os.path.abspath(a.root); setup(root, a.corpus)
    results = run(root, a.k)
    for name, (ok, msg) in results.items(): print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {msg}" if msg else ""))
    print(f"acceptance: {sum(ok for ok, _ in results.values())}/{len(results)} pass; stub methods exercised: {len(ST.CALLS)}")

if __name__ == "__main__":
    main()
