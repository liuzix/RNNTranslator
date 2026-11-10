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
        extra = ("she took a rest .", "i like the big dog .", "he said that he was tired .", "he gave me a book .",
                 # round 4: one anchor sentence per construction, each translated by the rules and refused by the cell while its item is open
                 "he is poor and could not buy a coat .", "she speaks english and also speaks swahili .", "when i was a child , i loved reading .",
                 "if it rains tomorrow , i will stay at home .", "she has a son who is a doctor .", "do you know any doctors who speak japanese ?",
                 "this problem is difficult to solve .", "it is a good idea to read this book .", "it is definite that he will go to america .",
                 "please come to talk to me .", "he comes to see me nearly every day .", "i am thinking of going to the mountains .",
                 "she insisted on going there .", "she stood looking at a boat on the river .", "she asked him to open the window .", "i made him change his plan .")
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

# ---------------------------------------------------------------- round 4: inner clauses folded as strings, read in the one frame
def agrees(root, sentences):
    """the cell translates every sentence as the rule transducer does"""
    comp, mp, sents = small_cell(root); bad = []
    for s in sentences:
        toks = R.merge_multiword(s.split()); want = R.generate(PR.parse_bounded(toks))
        try: got = R.generate(mp.parse(toks))
        except PR.ParseFail as e: got = f"refused: {str(e)[:60]}"
        if got != want: bad.append((s, want, got))
    assert not bad, f"{len(bad)} of {len(sentences)} sentences differ from the rules: {bad[0]}"

def test_clause_modes_end_at_stem_forms(root):
    """the printing rules have modes that end an inner clause at a stem form — te, the masu stem — rendered from the frame by the form
    gates, not sliced from a generated string"""
    import gen_module as GM
    cm = ST.ClauseModes(); modes = cm.modes(); rules = cm.stem_rules()
    assert modes and rules, "no stem-form print modes"
    F = GM.flatten(PR.parse_bounded(R.merge_multiword("he opened the window .".split()))); pr = GM.PrintRules()
    te = "".join(pr.run(dict(F, pmode="te"), "RESTORE")); stem = "".join(pr.run(dict(F, pmode="stem"), "RESTORE"))
    assert te.startswith("彼") and te.endswith("開けて") and stem.endswith("開け") and "。" not in te + stem

def test_coordinated_clauses_agree(root):
    """the clause before and / but / so is flushed at the conjunction in the link's mode, and the second clause follows"""
    assert ST.Coordination().link_rules(), "no rules at the conjunction"
    agrees(root, ["he is poor and could not buy a coat .", "she speaks english and also speaks swahili ."])

def test_subordinate_clause_first_agrees(root):
    """a sentence-initial subordinate clause is flushed at the comma with its conjunction, and the main clause follows"""
    assert ST.SubordinateClause().order_rules(), "no rules at the comma"
    agrees(root, ["when i was a child , i loved reading .", "if it rains tomorrow , i will stay at home ."])

def test_relative_clause_agrees(root):
    """the modified noun phrase is held while the relative clause is read and printed before it"""
    assert ST.RelativeClause().hold_rules(), "no hold of the noun phrase"
    agrees(root, ["she has a son who is a doctor .", "do you know any doctors who speak japanese ?"])

def test_infinitive_clauses_agree(root):
    """an infinitive complement of a copula is flushed in its mode: te before the adjective, のは, のには"""
    assert ST.Infinitive().mode_rules(), "no infinitive modes"
    agrees(root, ["this problem is difficult to solve .", "it is a good idea to read this book .", "it is definite that he will go to america ."])

def test_purpose_clauses_agree(root):
    """a purpose clause is flushed before the main verb held in the tags: the masu stem with に"""
    assert ST.PurposeClause().purpose_rules(), "no purpose-clause rules"
    agrees(root, ["please come to talk to me .", "he comes to see me nearly every day ."])

def test_gerund_clauses_agree(root):
    """a gerund clause is flushed before the verb held in the tags with のを, a manner clause with ながら"""
    assert ST.Gerund().gerund_rules(), "no gerund rules"
    agrees(root, ["i am thinking of going to the mountains .", "she insisted on going there .", "she stood looking at a boat on the river ."])

def test_control_clauses_agree(root):
    """the object of a control verb becomes the inner subject, and the clause is flushed with ように or in the causative with the outer tense"""
    assert ST.ControlClause().control_rules(), "no control-clause rules"
    agrees(root, ["she asked him to open the window .", "i made him change his plan ."])

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
