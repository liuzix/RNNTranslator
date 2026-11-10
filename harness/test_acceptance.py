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
    """the encoder and the decoder compiled on a few dev sentences plus the sentences the tests below need"""
    if "comp" not in _cache:
        extra = ("she took a rest .", "i like the big dog .", "he said that he was tired .", "he gave me a book .",
                 # round 4: one anchor sentence per construction, each translated by the rules and refused by the cell while its item is open
                 "he is poor and could not buy a coat .", "she speaks english and also speaks swahili .", "when i was a child , i loved reading .",
                 "i loved reading when i was a child .", "i went out although it was raining .",
                 "if it rains tomorrow , i will stay at home .", "she has a son who is a doctor .", "do you know any doctors who speak japanese ?",
                 "this problem is difficult to solve .", "it is a good idea to read this book .", "it is definite that he will go to america .",
                 "please come to talk to me .", "he comes to see me nearly every day .", "i am thinking of going to the mountains .",
                 "she insisted on going there .", "she stood looking at a boat on the river .", "she asked him to open the window .", "i made him change his plan .",
                 # round 5: the anchors of the gaps recorded after clause modes
                 "i was absent from the meeting .", "he was afraid of his wife .", "i 'm sure she will come .", "i 'm afraid she will fail .",
                 "i reached school before the rain started .", "my father and i played tennis on sunday .", "unless it rains , i will go , too .",
                 "when i was a child , i loved reading when it rains .", "in fact , when i was a child , i loved reading .",
                 "what a lovely day .", "thank you for your cooperation .", "three years is a long time , i feel .",
                 "how many pens do you have ?", "he can speak both english and french .", "jim is not a lawyer but a doctor .",
                 "the policeman 's quick arrival surprised us .", "she is now staying at her uncle 's house .", "he is reading a book .")
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

# ---------------------------------------------------------------- round 4: inner clauses folded as strings, each a frame of the context
def agrees(root, sentences):
    """the cell translates every sentence as the rule transducer does"""
    comp, mp, sents = small_cell(root); bad = []
    for s in sentences:
        toks = R.merge_multiword(s.split()); want = R.generate(PR.parse_bounded(toks))
        try: got = R.generate(mp.parse(toks))
        except PR.ParseFail as e: got = f"refused: {str(e)[:60]}"
        if got != want: bad.append((s, want, got))
    assert not bad, f"{len(bad)} of {len(sentences)} sentences differ from the rules: {bad[0]}"

def translates(root, sentences):
    """the whole cell — encoder, context, decoder — prints for every sentence what the rule transducer prints"""
    comp, mp, sents = small_cell(root); bad = []
    for s in sentences:
        toks = R.merge_multiword(s.split()); want = R.translate(s.split())
        try: got = mp.translate(toks)
        except PR.ParseFail as e: got = f"refused: {str(e)[:60]}"
        if got != want: bad.append((s, want, got))
    assert not bad, f"{len(bad)} of {len(sentences)} sentences differ from the rules: {bad[0]}"

def judged_right(root, sentences):
    """the rule transducer translates each of these corpus sentences, and what it prints is RIGHT in the lattice's sense: the
    corpus reference, or an output a verdict in harness/adjudications.json calls right"""
    verdicts, ungrammatical = LAT.load_adjudications(os.path.join(root, "harness", "adjudications.json"))
    index = {" ".join(en): (split, line, ja) for split in ("dev", "test") for line, en, ja in H.load_indexed(split)}
    bad = []
    for s in sentences:
        split, line, ja = index[s]
        try: out = R.translate(s.split())
        except PR.ParseFail as e: out = ""; why = str(e)[:50]
        level = LAT.sentence_level(out, ja, f"{split} {line}: {s}", verdicts, ungrammatical)
        if level != "RIGHT": bad.append((s, level, out or why))
    assert not bad, f"{len(bad)} of {len(sentences)} sentences are not right: {bad[0]}"

def test_clause_kinds_end_at_stem_forms(root):
    """a frame of a stem-form kind — te, the masu stem — prints to that form of its predicate, by the form gates and not by slicing a
    generated string, and the outer frame it is linked from continues after it; where the predicate's stem is already out, the
    ending is chosen by the class of that stem, which the rules read from their own output and not from their position"""
    import gen_module as GM
    kinds = GM.STEM_FORMS; pr = GM.PrintRules()
    rules = [r for r in pr.rules() if isinstance(r[0], tuple) and r[0][0] == "FIN" and r[0][2] in kinds]
    assert kinds == ("te", "stem") and rules, "no clause kinds that end at a stem form"
    outer = GM.flatten(PR.parse_bounded(R.merge_multiword("he came .".split()))); inner = GM.flatten(PR.parse_bounded(R.merge_multiword("he opened the window .".split())))
    for kind, ending in (("te", "開けて"), ("stem", "開け")):
        out = "".join(pr.run([dict(outer, link=kind), dict(inner, clause=kind)]))
        assert ending in out and out.endswith("来た。") and out.index(ending) < out.index("来た"), (kind, out)
    # after a stem that is out (the progressive's い), the same position prints the ending of the stem's class: read from the output
    prog = GM.flatten(PR.parse_bounded(R.merge_multiword("he is reading a book .".split())))
    assert "読んでいて" in "".join(pr.run([dict(outer, link="te"), dict(prog, clause="te")]))
    after_stem = [r for r in rules if r[0][1] == "S"]
    assert after_stem and all(any(l.lstrip("¬").startswith("d.cls.") for l in lits) for pos, lits, out in after_stem), "an ending after a stem that does not read the stem's class"
    assert all(conj == [f"out.class={name[6:]}"] for name, conjs in pr.derived().items() if name.startswith("d.cls.") for conj in conjs)
    assert {r[0][1] for r in pr.rules() if isinstance(r[0], tuple) and r[0][0] == "FIN"} == {"V", "VA", "ARU", "S", "NA"}, "a finishing position per class"

def test_coordinated_clauses_agree(root):
    """the clause before and / but / so is a frame of its own, printed to its linking form, and the second clause follows"""
    assert ST.Coordination().link_rules(), "no rules at the conjunction"
    agrees(root, ["he is poor and could not buy a coat .", "she speaks english and also speaks swahili ."])

def test_subordinate_clauses_agree_in_both_orders(root):
    """a subordinate clause is a frame of its own and is printed first, whether it came before its main clause or after it"""
    import gen_module as GM
    rules, derived = PM.clause_rules((), {(kind, "site") for kind in ("before", "main", "after")})      # the encoder's rules at the conjunction and at the comma
    assert {prev[1] for prev, lits, out in rules if prev[0] == "OPEN"} == {"before", "main", "after"} and any(PM.FLUSH in out[2] for prev, lits, out in rules if prev[0] == "OPEN")
    moves = {tok[1] for pos, lits, (nxt, tok) in GM.PrintRules().rules() if tok and tok[0] == "h" and pos in ("P_PRE", "P0")}          # the decoder's, that put the subordinate frame first
    assert moves == {"up", "down"}, "the printing rules do not move to a subordinate frame on either side"
    agrees(root, ["when i was a child , i loved reading .", "i loved reading when i was a child .", "if it rains tomorrow , i will stay at home .", "i went out although it was raining ."])

def test_relative_clause_agrees(root):
    """a relative clause is a frame of its own, printed before the noun phrase it modifies"""
    assert ST.RelativeClause().splice_rules(), "no splice of a relative clause"
    agrees(root, ["she has a son who is a doctor .", "do you know any doctors who speak japanese ?"])

def test_infinitive_clauses_agree(root):
    """an infinitive complement of a copula is a frame linked to the copula's: te before the adjective, のは, のには"""
    assert ST.Infinitive().link_rules(), "no infinitive links"
    agrees(root, ["this problem is difficult to solve .", "it is a good idea to read this book .", "it is definite that he will go to america ."])

def test_purpose_clauses_agree(root):
    """a purpose clause is a frame printed before the main verb: the masu stem with に"""
    assert ST.PurposeClause().purpose_rules(), "no purpose-clause rules"
    agrees(root, ["please come to talk to me .", "he comes to see me nearly every day ."])

def test_gerund_clauses_agree(root):
    """a gerund clause is a frame printed where the object goes, with のを; a manner clause one printed with ながら"""
    assert ST.Gerund().gerund_rules(), "no gerund rules"
    agrees(root, ["i am thinking of going to the mountains .", "she insisted on going there .", "she stood looking at a boat on the river ."])

def test_control_clauses_agree(root):
    """the object of a control verb becomes the inner clause's subject, and the clause prints with ように or in the causative with the outer tense"""
    assert ST.ControlClause().control_rules(), "no control-clause rules"
    agrees(root, ["she asked him to open the window .", "i made him change his plan ."])

# ---------------------------------------------------------------- round 5: the gaps recorded after clause modes
def test_copula_idiom_agrees(root):
    """an adjective with its preposition is a copula kind the frame holds: the printing rules print its clause as generate() does,
    and the cell translates the sentence"""
    import gen_module as GM
    c = PR.parse_bounded(R.merge_multiword("he was afraid of his wife .".split()))
    try: got = "".join(GM.PrintRules().render(c))
    except GM.Refuse as e: got = f"refused: {e}"
    assert got == R.translate("he was afraid of his wife .".split()), got
    translates(root, ["i was absent from the meeting .", "he was afraid of his wife ."])

def test_copula_that_clause_agrees(root):
    """a that-clause after a copula is a frame of its own, and the copula's predicate prints after it"""
    translates(root, ["i 'm sure she will come .", "i 'm afraid she will fail ."])

def test_second_reading_of_a_fork_agrees(root):
    """where the rule parser resolves a fork by its second reading — before as a conjunction, and after a noun phrase as part of it —
    the cell takes that reading too"""
    translates(root, ["i reached school before the rain started .", "my father and i played tennis on sunday ."])

def test_unless_clause_prints(root):
    """the printing rules print an unless-clause as generate() does — なければ in place of the ない its finite form ends in, なら
    after any other ending — and the cell translates the sentence"""
    import copy, gen_module as GM
    pr = GM.PrintRules()
    for sentence in ("unless it rains , i will go , too .", "unless he is busy , he will come ."):
        c = PR.parse_bounded(R.merge_multiword(sentence.split()))
        try: got = "".join(pr.render(copy.deepcopy(c)))
        except GM.Refuse as e: got = f"refused: {e}"
        assert got == R.generate(copy.deepcopy(c)), (sentence, got)
    translates(root, ["unless it rains , i will go , too ."])

def test_subordinate_chains_agree(root):
    """a second subordinate clause, and a subordinate clause after the main clause's front adverb, are frames the decoder reaches"""
    translates(root, ["when i was a child , i loved reading when it rains .", "in fact , when i was a child , i loved reading ."])

def test_fixed_templates_agree(root):
    """an exclamation, a thank-you and the hearsay tail are held by the frame and printed from their templates"""
    translates(root, ["what a lovely day .", "thank you for your cooperation .", "three years is a long time , i feel ."])

def test_composed_noun_phrases_agree(root):
    """two noun phrases in one role, and the noun phrase of a question word, are slots of the frame"""
    translates(root, ["how many pens do you have ?", "he can speak both english and french .", "jim is not a lawyer but a doctor ."])

def test_possessive_noun_phrases_agree(root):
    """a possessive with 's before a noun is read by the noun-phrase module"""
    translates(root, ["the policeman 's quick arrival surprised us .", "she is now staying at her uncle 's house ."])

def test_question_mark_in_an_unseen_shape(root):
    """a question mark is read as one on a path whose traces all ended in a full stop: the cell was compiled with `i went out
    although it was raining .` and prints the question the rules print for the same sentence with a question mark"""
    comp, mp, sents = small_cell(root)
    assert R.merge_multiword("i went out although it was raining .".split()) in sents and R.merge_multiword("i went out although it was raining ?".split()) not in sents
    assert R.translate("i went out although it was raining ?".split()).endswith("か。")
    translates(root, ["i went out although it was raining ?"])

def test_predicate_stages_read_the_class(root):
    """no stage of the predicate has a position per conjugation class: a plain verb and ある have their own, every other predicate
    one per stage, the class read from the output block"""
    import gen_module as GM
    staged = [p for p in GM.PrintRules().positions() if isinstance(p, tuple) and p[0] in ("K", "V2", "V3", "V4", "V5", "FIN")]
    per_class = sorted({str(p) for p in staged if p[1] not in ("V", "VA", "ARU", "S") and p[2:3] not in (("whcop",), ("whcop_emb",))})
    assert staged and not per_class, f"{len(per_class)} positions of the predicate's stages are per class, e.g. {per_class[0]}"

def test_decoder_keeps_only_the_class(root):
    """the decoder keeps no unit per morpheme of its output, and an ending still follows the stem before it"""
    comp, mp, sents = small_cell(root); L5 = comp["decoder"]["layers"][4][0]
    kept = [n for n in L5.names if n[0] == "keepout"]
    assert not kept, f"{len(kept)} keep units, one per morpheme of the output block"
    translates(root, ["he is reading a book .", "i like the big dog ."])

def test_verb_preposition_idioms_right(root): judged_right(root, ["the new plan will call for a lot of money .", "you must make up for lost time ."])
def test_noun_phrase_adverbials_right(root): judged_right(root, ["i like summer the best .", "my friend called me a coward ."])
def test_fronted_phrases_right(root): judged_right(root, ["in the summer , it 's very humid .", "because of illness , she couldn 't come .", "looking into the room , i found nobody there ."])
def test_nominal_adjectives_right(root): judged_right(root, ["cats can see in the dark .", "you ask me to do the impossible ."])
def test_wh_clauses_after_predicate_right(root): judged_right(root, ["we went to boston , where we stayed a week .", "that 's exactly what he said ."])

def test_every_output_is_judged(root):
    """no sentence item of the baseline is UNKNOWN: every output of the rule transducer is the reference or carries a verdict"""
    verdicts, ungrammatical = LAT.load_adjudications(os.path.join(root, "harness", "adjudications.json"))
    outs = baseline(root)["outputs"]["rules"]; refs = {split: {f"{line}: {' '.join(en)}": ja for line, en, ja in H.load_indexed(split)} for split in ("dev", "test")}
    unknown = [f"{split} {key}" for split in ("dev", "test") for key, out in outs[split].items()
               if LAT.sentence_level(out, refs[split].get(key), f"{split} {key}", verdicts, ungrammatical) == "UNKNOWN"]
    assert not unknown, f"{len(unknown)} outputs nobody has judged, e.g. {unknown[0]}"

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
