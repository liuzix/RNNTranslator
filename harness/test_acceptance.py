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
                 "the policeman 's quick arrival surprised us .", "she is now staying at her uncle 's house .", "he is reading a book .",
                 # round 7: the folds and registers no earlier item names; the sentences of its held-out tests (he went home, he can come,
                 # the train arrived, did you read a book) stay out of this list on purpose
                 "do you know how to drive a car ?", "i don 't care whether he agrees or not .", "please tell me where the bus stop is .",
                 "i wonder what he bought .", "you have no obligation to help us .",
                 "my father boasts of the fact that he has never had a traffic accident .", "how about going to the movies tonight ?",
                 "the mystery story is so hard that i don 't like it .", "i can hear someone talking in the next room .")
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

def differing(root, sentences):
    """the sentences for which the whole cell — encoder, context, decoder — does not print what the rule transducer prints"""
    comp, mp, sents = small_cell(root); bad = []
    for s in sentences:
        toks = R.merge_multiword(s.split()); want = R.translate(s.split())
        try: got = mp.translate(toks)
        except PR.ParseFail as e: got = f"refused: {str(e)[:60]}"
        if got != want: bad.append((s, want, got))
    return bad

def translates(root, sentences):
    """the whole cell — encoder, context, decoder — prints for every sentence what the rule transducer prints"""
    bad = differing(root, sentences)
    assert not bad, f"{len(bad)} of {len(sentences)} sentences differ from the rules: {bad[0]}"

def not_right(root, sentences):
    """the corpus sentences among these that the rule transducer refuses, or translates into something that is not RIGHT in the
    lattice's sense: the corpus reference, or an output a verdict in harness/adjudications.json calls right"""
    verdicts, ungrammatical = LAT.load_adjudications(os.path.join(root, "harness", "adjudications.json"))
    index = {" ".join(en): (split, line, ja) for split in ("dev", "test") for line, en, ja in H.load_indexed(split)}
    bad = []
    for s in sentences:
        split, line, ja = index[s]
        try: out = R.translate(s.split())
        except PR.ParseFail as e: out = ""; why = str(e)[:50]
        level = LAT.sentence_level(out, ja, f"{split} {line}: {s}", verdicts, ungrammatical)
        if level != "RIGHT": bad.append((s, level, out or why))
    return bad

def judged_right(root, sentences):
    """the rule transducer translates each of these corpus sentences, and what it prints is RIGHT in the lattice's sense"""
    bad = not_right(root, sentences)
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

# ---------------------------------------------------------------- round 6: what the verdicts on every output blame
def none_judged(root, levels, causes):
    """no output of the baseline carries a verdict of one of these levels that names one of these causes"""
    outs = baseline(root)["outputs"]["rules"]; live = {f"{split} {key}": out for split in ("dev", "test") for key, out in outs[split].items()}
    doc = json.load(open(os.path.join(root, "harness", "adjudications.json"), encoding="utf-8"))["sentences"]
    bad = [a for a in doc if a["verdict"] in levels and a.get("cause") in causes and live.get(a["citation"]) == a.get("output")]
    assert not bad, f"{len(bad)} outputs judged {' or '.join(levels)} for {', '.join(causes)}, e.g. {bad[0]['citation']} -> {bad[0]['output']}"

def test_no_idiom_read_word_for_word(root): none_judged(root, ("wrong",), ("idiom",))
def test_no_sense_against_its_argument(root): none_judged(root, ("wrong",), ("sense",))
def test_no_preposition_against_its_noun(root): none_judged(root, ("wrong",), ("preposition",))
def test_no_particle_against_its_verb(root): none_judged(root, ("wrong",), ("case",))
def test_no_giving_verb_against_its_persons(root): none_judged(root, ("wrong",), ("giving",))
def test_no_form_that_is_not_japanese(root): none_judged(root, ("wrong",), ("morphology",))
def test_no_modal_or_tense_misread(root): none_judged(root, ("wrong",), ("tense",))
def test_no_construction_misread(root): none_judged(root, ("wrong",), ("structure",))
def test_no_count_without_its_counter(root): none_judged(root, ("wrong",), ("quantity",))
def test_no_dummy_it_printed(root): none_judged(root, ("wrong",), ("it",))
def test_no_word_dropped(root): none_judged(root, ("wrong",), ("omission",))
def test_no_negation_lost_or_widened(root): none_judged(root, ("wrong",), ("negation",))
def test_pronouns_said_once(root): none_judged(root, ("stiff", "wrong"), ("pronoun",))
def test_no_output_stiff(root): none_judged(root, ("stiff",), tuple(c for c in LAT.CAUSES if c != "pronoun"))

def test_situation_is_an_input(root):
    """the register is an input and not a choice of the rules: a situation head beside the sentence block has at least the values
    plain and polite, the same sentence prints in either form as the head says, a sentence whose own words ask for the other
    form is refused, and every judged output is labelled with the value of the head it was printed for"""
    head = ST.SituationHead(); values = head.values(); s = "she broke the cup .".split()
    plain, polite = head.translate(s, "plain"), head.translate(s, "polite")
    clue, none = head.clue("she broke the cup , sir .".split()), head.clue(s)
    refused = head.translate("she broke the cup , sir .".split(), "plain")
    doc = json.load(open(os.path.join(root, "harness", "adjudications.json"), encoding="utf-8"))["sentences"]
    unlabelled = [a["citation"] for a in doc if a.get("output") and a.get("situation") not in tuple(values) + ("any",)]
    assert {"plain", "polite"} <= set(values), values
    assert plain == "彼女はカップを壊した。" and polite == "彼女はカップを壊しました。", (plain, polite)
    assert clue == "polite" and none is None and refused == "", (clue, none, refused)
    assert not unlabelled, f"{len(unlabelled)} judged outputs carry no situation, e.g. {unlabelled[0]}"

def test_model_verdicts_audited(root):
    """a person has given or confirmed at least one verdict in every group of judged outputs — each level, and each cause of a
    stiff or wrong one — so no group of the frontier rests on the model's reading alone"""
    doc = json.load(open(os.path.join(root, "harness", "adjudications.json"), encoding="utf-8"))["sentences"]
    groups = {(a["verdict"], a.get("cause")) for a in doc}; human = {(a["verdict"], a.get("cause")) for a in doc if a.get("judge") == "human"}
    assert groups == human, f"{len(groups - human)} of {len(groups)} groups hold no verdict a person gave, e.g. {sorted(groups - human, key=str)[0]}"

# ---------------------------------------------------------------- round 7: out of sample
HELD_OUT = ("he went home .", "he can come .", "the train arrived .", "did you read a book ?")     # never among the sentences the small cell is compiled from

def heldout_misfires(root):
    """the lines of dev the small cell was not compiled from on which it prints what the rule transducer does not print: another
    translation, or a translation of a sentence the rules refuse"""
    if "heldout" not in _cache:
        comp, mp, sents = small_cell(root); bad = []
        for line, en, ja in H.load_indexed("dev")[60:]:
            toks = R.merge_multiword(en)
            if toks in sents: continue
            try: want = R.translate(en)
            except PR.ParseFail: want = ""
            try: got = mp.translate(toks)
            except PR.ParseFail: got = ""
            if got and got != want: bad.append((" ".join(en), want or "(refused)", got))
        _cache["heldout"] = bad
    return _cache["heldout"]

def induced_in(root, methods):
    """the rules of the small cell induced from traces at a control point inside one of these methods of the rule parser"""
    comp, mp, sents = small_cell(root)
    assert not any(R.merge_multiword(s.split()) in sents for s in HELD_OUT), "a held-out sentence is among those the small cell is compiled from"
    return [prev for prev, lits, out in comp["encoder"]["rules"] if out[0] != PM.FAIL and isinstance(prev[0], tuple) and prev[0] and prev[0][-1] in methods]

def refused_for(reason, sentences):
    """the sentences the rule parser refuses for this reason"""
    bad = []
    for s in sentences:
        try: R.translate(s.split())
        except PR.ParseFail as e:
            if reason in str(e): bad.append((s, str(e)))
    return bad

def test_held_out_split_is_measured(root):
    """lines of train — sentences no rule, no table row and no trace of the cell was written from — are pinned in the corpus lock, none
    of them a line of dev or test; an action runs the transducer and the cell on them, every line is an item on the matrix chain, the
    cell against the rules, and the baseline holds what both printed"""
    held = ST.HeldOut(); lines = held.lines(); levels = held.levels()
    lock = json.load(open(os.path.join(root, "harness", "corpus.lock.json"), encoding="utf-8"))
    seen = {" ".join(en) for split in ("dev", "test") for line, en, ja in H.load_indexed(split)}
    assert {"train.en", "train.ja"} <= set(lock["files"]), f"the corpus lock pins {sorted(lock['files'])} and no line of train"
    assert lines and not seen & set(lines), "no held-out lines, or lines of dev and test among them"
    assert set(levels) == set(lines) and set(levels.values()) <= set(LAT.CHAINS["matrix"]), "a held-out line with no level on the matrix chain"
    assert baseline(root)["outputs"].get("heldout"), "the baseline holds no held-out outputs"

def test_cell_on_a_control_is_an_item(root):
    """the cell on a grammatical control is an item on the matrix chain — AGREE when it prints what the rules print, GAP when it
    refuses, MISFIRE when it prints anything else — so a control the cell translates differently is counted and the gate holds it"""
    rows = baseline(root)["outputs"]["negatives"]
    want = {r["control"]: "AGREE" if r.get("control.matrix") == r["control.rules"] else ("MISFIRE" if r.get("control.matrix") else "GAP") for r in rows}
    L = LAT.levels({}, {}, rows, 0, {}, {}, {})
    got = {c: L.get(f"control.matrix {c}", (None, None, None))[1] for c in want}
    differ = sorted(c for c, lv in want.items() if lv == "MISFIRE"); refused = sorted(c for c, lv in want.items() if lv == "GAP")
    assert got == want, (f"the lattice holds no item for the cell on a control: {len(differ)} the cell translates differently from the rules "
                         f"({', '.join(differ)}) and {len(refused)} it refuses ({', '.join(refused)}) are counted nowhere")
    assert LAT.cmp("control.matrix", "MISFIRE", "GAP") == "up" and LAT.TOP["control.matrix"] == "AGREE"

def test_frontier_names_every_blocker(root):
    """every sentence the cell or the printing rules refuse where the rules translate carries, in the baseline, the reason it was
    refused — outputs.gaps.<matrix|printer>.<split citation> — and every refusal of the rule parser names what it stopped on beside
    where (`<phase>: <why> at <the class of the token>`), so no group of the frontier is a bare `matrix gap`, `printer gap` or `VP0: tail`"""
    outs = baseline(root)["outputs"]; why = outs.get("gaps", {})
    unnamed = [(kind, f"{split} {key}") for kind in ("matrix", "printer") for split in ("dev", "test") for key, out in outs["rules"][split].items()
               if out and key in outs[kind][split] and not outs[kind][split][key] and not why.get(kind, {}).get(f"{split} {key}")]
    bare = [f"{split} {key}: {reason}" for split in ("dev", "test") for key, reason in outs["refusals"][split].items() if " at " not in reason]
    assert not unnamed and not bare, (f"{len(unnamed)} gaps of the cell and the printing rules carry no reason, e.g. {unnamed[:1]}; "
                                      f"{len(bare)} refusals name where the parser stopped and not what it stopped on, e.g. {bare[:1]}")

def test_comment_pins_resolve(root):
    """every pin a comment or a docstring of the code names — a test, a check, an action in backticks, a suite — exists, and a static
    check of the harness says so on every commit"""
    import re
    C = load_module(os.path.join(root, "harness", "check_todo.py"), "check_todo_for_pins")
    pins = C.pins_available(root) | {n for n in globals() if n.startswith("test_")}
    suites = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(root, "harness", "suites", "*.json"))}
    dangling = []
    for path in sorted(glob.glob(os.path.join(root, "*.py")) + glob.glob(os.path.join(root, "harness", "*.py")) + glob.glob(os.path.join(root, "tables", "*.tbl"))):
        for name in re.findall(r"pinned by[\s#]+(?:the[\s#]+)?(test_\w+|check_\w+|`\w+`|suite [A-Z])", open(path, encoding="utf-8").read()):
            if not (name[6:] in suites if name.startswith("suite ") else name.strip("`") in pins): dangling.append((os.path.relpath(path, root), name))
    checked = '"harness/check_comments.py"' in open(os.path.join(root, "harness", "run.py"), encoding="utf-8").read()
    assert not dangling and checked, (f"{len(dangling)} pin(s) named in a comment that no test, action, check or suite answers to, e.g. {dangling[:1]}; "
                                      + ("a static check reads the comments" if checked else "no static check of the harness reads the comments"))

def test_table_rows_are_reachable(root):
    """no row of a table keys on a word the lexicon does not hold as the part of speech the row needs, and the integrity file declares
    every such reference an error, so a row no sentence can reach cannot be committed"""
    CS = load_module(os.path.join(root, "harness", "check_schemas.py"), "check_schemas_for_acceptance")
    errors, warnings, rows = CS.check_tables(root)
    soft = [f"{r['from']['table']}.{r['from']['field']}" for r in json.load(open(os.path.join(root, "schemas", "tables", "integrity.json"), encoding="utf-8"))["references"] if r["level"] != "error"]
    assert not errors and not warnings and not soft, (f"{len(soft)} reference(s) declared at the level of a warning ({', '.join(soft)}); {len(errors + warnings)} of the references hold "
                                                      f"rows that key on a word the lexicon lacks, e.g. {(errors + warnings)[:1]}")

def test_verb_phrase_by_hand(root):
    """verb_phrase is a module written by hand: a sentence the cell was not compiled from — `he went home .` — gets the verb phrase the
    rules give it, the module's rules exist, and no rule induced from traces is left at a control point inside the method"""
    rules = ST.VerbPhraseModule().rules(); induced = induced_in(root, ("verb_phrase",)); bad = differing(root, ["he went home ."])
    assert not bad and not induced and rules, (f"a sentence the cell was not compiled from differs from the rules: {bad[:1]}; {len(induced)} rules induced from traces at "
                                               f"control points inside verb_phrase; {'a' if rules else 'no'} module written by hand")

def test_sentence_start_by_hand(root):
    """pre, after_wh and question_inversion are a module written by hand: a question that opens as no sentence the cell was compiled from
    does is not refused at its first token, the module's rules exist, and no rule induced from traces is left inside the three methods"""
    rules = ST.SentenceStartModule().rules(); induced = induced_in(root, ("pre", "after_wh", "question_inversion"))
    comp, mp, sents = small_cell(root); stopped = []
    for s in ("did you read a book ?",):
        try: mp.translate(R.merge_multiword(s.split()))
        except PR.ParseFail as e:
            if e.pos == 0: stopped.append((s, str(e)))
    assert not stopped and not induced and rules, (f"refused at the first token: {stopped[:1]}; {len(induced)} rules induced from traces at control points inside pre, after_wh "
                                                   f"and question_inversion; {'a' if rules else 'no'} module written by hand")

def test_no_rule_is_induced(root):
    """what the cell does with a sentence does not depend on the sentences it was compiled from: on the lines of dev the small cell was
    not compiled from it prints what the rules print or refuses, the simple sentences whose path no trace took are translated — by the
    small cell and by the committed one — and no row of the compiler comes from a trace"""
    comp, mp, sents = small_cell(root); bad = heldout_misfires(root); gaps = differing(root, ["he can come .", "the train arrived ."])
    committed = {r["control"]: r.get("control.matrix") for r in baseline(root)["outputs"]["negatives"]}
    refused = [c for c in ("he can come .", "the train arrived .") if not committed[c]]
    assert not any(R.merge_multiword(s.split()) in sents for s in HELD_OUT), "a held-out sentence is among those the small cell is compiled from"
    assert not bad and not gaps and not refused and not comp["rows"], (
        f"{len(bad)} lines of dev the small cell was not compiled from are translated differently from the rules, e.g. {bad[:1]}; it differs on {len(gaps)} simple "
        f"sentences whose path no trace took, e.g. {gaps[:1]}; the committed cell refuses {refused}; {len(comp['rows'])} rows of the compiler come from traces")

def test_embedded_questions_agree(root):
    """an embedded question is a frame of its own, linked from the frame of the verb before it and printed in plain form with か,
    かどうか or べきか"""
    rules = ST.EmbeddedQuestion().link_rules()
    translates(root, ["do you know how to drive a car ?", "i don 't care whether he agrees or not .", "please tell me where the bus stop is .", "i wonder what he bought ."])
    assert rules, "no rules at the opening of an embedded question"

def test_unnamed_folds_agree(root):
    """a clause after a noun, how about, the result clause after so, make it a rule to and a perception verb's clause are frames the
    cell reads and prints"""
    translates(root, ["you have no obligation to help us .", "my father boasts of the fact that he has never had a traffic accident .",
                      "how about going to the movies tonight ?", "the mystery story is so hard that i don 't like it .",
                      "i make it a rule to get up at six every morning .", "i can hear someone talking in the next room ."])

def test_transducer_is_total(root):
    """translate() returns a string or raises a refusal, whatever the sentence: none of these, each of which raises another exception
    today, and generate() raises on no frame of the printer's sweep"""
    crashed = []
    for s in ("the story appears to be true .", "he used to be a nice boy .", "i want to know about this mountain .", "please let me know about it .",
              "don 't have me on !", "let it be on your head !", "i want the concert to be a success at any cost ."):
        try: R.translate(s.split())
        except PR.ParseFail: pass
        except Exception as e: crashed.append(f"{s} ({type(e).__name__})")
    sweep = baseline(root)["detail"]["printer"]["sweep"]
    assert not crashed and sweep["generate_crash"] == 0, (f"{len(crashed)} sentences raise an exception that is not a refusal: {'; '.join(crashed)}; "
                                                          f"generate() raises on {sweep['generate_crash']} frames of the sweep, which skips them")

def test_passive_voice_right(root): judged_right(root, ["the tower can be seen from here .", "the water was cut off yesterday .", "this box is made of paper ."])

def test_bare_time_phrases_right(root):
    """a time phrase with no preposition is read as when: not as the object of a verb that has none, and not refused after an object"""
    objects = []
    for s, wrong in (("he came last month .", "最後"), ("he came every night .", "毎夜を")):
        try: out = R.translate(s.split())
        except PR.ParseFail as e: out = f"{wrong} (refused: {e})"
        if wrong in out: objects.append((s, out))
    bad = not_right(root, ["he will be playing tennis tomorrow afternoon .", "i want to come here next winter again ."])
    assert not objects and not bad, f"{len(objects)} time phrases read as an object, e.g. {objects[:1]}; {len(bad)} sentences with one after the object are not right, e.g. {bad[:1]}"

def test_sentence_initial_adverbs_right(root):
    """the adverb and the conjunction the parser reads after the subject, or does without, it reads before the subject too"""
    assert R.translate("i sometimes run out of money .".split()) and R.translate("his name is slightly familiar to me .".split())
    opening = refused_for("PRE: noun phrase", ["sometimes i run out of money .", "but his name is slightly familiar to me .", "and i will raise it again in three days ."])
    bad = not_right(root, ["sometimes i run out of money .", "but his name is slightly familiar to me ."])
    assert not opening and not bad, f"{len(opening)} sentences refused at the word they open with: {opening}; {len(bad)} are not right"

def test_missing_readings_right(root):
    """the lexicon holds the noun a sentence uses: love, meaning and building are not verbs alone, and their sentences are right"""
    verbs = [w for w in ("love", "meaning", "building") if {cls for cls, payload in R.classify(w)} == {"verb"}]
    bad = not_right(root, ["his love grew cold .", "the meaning of the words is intelligible .", "the fire destroyed the tall building ."])
    assert not verbs and not bad, f"the lexicon holds {', '.join(verbs) or 'none of them'} as a verb alone; {len(bad)} of their sentences are not right: {bad}"

def test_possessive_own_right(root):
    """own after a possessive is read, before a noun and at the end of an of-phrase"""
    bad = refused_for("possessive pronoun without noun", ["we want a house of our own .", "show your own business .", "this is a problem of your own making ."])
    assert not bad, f"{len(bad)} possessives with own refused: {bad}"
    judged_right(root, ["we want a house of our own ."])

def test_subject_case_guard_spares_one(root):
    """the refusal for a subject's case fires on a pronoun in subject position and on nothing else"""
    bad = refused_for("subject case", ["one day the men got to his village .", "one morning he went for a walk .", "do one thing at a time .", "no one is coming ."])
    kept = refused_for("subject case", ["me went to the store .", "himself saw john ."])
    assert not bad and len(kept) == 2, f"{len(bad)} grammatical sentences refused for the case of a subject: {bad}; {len(kept)} of 2 object pronouns as subjects refused for it"
    judged_right(root, ["no one is coming ."])

def test_experiential_perfect_right(root):
    judged_right(root, ["he has been to london three times .", "i have been to the station .", "i 've never been to such a nice party ."])

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
