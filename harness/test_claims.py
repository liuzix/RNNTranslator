"""The tests the README's claims are pinned to (see the README's appendix and harness/check_readme.py).

Each test_* function asserts one claim against the live tree.  Run as a script: every test runs, failures are listed,
exit 1 on any failure.  Claims about corpus-level numbers are pinned to the actions instead (rules, matrix, np), whose
results the README renders from harness/baseline.json.

    uv run python harness/test_claims.py [--root DIR]
"""
import sys, os, re, json, ast, glob, tarfile, argparse, subprocess, tempfile, shutil, collections

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
    global R, P, PR, PM, NPM, H, GM
    import rules_ja as R, parser as P, parser_rules as PR, parser_matrix as PM, np_module as NPM, run_rules as H, gen_module as GM

def small_cell(root):
    """the hybrid cell compiled on a few dev sentences (seconds), shared by the cell tests"""
    if "comp" not in _cell:
        sents = [R.merge_multiword(en) for en, ja in H.load("dev")[:60]] + [R.merge_multiword(s.split()) for s in ("she took a rest .", "i like the big dog .", "he said that he was tired .")]
        _cell["comp"] = PM.build_hybrid(sents); _cell["mp"] = PM.MatrixParser(_cell["comp"])
    return _cell["comp"], _cell["mp"]

# ---------------------------------------------------------------- the circuit
def test_cell_weights_are_named_integers(root):
    """every nonzero of every layer is an integer weight between two named units, not a learned float"""
    comp, mp = small_cell(root)
    import numpy as np
    for (L, W, b), (M, bias) in zip(comp["layers"], mp.mats):
        assert np.all(np.round(M.data) == M.data), "non-integer weight"
        assert set(np.unique(M.data)) <= {-5.0, -1.0, 1.0}, f"weights {set(np.unique(M.data))}"   # -5: a literal the cell cannot see silences its rule
        assert all(float(v).is_integer() for v in b.values()), "a non-integer threshold"
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
    kinds = {"module": 0, "outer": 0, "print": 0, "clause": 0, "flush": 0}
    for prev, lits, out in comp["rules"]:
        if prev[0] == "P" or prev == ("DONE",): kinds["print"] += 1
        elif prev[0] in ("OPEN", "CLOSE", "RESTORE"): kinds["flush"] += 1
        elif prev[0] in ("M", "RETM") or (prev[0] == "RET" and out[0][0] == "M"): kinds["clause"] += 1
        elif isinstance(prev[0], str) and prev[0].startswith("NP"): kinds["module"] += 1
        elif not (prev[0] == "RET" or out[0][0] == "NP0" or out[0] == PM.FAIL): kinds["outer"] += 1
    assert kinds["module"] > 0 and kinds["outer"] > 0 and kinds["print"] > 0 and kinds["clause"] > 0 and kinds["flush"] > 0
    assert ("shift",) in comp["S"] and ("emit",) in comp["S"]
    assert {n[0] for n in comp["S"].names} == {"ctrl", "slot", "pend", "shift", "emit", "out"}, "the state is not the six kinds of block"
    assert all(("ctrl", prev) in comp["S"] for prev, _, _ in comp["rules"] if prev[0] == "P"), "a printing position outside the control block"
    assert all(n[0] in ("pred", "src", "prep0", "one") for n in comp["X"].names), "an input unit that is not a reading of the window"
    assert all(w == 1.0 for _, _, w in comp["W6"]), "the readout is not 0/1 sums"

def test_slot_write_circuit(root):
    """one slot write as docs/cell.puml draws it: a block gate in L4 that is an OR of rule units; a source gate in L4; write
    units in L5 = ReLU(source gate + source value - 1), one per value; keep units in L5 = ReLU(old value - block gate); and
    W6 sums keep + write back into the same one-hot block of the next state"""
    comp, mp = small_cell(root)
    S, X = comp["S"], comp["X"]; layers = comp["layers"]
    L3, L4, L5 = layers[2][0], layers[3][0], layers[4][0]
    W4, b4 = layers[3][1], layers[3][2]; W5, b5 = layers[4][1], layers[4][2]
    into4 = collections.defaultdict(list); into5 = collections.defaultdict(list)
    for u, src, w in W4: into4[u].append((src, w))
    for u, src, w in W5: into5[u].append((src, w))
    # L4: every block gate and source gate is an OR of L3 rule units, weights +1, no bias
    gates = [i for i, n in enumerate(L4.names) if n[0] in ("gate", "gcopy") and n[1] != "pend"]   # the pend gate is an AND with the window, drawn elsewhere
    assert gates
    for g in gates:
        assert into4[g] and all(src[0] == "L3" and L3.names[src[1]][0] == "rule" and w == 1.0 for src, w in into4[g]) and b4.get(g, 0.0) == 0.0
    # L5: write units = AND(source gate, source value); keep units = old value - block gate
    writes = [(i, n) for i, n in enumerate(L5.names) if n[0] == "write" and n[1] != "pend"]
    keeps = [(i, n) for i, n in enumerate(L5.names) if n[0] == "keep" and n[1] != "pend"]
    assert writes and keeps
    for i, (_, slot, s, v) in writes:
        assert b5.get(i) == -1.0 and sorted(into5[i], key=str) == sorted([(("L4", L4[("gcopy", slot, s)]), 1.0), (("X", X[("src", s, v)]), 1.0)], key=str), f"write {slot} {s} {v}: {into5[i]}"
    gated = 0
    for i, (_, slot, v) in keeps:
        ins = dict((src, w) for src, w in into5[i])
        assert ins.pop(("S", S[("slot", slot, v)])) == 1.0 and b5.get(i, 0.0) == 0.0
        if ("gate", slot) in L4: assert ins.pop(("L4", L4[("gate", slot)])) == -1.0; gated += 1
        assert all(w == -1.0 and src[0] == "L4" for src, w in ins.items()), f"keep {slot} {v}: {ins}"   # the USE gates also cut what the lookup rewrites
    assert gated > 0
    # W6: keep and write of the same (slot, value) both add +1 into that unit of the next state, and nothing else does
    into6 = collections.defaultdict(list)
    for tgt, src, w in comp["W6"]: into6[tgt].append((src, w))
    for i, (_, slot, s, v) in writes: assert (("L5", i), 1.0) in into6[S[("slot", slot, v)]]
    for i, (_, slot, v) in keeps: assert (("L5", i), 1.0) in into6[S[("slot", slot, v)]]
    for tgt, srcs in into6.items():
        if S.names[tgt][0] == "slot": assert all(w == 1.0 and src[0] == "L5" and L5.names[src[1]][0] in ("keep", "write", "writec", "writem", "writeu") for src, w in srcs)

def test_print_rules_normal_form(root):
    """every printing rule is (position, literals, (next position, token)): the literals are base predicates of the flat frame or
    derived predicates whose DNF is over base literals, the tokens are of the four kinds, every next position is a position, a
    terminal (DONE; OPEN and RESTORE at a that-clause's boundary) or FAIL, and the rules reach the sentence end from P0"""
    import gen_module as G
    pr = G.PrintRules(); D = pr.derived(); positions = set(pr.positions()); F = G.flatten(PR.parse_bounded(R.merge_multiword("i like the big dog .".split())))
    assert pr.positions()[0] == "P0" and "DONE" not in positions
    def check_base(name):
        G.base_pred(name, F)                                                      # KeyError if it is not a predicate of the flat frame
    for name, conjs in D.items():
        assert name.startswith("d.") and conjs
        for conj in conjs:
            for l in conj: assert not l.lstrip("¬").startswith("d."), f"{name} nests a derived predicate"; check_base(l.lstrip("¬"))
    reach = {"P0"}; frontier = ["P0"]; nxts = set()
    while frontier:
        p = frontier.pop()
        for pos, lits, (nxt, tok) in pr.rules_for(p):
            assert pos == p and isinstance(lits, tuple)
            for l in lits:
                q = l.lstrip("¬")
                if q.startswith("d."): assert q in D
                else: check_base(q)
            assert tok is None or tok[0] in ("c", "s", "vf", "af"), tok
            nxts.add(nxt)
            if nxt not in reach and not G.is_terminal(nxt): reach.add(nxt); frontier.append(nxt)
    assert {"DONE", "OPEN", "RESTORE"} <= nxts and reach == positions, f"{len(positions - reach)} positions unreachable from P0, {len(reach - positions)} targets without rules"
    assert "".join(pr.render(F)) == R.generate(PR.parse_bounded(R.merge_multiword("i like the big dog .".split())))

def test_decoder_is_the_cell_run_backwards(root):
    """the printing rules are rules of the same cell: their positions are control states of the one control block, entered from
    DONE at the end of reading; the frame the read head leaves is the frame the print head prints, and the print head's string
    is what the rule transducer says on sentences the cell parses"""
    comp, mp = small_cell(root); S = comp["S"]
    prints = [r for r in comp["rules"] if r[0][0] == "P"]
    assert prints and (("DONE",), (), (("P", "P0"), 0, ())) in comp["rules"]
    assert all(("ctrl", r[0]) in S for r in prints) and ("ctrl", ("P", "DONE")) in S and ("ctrl", ("DONE",)) in S
    for s in ("i like the big dog .", "she took a rest ."):
        toks = R.merge_multiword(s.split())
        assert mp.translate(toks) == R.translate(s.split())
        frame, units = mp.run(toks)
        assert "".join(PM.output_lexicon(u) for u in units) == R.generate(PM.frame_to_clause(frame)) and len(units) > 3
    assert "".join(PM.output_lexicon(u) for u in mp.decode(PR.parse_bounded(R.merge_multiword("i like the big dog .".split())))) == "私は大きい犬が好きだ。"

def test_decoder_mirrors_encoder(root):
    """emit is to the printing rules what shift is to the consuming rules; a read gate is an OR of the rules that print a block and a
    read unit is AND(read gate, value) as a write unit is AND(source gate, source value); a lookup unit is AND(form gate, verb);
    a read with nothing in the block, or a form the verb has not, fires into FAIL as a copy with no source does"""
    comp, mp = small_cell(root)
    S = comp["S"]; L3, L4, L5 = comp["layers"][2][0], comp["layers"][3][0], comp["layers"][4][0]
    W4, b4 = comp["layers"][3][1], comp["layers"][3][2]; W5, b5 = comp["layers"][4][1], comp["layers"][4][2]
    into4 = collections.defaultdict(list); into5 = collections.defaultdict(list)
    for u, src, w in W4: into4[u].append((src, w))
    for u, src, w in W5: into5[u].append((src, w))
    emit, shift = L4[("emit",)], L4[("shift",)]
    rule_of = lambda src: comp["rules"][L3.names[src[1]][1]]
    assert into4[emit] and all(src[0] == "L3" and L3.names[src[1]][0] == "rule" and w == 1.0 and rule_of(src)[0][0] == "P" for src, w in into4[emit]) and b4.get(emit, 0.0) == 0.0
    assert all(rule_of(src)[2][1] >= 1 for src, w in into4[shift] if L3.names[src[1]][0] == "rule")      # shift: the rules that consume; emit: the rules that print
    gates = [i for i, n in enumerate(L4.names) if n[0] in ("gread", "gform", "gaform")]
    assert gates and all(into4[g] and all(src[0] == "L3" and w == 1.0 and rule_of(src)[0][0] == "P" for src, w in into4[g]) and b4.get(g, 0.0) == 0.0 for g in gates)
    reads = [(i, n) for i, n in enumerate(L5.names) if n[0] == "read"]; forms = [(i, n) for i, n in enumerate(L5.names) if n[0] == "form"]
    assert reads and forms
    for i, (_, slot, idx, v) in reads:
        assert b5.get(i) == -1.0 and sorted(into5[i], key=str) == sorted([(("L4", L4[("gread", slot, idx)]), 1.0), (("S", S[("slot", slot, v)]), 1.0)], key=str)
    for i, (_, name, alt, v) in forms:
        assert b5.get(i) == -1.0 and sorted(into5[i], key=str) == sorted([(("L4", L4[("gform", name, alt)]), 1.0), (("S", S[("slot", "verb", v)]), 1.0)], key=str)
    into6 = collections.defaultdict(list)
    for tgt, src, w in comp["W6"]: into6[tgt].append((src, w))
    for i, (_, slot, idx, v) in reads: assert into6[S[("out", "s", slot, idx, v)]] == [(("L5", i), 1.0)]
    crashes = [i for i, n in enumerate(L5.names) if n[0] in ("noread", "noform", "noaform")]
    assert crashes and all((("L5", i), 1.0) in into6[S[("ctrl", PM.FAIL)]] for i in crashes)
    for i in crashes:
        ins = dict(into5[i]); assert sum(1 for src, w in ins.items() if w == 1.0 and src[0] == "L4") == 1 and all(w == -1.0 for src, w in ins.items() if src[0] == "S")

def test_output_is_one_hot_morphemes(root):
    """y(t): at every printing step exactly one output unit is on when emit is, none when it is not; every output unit is a constant,
    a slot's value, or a form of a verb or an adjective, and the output lexical layer gives each a string"""
    comp, mp = small_cell(root); S = comp["S"]
    assert {n[1] for n in S.names if n[0] == "out"} == {"c", "s", "vf", "af"}
    for n in S.names:
        if n[0] == "out": assert isinstance(PM.output_lexicon(n), str)
    toks = R.merge_multiword("she took a rest .".split())
    s, frame, out = mp.read(toks); steps = 0; emitted = []
    assert out == []                                                              # no that-clause: the print head emits nothing while reading
    while True:
        ctrl, _ = mp.read_state(s)
        if ctrl == ("P", "DONE"): break
        s = mp.step(s, mp.empty_input()); steps += 1
        on = [S.names[i] for i in mp.out_units if s[i] > 0.5]
        assert len(on) == (1 if s[mp.emit_unit] > 0.5 else 0), (ctrl, len(on))
        emitted += on
    assert emitted == mp.run(toks)[1] and steps > len(emitted) and "".join(PM.output_lexicon(u) for u in emitted) == "彼女は休んだ。"

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

def test_verb_hits_are_layer_zero(root):
    """the table rows that join the frame's verb with the window — the idiom tails, the particle after a verb, the use nouns — are
    units of the cell: one AND unit per row at layer 1 over the verb's unit and the window's table keys (threshold the number of
    keys), ORed at layer 2 into the hit predicates the rules read; the input carries no hit the verb decides, so the runner hands
    the cell the window alone and the cell still reads `she took a rest` as the rules do"""
    comp, mp = small_cell(root); S, X = comp["S"], comp["X"]; L1, L2 = comp["layers"][0][0], comp["layers"][1][0]
    rows = [n for n in L1.names if n[0] == "hitrow"]; hits = {n[1] for n in L2.names if n[0] == "pred" and PM.verb_hit(n[1])}
    assert rows and "hit=use" in hits and not any(n[0] == "pred" and PM.verb_hit(n[1]) for n in X.names)
    assert set(rows) <= set(PM.Layer0Lexical().hit_units(comp))
    W1, b1 = comp["layers"][0][1], comp["layers"][0][2]; W2 = comp["layers"][1][1]
    into = collections.defaultdict(list)
    for u, ref, w in W1: into[u].append((ref, w))
    for n in rows:
        refs = into[L1[n]]
        assert all(w == 1.0 for ref, w in refs) and sum(1 for ref, w in refs if ref[0] == "S" and S.names[ref[1]][:2] == ("slot", "verb")) == 1
        keys = [ref for ref, w in refs if ref[0] == "X"]
        assert keys and b1[L1[n]] == -len(keys) and all(X.names[j][1].split("=")[0].endswith((".tbl", ".prole")) for _, j in keys)
    rows_of = collections.defaultdict(set)
    for u, ref, w in W2:
        if L2.names[u][0] == "pred" and PM.verb_hit(L2.names[u][1]): assert ref[0] == "L1" and L1.names[ref[1]][0] == "hitrow" and w == 1.0; rows_of[L2.names[u][1]].add(ref[1])
    assert rows_of["hit=use"] == {L1[n] for n in rows if n[1] == "hit=use"}
    assert any(n[0] == "pred" and n[1] == "w0.tbl=rest" for n in X.names)
    saved = PM.table_hits
    try:
        PM.table_hits = lambda toks, i, verb: []
        assert mp.translate(R.merge_multiword("she took a rest .".split())) == "彼女は休んだ。"
    finally: PM.table_hits = saved

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

def test_clause_modules_replace_extraction(root):
    """the PP machine, predicate and copula_predicate are hand-translated modules spliced into the cell: no extracted rule is left
    whose control point lies in those methods, each module's positions decide completely (every position has an otherwise-rule
    whose literals are the negations of its branches' conditions), and the cell translates the small cell's in-scope sentences
    as the rule transducer does"""
    import pp_module as PPM, predicate_module as PDM
    comp, mp = small_cell(root)
    for prev, lits, out in comp["rules"]:
        for cp in (prev, out[0]):
            if isinstance(cp, tuple) and cp and isinstance(cp[0], tuple):
                assert cp[0][-1] not in PM.MODULE_METHODS, f"an extracted rule at {cp}"
    hm = PM.HandModules(); modules = {"pp": hm.pp_rules(), "pred": hm.predicate_rules(), "cop": hm.copula_rules()}
    assert all(len(rs) > 20 for rs in modules.values())
    for mod, rs in modules.items():
        by_state = collections.defaultdict(list)
        for st, lits, out in rs: by_state[st].append((lits, out))
        assert ("M", mod, PM.MODULE_START[mod][2]) in {("M", mod, st) for st in by_state if isinstance(st, str)}
        for st, rules in by_state.items():
            if not isinstance(st, str) or not any(l.startswith(f"d.{mod}.{st}.") for lits, _ in rules for l in lits): continue
            conds = sorted({l.lstrip("¬") for lits, _ in rules for l in lits if l.startswith(f"d.{mod}.{st}.")})
            assert any(set(lits) == {"¬" + c for c in conds} for lits, _ in rules), f"{mod} {st}: no otherwise-rule"
    assert any(out[0] == PM.FAIL and prev[0] == "M" for prev, lits, out in comp["rules"]), "no refusal in the modules"
    for s in ("i like the big dog .", "she took a rest ."):
        toks = R.merge_multiword(s.split()); assert mp.translate(toks) == R.translate(s.split())

def test_inner_clause_in_one_frame(root):
    """a that-clause is read in the one frame.  At the boundary the print head emits the outer prefix (the run of the printing rules in
    mode prefix ends at OPEN); OPEN moves the outer predicate into the tag slots — exactly the registers the suffix positions read,
    the negation the printed arguments carried folded into tag.neg — and clears every other slot by its own gate; the clause is read
    into the cleared slots; CLOSE flushes it (printed in mode embedded, が on its subject, then と) and RESTORE clears it, moves the
    tags back and leaves the print head at DONE in mode suffix for the outer predicate.  OPEN refuses a frame whose tags are in use
    (a that-clause inside a that-clause) or whose predicate the tags cannot hold.  The cell's string is the rule transducer's"""
    comp, mp = small_cell(root); S = comp["S"]; V = comp["slot_vocab"]
    cf = PM.ClauseFlush(); tags = cf.tag_slots()
    assert tags == [f"tag.{k}" for k in GM.TAG_KEYS] and all(any(n[0] == "slot" and n[1] == t for n in S.names) for t in ("tag.verb", "tag.past", "tag.neg", "tag.question"))
    assert all(("slot", "pmode", m) in S for m in GM.PMODES)
    # the suffix positions read tag slots, the print mode, the negative-polarity sources folded into tag.neg, or slots of a predicate OPEN refuses
    by_pos = collections.defaultdict(list)
    for r in GM.PrintRules().rules(): by_pos[r[0]].append(r)
    seen = {"M_TAIL"}; frontier = ["M_TAIL"]
    while frontier:
        for pos, lits, (nxt, tok) in by_pos.get(frontier.pop(), []):
            if nxt not in seen and not GM.is_terminal(nxt): seen.add(nxt); frontier.append(nxt)
    D = GM.derived()
    def names(l):
        q = l.lstrip("¬")
        return [x for conj in D[q] for c in conj for x in names(c)] if q.startswith("d.") else [q]
    def slot_of(q):
        if q.startswith("flag."): return q[5:]
        if q.startswith("slot."): return q[5:].split("=")[0]
        return GM.family_of(q)[0]
    read = {slot_of(q) for pos in seen for _, lits, _ in by_pos.get(pos, []) for l in lits for q in names(l)}
    folded = {"adv1", "adv2", "subj.noun", "obj.noun"}                               # d.neg: never, nobody, nothing -> tag.neg; want's object (refused)
    refused = {"wh", "there", "subj.cls", "cmp.noun"} | {sl for sl in read if sl.startswith(("cop.", "copof."))}       # whcop, there, いる/ある, the stem's より, the copula noun phrase
    assert read <= set(GM.TAG_KEYS) | {"pmode"} | folded | refused, read - set(GM.TAG_KEYS) - {"pmode"} - folded - refused
    # OPEN: a move for every tag, a clear for every other slot of the cell, tag.neg from d.neg; RESTORE: the tags back, the rest cleared
    opens = [r for r in comp["rules"] if r[0] == ("OPEN",)]; restore = [r for r in comp["rules"] if r[0] == ("RESTORE",)]
    assert len(restore) == 1 and len(opens) >= 2 and any(out[0] == PM.FAIL for _, _, out in opens)
    into_inner = [r for r in opens if r[2][0] == ("INNER",)]
    assert len(into_inner) == 2 and {dict(lits).get("d.neg") for _, lits, _ in into_inner} == {True, False}
    for prev, lits, (cp, k, acts) in into_inner:
        moved = {slot: src for slot, v, src in acts if src[0] == "move"}; cleared = {slot for slot, v, src in acts if src[0] == "clear"}
        assert all(moved.get(f"tag.{key}") == ("move", key) for key in GM.TAG_KEYS if key != "neg")
        assert (("tag.neg", True, ("const",)) in acts) == dict(lits)["d.neg"]
        assert cleared >= {sl for sl in V if V[sl] and not sl.startswith("tag.") and sl != "pmode" and sl != "neg"}, "a slot OPEN does not clear"
        assert all(dict(lits).get(q) is False for q in ("slot.tag.verb", "slot.tag.cop_kind") + GM.OPEN_REFUSES)
    acts = restore[0][2][2]
    assert all((key, None, ("move", f"tag.{key}")) in acts for key in GM.TAG_KEYS) and ("pmode", "suffix", ("const",)) in acts
    assert {sl for sl, v, src in acts if src[0] == "clear"} >= {sl for sl in V if V[sl] and sl not in GM.TAG_KEYS and sl != "pmode"}
    assert (("CLOSE",), (("slot.pmode=prefix", True),), (("P", "P0"), 0, (("pmode", "embedded", ("const",)),))) in comp["rules"]
    assert (("CLOSE",), (("slot.pmode=prefix", False),), (("DONE",), 0, ())) in comp["rules"]
    # the sentence: prefix emitted while reading, the clause and と at its end, the outer predicate after DONE; the same string as the rules
    toks = R.merge_multiword("he said that he was tired .".split())
    s, frame, out = mp.read(toks)
    prefix_inner = "".join(PM.output_lexicon(u) for u in out); suffix = "".join(PM.output_lexicon(u) for u in mp.print_out(s))
    assert prefix_inner == "彼は彼が疲れていたと" and suffix == "言った。" and frame.get("pmode") == "suffix" and frame.get("verb") == ("say", "言う", "g") and "subj.noun" not in frame
    assert mp.translate(toks) == R.translate("he said that he was tired .".split()) == R.generate(mp.parse(toks))
    for s_ in ("he said that she thought that he was tired .", "i 'm sure that he will come ."):           # tags in use; a copula's that-clause
        try: mp.translate(R.merge_multiword(s_.split())); assert False, s_
        except PR.ParseFail: pass

def test_np_of_phrase(root):
    """an of-phrase is the noun-phrase module run once more on the role's of-phrase slots, printing before the head with の, and a
    second of-phrase inside it is refused; the frame marks a folded inner clause, which the cell does not hold"""
    toks = ["the", "name", "of", "that", "designer", "."]
    p = PR.Parser(list(toks), []); ja, cls = p.parse_np()
    out = NPM.run_module("obj", toks, 0, {"verb": None, "cop_kind": None, "subj": None}, None)
    assert out[0] == "RET" and out[1] == p.i == 5 and NPM.np_string(out[2], "obj") == ja == "そのデザイナーの名前" and out[2]["objof.det"] == "その" and out[2]["objof.noun"] == "デザイナー"
    nested = NPM.run_module("obj", ["the", "study", "of", "music", "of", "the", "world", "."], 0, {"verb": None, "cop_kind": None, "subj": None}, None)
    assert nested[0] == "FAIL" and nested[1] == "nested of-phrase"
    assert PR.parse_bounded(R.merge_multiword("it is a good idea to read this book .".split())).folded and not PR.parse_bounded(["i", "like", "dogs", "."]).folded
    assert "folded" in PM.UNSUPPORTED_REGS

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
               "np": {"mismatches": 0}, "negatives": {"accepted": 0, "controls_refused_by_rules": 0, "matrix_accepted": 0, "rows": negatives},
               "suites": {"classes": {}, "cases": 0, "right": 0, "closed": 0, "open": 0, "rows": []},
               "torch": {"sentences": 2, "steps": 8, "diverging": 0, "torch": "0", "dtype": "float32"},
               "printer": {"agree": 0, "gap": 0, "misfire": 0, "rules": 0, "positions": 0, "per_split": {}, "outputs": {"dev": {}, "test": {}},
                           "sweep": {"frames": 0, "match": 0, "mismatches": 0, "refused": 0, "generate_crash": 0}}}
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
    assert LT.CHAINS["torch.exact"] == ["DIVERGING", "EXACT"] and LT.TOP["torch.exact"] == "EXACT" and LT.cmp("torch.exact", "EXACT", "DIVERGING") == "down"
    empty = {"dev": {}, "test": {}}
    assert LT.levels({}, {}, [], 0, empty, {}, {}, [], torch_diverging=0)["torch.exact"] == ("torch.exact", "EXACT", 0)
    assert LT.levels({}, {}, [], 0, empty, {}, {}, [], torch_diverging=3)["torch.exact"][1] == "DIVERGING" and "torch.exact" not in LT.levels({}, {}, [], 0, empty, {}, {}, [])
    assert LT.CHAINS["printer"] == ["MISFIRE", "GAP", "AGREE"] and LT.CHAINS["printer.sweep"] == ["MISMATCHES", "MATCH"] and LT.cmp("printer", "GAP", "MISFIRE") == "down"
    ro = {"dev": {"1: a .": "あ。", "2: b .": "い。", "3: c .": "う。", "4: d .": ""}, "test": {}}; po = {"dev": {"1: a .": "あ。", "2: b .": "", "3: c .": "え。", "4: d .": ""}}
    Lp = LT.levels(ro, {}, [], 0, {"dev": {}, "test": {}}, {}, {}, [], printer_outputs=po, printer_sweep=0)
    assert [Lp[f"printer dev {k}"][1] for k in ("1: a .", "2: b .", "3: c .")] == ["AGREE", "GAP", "MISFIRE"] and "printer dev 4: d ." not in Lp and Lp["printer.sweep"][1] == "MATCH"

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
        assert not gate({k1: "私は猫が好きだ。", k2: "彼も来た。"}, {k1: "私は猫が好きだ。", k2: ""})                   # UNKNOWN -> another unjudged output needs verdicts
        assert not gate({k1: "私は猫が好きだ。", k2: "彼も来た。"}, {k1: "私は猫が好きだ。", k2: ""}, adj=[{"citation": "dev 2: he came .", "verdict": "wrong", "output": "彼が来た。", "reason": "wrong particle"}])   # the old one judged, the new one still unjudged
        assert gate({k1: "私は猫が好きだ。", k2: "彼も来た。"}, {k1: "私は猫が好きだ。", k2: ""}, adj=[{"citation": "dev 2: he came .", "verdict": "wrong", "output": "彼が来た。", "reason": "wrong particle"},
                                                                                              {"citation": "dev 2: he came .", "verdict": "right", "output": "彼も来た。", "reason": "も for the context"}])   # both judged: WRONG -> RIGHT is up
        assert not gate({k1: "私は猫が好きだ。", k2: "彼が来た。"}, {k1: "私は猫が好きだ。", k2: ""}, negatives=[dict(neg[0], **{"sentence.rules": "私は行く。"})])   # a negative accepted is down
    finally: shutil.rmtree(tmp)

def test_suites_are_items(root):
    """every suite case is a lattice item `suite <class> <sentence>` on WRONG < REFUSED < RIGHT, RIGHT exactly when the output is
    the expectation; a cited case's expectation adjudicates that output right for the corpus sentence; a case moving down is refused"""
    import importlib.util, io, contextlib
    spec = importlib.util.spec_from_file_location("lattice", os.path.join(root, "harness", "lattice.py")); LT = importlib.util.module_from_spec(spec); spec.loader.exec_module(LT)
    assert LT.CHAINS["suite"] == ["WRONG", "REFUSED", "RIGHT"] and LT.TOP["suite"] == "RIGHT"
    assert LT.cmp("suite", "REFUSED", "RIGHT") == "up" and LT.cmp("suite", "RIGHT", "WRONG") == "down" and LT.cmp("suite", "WRONG", "REFUSED") == "up"
    # every case of every suite file is an item, and the level is computed from output vs expectation
    import glob
    cases = [(json.load(open(p, encoding="utf-8")), p) for p in sorted(glob.glob(os.path.join(root, "harness", "suites", "*.json")))]
    assert cases, "no suites"
    rows = [{"class": s["class"], "sentence": c["sentence"], "expect": c["expect"], "output": c["expect"] if i % 3 == 0 else ("" if i % 3 == 1 else "x"),
             "level": ["RIGHT", "REFUSED", "WRONG"][i % 3]} for s, _ in cases for i, c in enumerate(s["cases"])]
    L = LT.levels({}, {}, [], 0, {"dev": {}, "test": {}}, {}, {}, rows)
    for r in rows: assert L[f"suite {r['class']} {r['sentence']}"] == ("suite", r["level"], r["output"])
    assert sum(1 for k in L if k.startswith("suite ")) == sum(len(s["cases"]) for s, _ in cases)
    # the expectation of a cited case is a right verdict for that corpus sentence
    tmp = tempfile.mkdtemp()
    try:
        sd = os.path.join(tmp, "suites"); os.makedirs(sd)
        json.dump({"class": "Z", "title": "t", "cases": [{"sentence": "he came .", "expect": "彼が来た。", "citation": "dev 2"}]}, open(os.path.join(sd, "Z.json"), "w", encoding="utf-8"), ensure_ascii=False)
        verdicts, _ = LT.load_adjudications(os.path.join(tmp, "none.json"), sd)
        assert verdicts["dev 2: he came ."] == {"彼が来た。": "right"}
        # the gate: a suite case RIGHT -> REFUSED or RIGHT -> WRONG is down; REFUSED -> RIGHT is up
        runner = _runner(root)
        k1, k2 = "1: i like cats .", "2: he came ."
        row = {"class": "Z", "sentence": "he came .", "expect": "彼が来た。", "output": "彼が来た。", "level": "RIGHT", "reason": None, "citation": "dev 2"}
        corpus, src, base = _fake_world(tmp, {k1: "私は猫が好きだ。", k2: "彼が来た。"}, {k1: "私は猫が好きだ。", k2: ""}, [])
        base["suites"] = {"classes": {"Z": {"title": "t", "cases": 1, "right": 1, "refused": 0, "wrong": 0, "closed": True}}, "cases": 1, "right": 1, "closed": 1, "open": 0, "rows": [row]}
        bp = os.path.join(src, "harness", "baseline.json"); json.dump(runner.as_document(base), open(bp, "w", encoding="utf-8"), ensure_ascii=False)
        def gate(level, output):
            _, src2, res = _fake_world(tmp, {k1: "私は猫が好きだ。", k2: "彼が来た。"}, {k1: "私は猫が好きだ。", k2: ""}, [])
            res["suites"] = dict(base["suites"], rows=[dict(row, level=level, output=output)])
            with contextlib.redirect_stdout(io.StringIO()): return runner.compare(res, bp, corpus, src2)
        assert gate("RIGHT", "彼が来た。") and not gate("REFUSED", "") and not gate("WRONG", "彼は来た。")
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
    pkgs = [b for b in text.split("[[package]]")[1:] if re.search(r"^source = \{ registry", b, re.M)]   # the block's own source line, not a dependency's
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
