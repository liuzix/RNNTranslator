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
    """the machine compiled on a few dev sentences (seconds), shared by the cell tests: -> (the compiled encoder and decoder, the runner)"""
    if "comp" not in _cell:
        extra = ("she took a rest .", "i like the big dog .", "he said that he was tired .", "he said that he was tired ?", "he said that she thought that he was tired .",
                 "when i was a child , i loved reading .", "i loved reading when i was a child .", "i went out although it was raining .", "i went out when it was raining ?",
                 "if it rains tomorrow , i will stay at home .", "unless it rains , i will go , too .", "speak clearly so that everyone may hear you .", "in fact , i like the big dog .",
                 "while walking , he sang a song .", "while having a car , he walks .", "he is reading a book .", "he can swim .")
        sents = [R.merge_multiword(en) for en, ja in H.load("dev")[:60]] + [R.merge_multiword(s.split()) for s in extra]
        _cell["comp"] = PM.build_hybrid(sents); _cell["mp"] = PM.MatrixParser(_cell["comp"])
    return _cell["comp"], _cell["mp"]

# ---------------------------------------------------------------- the circuit
def test_cell_weights_are_named_integers(root):
    """every nonzero of every layer of both cells is an integer weight between two named units, not a learned float"""
    comp, mp = small_cell(root)
    import numpy as np
    for cell in (mp.enc, mp.dec):
        for (L, W, b), (M, bias) in zip(cell.c["layers"], cell.mats):
            assert np.all(np.round(M.data) == M.data), "non-integer weight"
            assert set(np.unique(M.data)) <= {-5.0, -1.0, 1.0}, f"weights {set(np.unique(M.data))}"   # -5: a literal the cell cannot see silences its rule
            assert all(float(v).is_integer() for v in b.values()), "a non-integer threshold"
            assert M.shape[0] == len(L.names), "an unnamed unit"
        assert set(np.unique(cell.W6.data)) <= {1.0}, "the readout is not 0/1 sums"
        assert all(isinstance(n, tuple) for n in cell.S.names) and all(isinstance(n, tuple) for n in cell.X.names)

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
    """the machine is two cells of five layers each.  The encoder's state is the control point, the slots of the working frame, the
    pending preposition and the shift and flush units; its input is readings of the window; its rules are the noun-phrase module,
    the clause modules, the rules extracted from traces and the clause rules.  The decoder's state is the printing position, the
    sentence block, the emit, down and up units and the output morpheme; its input is one frame — unit for unit the encoder's
    slot block — and the constant; its rules are the printing rules"""
    comp, mp = small_cell(root); enc, dec = comp["encoder"], comp["decoder"]
    assert len(enc["layers"]) == 5 and len(dec["layers"]) == 5
    kinds = {"module": 0, "outer": 0, "clause": 0, "flush": 0}
    for prev, lits, out in enc["rules"]:
        assert prev[0] != "P", "a printing rule in the encoder"
        if prev[0] in ("OPEN", "TURN", "CLOSE"): kinds["flush"] += 1
        elif prev[0] in ("M", "RETM") or (prev[0] == "RET" and out[0][0] == "M"): kinds["clause"] += 1
        elif isinstance(prev[0], str) and prev[0].startswith("NP"): kinds["module"] += 1
        elif not (prev[0] == "RET" or out[0][0] == "NP0" or out[0] == PM.FAIL): kinds["outer"] += 1
    assert all(n > 0 for n in kinds.values()), kinds
    assert dec["rules"] and all(prev[0] == "P" for prev, _, _ in dec["rules"]), "a rule of the decoder that is not a printing rule"
    assert {n[0] for n in enc["S"].names} == {"ctrl", "slot", "pend", "shift", "flush"}, "the encoder's state is not its five kinds of block"
    assert {n[0] for n in dec["S"].names} == {"ctrl", "slot", "emit", "down", "up", "out"}, "the decoder's state is not its six kinds of block"
    assert all(n[0] in ("pred", "src", "prep0", "one") for n in enc["X"].names), "an encoder input unit that is not a reading of the window"
    frame = [n for n in enc["S"].names if n[0] == "slot"]
    assert dec["X"].names == frame + [("one",)] and mp.frame_units == frame, "the decoder's input is not the encoder's slot block and the constant"
    assert {n[1] for n in dec["S"].names if n[0] == "slot"} == set(PM.SENT_SLOTS), "a decoder register that is not of the sentence block"
    assert all(("ctrl", prev) in dec["S"] for prev, _, _ in dec["rules"]) and ("ctrl", ("P", "DONE")) in dec["S"]

def test_slot_write_circuit(root):
    """one slot write as docs/cell.puml draws it: a block gate in L4 that is an OR of rule units; a source gate in L4; write
    units in L5 = ReLU(source gate + source value - 1), one per value; keep units in L5 = ReLU(old value - block gate); and
    W6 sums keep + write back into the same one-hot block of the next state"""
    comp, mp = small_cell(root); comp = comp["encoder"]
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
    """every printing rule is (position, literals, (next position, token)): the literals are base predicates of the flat frame or of
    the last morpheme printed, or derived predicates whose DNF is over base literals, a token is one of the four kinds of morpheme
    or a move of the attention, every next position is a position, DONE or FAIL, and the rules reach the sentence end from P0"""
    import gen_module as G
    pr = G.PrintRules(); D = pr.derived(); positions = set(pr.positions()); F = G.flatten(PR.parse_bounded(R.merge_multiword("i like the big dog .".split())))
    assert pr.positions()[0] == "P0" and "DONE" not in positions
    def check_base(name):
        if name.startswith("out."): G.out_pred(name, None)                        # KeyError if it is not a predicate of the last morpheme
        else: G.base_pred(name, F)                                                # KeyError if it is not a predicate of the flat frame
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
            assert tok is None or tok[0] in ("c", "s", "vf", "af") or tok in (("h", "down"), ("h", "up")), tok
            nxts.add(nxt)
            if nxt not in reach and not G.is_terminal(nxt): reach.add(nxt); frontier.append(nxt)
    assert "DONE" in nxts and reach == positions, f"{len(positions - reach)} positions unreachable from P0, {len(reach - positions)} targets without rules"
    assert "".join(pr.render(F)) == R.generate(PR.parse_bounded(R.merge_multiword("i like the big dog .".split())))

def test_encoder_and_decoder_share_only_the_context(root):
    """the decoder is a second cell and nothing passes between the two but the context: the frames the encoder flushed and the
    sentence block of its final state.  No decoder unit reads a token, no encoder unit emits; a printing rule reads only the
    frame under the attention, the sentence block and the decoder's own output block, and writes no slot; and the decoder
    alone, given the frames a Clause spells, prints what the rule generator prints"""
    import numpy as np
    comp, mp = small_cell(root); enc, dec = comp["encoder"], comp["decoder"]
    assert not any(n[0] in ("out", "emit", "down", "up") for n in enc["S"].names) and not any(n[0] in ("pend", "shift", "flush") for n in dec["S"].names)
    for prev, lits, (cp, k, acts) in dec["rules"]:
        assert k == 0 and all(src in (PM.PRINT_SRC, PM.HEAD_SRC) for slot, v, src in acts), "a printing rule that consumes a token or writes a slot"
    for prev, lits, (cp, k, acts) in enc["rules"]: assert not any(src in (PM.PRINT_SRC, PM.HEAD_SRC) for slot, v, src in acts), "a reading rule that prints"
    # layer 1 and layer 3 of the decoder read the frame (input), the sentence block, the printing position and the output block (state), and the constant
    D = dec["S"]; frame_or_one = len(dec["X"])
    for L, W, b in (dec["layers"][0], dec["layers"][2]):
        for u, ref, w in W:
            if ref[0] == "S": assert D.names[ref[1]][0] in ("ctrl", "out") or D.names[ref[1]][1] in PM.SENT_SLOTS, D.names[ref[1]]
            else: assert ref[0] in ("X", "L2") and (ref[0] != "X" or ref[1] < frame_or_one)
    # what the runner hands over: the frames, and the sentence block, decoder register <- encoder state unit of the same name
    assert mp.handoff and all(D.names[d] == enc["S"].names[e] and D.names[d][1] in PM.SENT_SLOTS for d, e in mp.handoff)
    for s in ("i like the big dog .", "she took a rest ."):
        toks = R.merge_multiword(s.split())
        frames, final = mp.encode(toks)
        assert len(frames) == 1 and len(frames[0]) == len(mp.frame_units) and set(np.unique(frames[0])) <= {0.0, 1.0}
        units = mp.decode(frames, final)
        assert "".join(PM.output_lexicon(u) for u in units) == R.translate(s.split()) == mp.translate(toks) and len(units) > 3
        assert R.generate(PM.frames_to_clause([mp.frame_dict(f) for f in frames])) == R.translate(s.split())
    assert "".join(PM.output_lexicon(u) for u in mp.print_clause(PR.parse_bounded(R.merge_multiword("i like the big dog .".split())))) == "私は大きい犬が好きだ。"

def test_decoder_mirrors_encoder(root):
    """each head unit is an OR of rules: shift of the encoder's rules that consume a token, flush of those that hand a frame to the
    context, emit of the decoder's rules that print, down and up of those that move the attention.  A read gate is an OR of the
    rules that print a block and a read unit is AND(read gate, the frame's value) as a write unit is AND(source gate, source
    value); a lookup unit is AND(form gate, verb); a read with nothing in the block, or a form the verb has not, fires into
    FAIL as a copy with no source does"""
    comp, mp = small_cell(root); enc, dec = comp["encoder"], comp["decoder"]
    def into(c, k):
        d = collections.defaultdict(list)
        for u, src, w in c["layers"][k][1]: d[u].append((src, w))
        return d
    E3, E4 = enc["layers"][2][0], enc["layers"][3][0]; e4 = into(enc, 3); eb4 = enc["layers"][3][2]
    erule = lambda src: enc["rules"][E3.names[src[1]][1]]
    shift, flush = E4[("shift",)], E4[("flush",)]
    assert all(erule(src)[2][1] >= 1 for src, w in e4[shift] if E3.names[src[1]][0] == "rule")
    assert e4[flush] and all(src[0] == "L3" and w == 1.0 and PM.FLUSH in erule(src)[2][2] for src, w in e4[flush]) and eb4.get(flush, 0.0) == 0.0
    assert len(e4[flush]) == sum(1 for prev, lits, out in enc["rules"] if PM.FLUSH in out[2])
    S, X = dec["S"], dec["X"]; L3, L4, L5 = dec["layers"][2][0], dec["layers"][3][0], dec["layers"][4][0]
    b4, b5 = dec["layers"][3][2], dec["layers"][4][2]; into4, into5 = into(dec, 3), into(dec, 4)
    rule_of = lambda src: dec["rules"][L3.names[src[1]][1]]
    emit = L4[("emit",)]
    assert into4[emit] and all(src[0] == "L3" and w == 1.0 and any(a[2] == PM.PRINT_SRC for a in rule_of(src)[2][2]) for src, w in into4[emit]) and b4.get(emit, 0.0) == 0.0
    for move in ("down", "up"):
        g = L4[(move,)]
        assert into4[g] and all(src[0] == "L3" and w == 1.0 and ("att", move, PM.HEAD_SRC) in rule_of(src)[2][2] for src, w in into4[g]) and b4.get(g, 0.0) == 0.0
    gates = [i for i, n in enumerate(L4.names) if n[0] in ("gread", "gform", "gaform")]
    assert gates and all(into4[g] and all(src[0] == "L3" and w == 1.0 for src, w in into4[g]) and b4.get(g, 0.0) == 0.0 for g in gates)
    reads = [(i, n) for i, n in enumerate(L5.names) if n[0] == "read"]; forms = [(i, n) for i, n in enumerate(L5.names) if n[0] == "form"]
    assert reads and forms
    for i, (_, slot, idx, v) in reads:
        assert b5.get(i) == -1.0 and sorted(into5[i], key=str) == sorted([(("L4", L4[("gread", slot, idx)]), 1.0), (("X", X[("slot", slot, v)]), 1.0)], key=str)
    for i, (_, name, alt, cls, v) in forms:
        assert b5.get(i) == -1.0 and sorted(into5[i], key=str) == sorted([(("L4", L4[("gform", name, alt, cls)]), 1.0), (("X", X[("slot", "verb", v)]), 1.0)], key=str)
    into6 = collections.defaultdict(list)
    for tgt, src, w in dec["W6"]: into6[tgt].append((src, w))
    for i, (_, slot, idx, v) in reads:                              # an output unit is set by its read unit, or kept by its keep unit
        assert sorted(into6[S[("out", "s", slot, idx, v)]]) == sorted([(("L5", i), 1.0), (("L5", L5[("keepout", ("out", "s", slot, idx, v))]), 1.0)])
    crashes = [i for i, n in enumerate(L5.names) if n[0] in ("noread", "noform", "noaform")]
    assert crashes and all((("L5", i), 1.0) in into6[S[("ctrl", PM.FAIL)]] for i in crashes)
    for i in crashes:
        ins = dict(into5[i]); assert sum(1 for src, w in ins.items() if w == 1.0 and src[0] == "L4") == 1 and all(w == -1.0 for src, w in ins.items() if src[0] == "X")

def test_output_is_one_hot_morphemes(root):
    """y(t): the output block holds one morpheme — on a step that emits, the unit of that morpheme and no other; on a step that
    does not, the unit of the last morpheme printed, and none before the first; every output unit is a constant, a slot's value,
    or a form of a verb or an adjective, and the output lexical layer gives each a string"""
    comp, mp = small_cell(root); S = comp["decoder"]["S"]
    assert {n[1] for n in S.names if n[0] == "out"} == {"c", "s", "vf", "af"}
    for n in S.names:
        if n[0] == "out": assert isinstance(PM.output_lexicon(n), str)
    toks = R.merge_multiword("she took a rest .".split())
    frames, final = mp.encode(toks); seen = []; held = []
    def on_step(which, s, x, s2):
        on = [S.names[i] for i in mp.out_units if s2[i] > 0.5]; emits = s2[mp.emit_unit] > 0.5
        assert which == "decoder" and len(on) == (1 if emits or held else 0), len(on)
        if not emits: assert on == held[-1:], "a step that prints nothing changed the output block"
        else: seen.append(on[0])
        held[:] = on
    units = mp.decode(frames, final, on_step=on_step)
    assert seen == units and "".join(PM.output_lexicon(u) for u in units) == "彼女は休んだ。"

def test_decoder_reads_its_own_output(root):
    """the decoder is autoregressive: its output block — the last morpheme printed — stays in its state, kept by one keep unit per
    morpheme that the emit unit cuts, and its rules read it.  A stem is printed with its conjugation class (a constant or a form
    lookup declares it, an adjective's stem and the copula's noun have theirs), a class predicate is one layer-1 unit, the OR of
    the block's units of that class, and the finishing stage of the predicate takes the class from there: it has one position
    per mode for every predicate whose stem is out, and the ending it prints follows the block, whatever printed the stem"""
    import numpy as np, gen_module as G
    comp, mp = small_cell(root); dec = comp["decoder"]; S, X = dec["S"], dec["X"]
    L1, W1, b1 = dec["layers"][0]; L4 = dec["layers"][3][0]; L5, W5, b5 = dec["layers"][4]
    into5 = collections.defaultdict(list)
    for u, src, w in W5: into5[u].append((src, w))
    outs = [n for n in S.names if n[0] == "out"]
    for n in outs:                                                                # kept unless this step emits
        assert sorted(into5[L5[("keepout", n)]]) == sorted([(("S", S[n]), 1.0), (("L4", L4[("emit",)]), -1.0)]) and b5.get(L5[("keepout", n)], 0.0) == 0.0
    into1 = collections.defaultdict(list)
    for u, src, w in W1: into1[u].append((src, w))
    classes = G.stem_classes(); seen_classes = set()
    for c in classes:                                                             # a class predicate: the OR of the block's units of that class
        members = sorted((("S", S[n]), 1.0) for n in outs if PM.output_class(n) == c)
        key = ("conj", (f"out.class={c}",))
        if not members: assert key not in L1; continue
        assert sorted(into1[L1[key]]) == members and b1.get(L1[key], 0.0) == 0.0; seen_classes.add(c)
    assert {"V1C", "IA", "NA"} <= seen_classes
    assert PM.output_class(("out", "c", "い", "V1C")) == "V1C" and PM.output_class(("out", "c", "い")) is None and ("out", "c", "い", "V1C") in S and ("out", "c", "い") in S, "the same string, two morphemes"
    # the finishing stage: no position per class
    pr = G.PrintRules(); fin = [p for p in pr.positions() if isinstance(p, tuple) and p[0] == "FIN"]
    assert {p[1] for p in fin} == {"V", "VA", "ARU", "S", "NA"} and {p[2] for p in fin if p[1] == "NA"} == {"whcop", "whcop_emb"}
    assert {p[2] for p in fin if p[1] == "S"} == {"flags", "plain", "noneg", "conj", "conj_nn", "lets", "te", "stem"}
    # the ending follows the output block: the same position, the same frame, another stem in the block
    D = mp.dec
    def ending(stem_unit, frame):
        s = np.zeros(D.nS); s[S[("ctrl", ("P", ("FIN", "S", "flags")))]] = 1.0
        if stem_unit is not None: s[S[stem_unit]] = 1.0
        s2 = mp.step_decoder(s, np.concatenate([mp.frame_vector(frame), np.ones(1)]))
        return D.control(s2), [PM.output_lexicon(u) for u in mp.emitted(s2)]
    past = {"past": True}
    assert ending(("out", "c", "い", "V1C"), past) == (("P", "Q"), ["た"]) and ending(("out", "c", "い", "V1C"), {}) == (("P", "Q"), ["る"])
    assert ending(("out", "c", "た", "IA"), past) == (("P", "Q"), ["かった"]) and ending(("out", "c", "好き", "NA"), past) == (("P", "Q"), ["だった"])
    assert ending(None, past)[0] == PM.FAIL and ending(("out", "c", "。"), past)[0] == PM.FAIL, "an ending with no stem in the block"
    # end to end: predicates that chain stems, each ending read off the stem before it
    for s_, want in (("he is reading a book .", "彼は本を読んでいる。"), ("he can swim .", "彼は泳げる。"), ("i like the big dog .", "私は大きい犬が好きだ。")):
        toks = R.merge_multiword(s_.split()); assert mp.translate(toks) == R.translate(s_.split()) == want, s_

def test_clause_kind_decides_the_ending(root):
    """a frame's clause kind decides the form its predicate ends in: a clause marked te or stem, spliced into another like a
    that-clause, prints up to the te form or the masu stem of the predicate its registers build — by a lookup for a plain verb,
    by the ending of the class of the stem just printed otherwise, never by slicing — and gives the attention back with nothing
    after it; the clause it is spliced into goes on.  The rule generator prints such a clause the same way (generate(form=…)),
    and where a predicate has no such form — a fixed phrase, must, an imperative — both refuse.  A bare -ing clause under while
    is the first construction that uses it: it ends at its masu stem and takes ながら"""
    import copy, gen_module as G
    comp, mp = small_cell(root); pr = G.PrintRules()
    def host(): c = PR.parse_bounded(R.merge_multiword("he came .".split())); return c
    def inner(sentence, form, **regs):
        c = PR.parse_bounded(R.merge_multiword(sentence.split())); c.inner_of = form
        for k, v in regs.items(): setattr(c, k, v)
        return c
    cases = [("he opened the window .", "te", {}, "彼は彼が窓を開けて来た。"), ("he opened the window .", "stem", {}, "彼は彼が窓を開け来た。"),
             ("he is reading a book .", "te", {}, "彼は彼が本を読んでいて来た。"), ("he is reading a book .", "stem", {}, "彼は彼が本を読んでい来た。"),
             ("he can swim .", "te", {}, "彼は彼が泳げて来た。"), ("i like the big dog .", "te", {}, "彼は私が大きい犬が好きで来た。")]
    for sentence, form, regs, want in cases:
        c = host(); c.embedded = inner(sentence, form, **regs)
        frames = G.frames_of(copy.deepcopy(c))
        assert [F.get("link") for F in frames] == [form, None] and [F.get("clause") for F in frames] == [None, form]
        assert "".join(pr.render(copy.deepcopy(c))) == R.generate(copy.deepcopy(c)) == want, (sentence, form)
        assert "".join(PM.output_lexicon(u) for u in mp.print_clause(copy.deepcopy(c))) == want, (sentence, form, "the compiled decoder")
    for sentence, form, regs in (("he opened the window .", "te", {"modal": "must"}), ("he opened the window .", "stem", {"imp": True}), ("i like the big dog .", "stem", {})):
        c = host(); c.embedded = inner(sentence, form, **regs)
        try: R.generate(copy.deepcopy(c)); assert False, "generate() printed a form the predicate has not"
        except ValueError: pass
        try: pr.render(copy.deepcopy(c)); assert False, "the rules printed a form the predicate has not"
        except G.Refuse: pass
        try: mp.print_clause(copy.deepcopy(c)); assert False, "the decoder printed a form the predicate has not"
        except PR.ParseFail: pass
    # the attention: down at the link, back up from the stem, on to the outer predicate
    c = host(); c.embedded = inner("he opened the window .", "te"); path = []; at = [0]
    def on_step(which, s, x, s2):
        if not path or path[-1] != at[0]: path.append(at[0])
        at[0] += int(s2[mp.down_unit] > 0.5) - int(s2[mp.up_unit] > 0.5)
    mp.decode(mp.clause_context(c), on_step=on_step); assert path == [0, 1, 0], path
    # while + a bare -ing clause, read by the encoder and printed through the stem mode
    for s_, want in (("while walking , he sang a song .", "歩きながら彼は歌を歌った。"), ("while having a car , he walks .", "車を持っていながら彼は歩く。")):
        toks = R.merge_multiword(s_.split()); assert mp.translate(toks) == R.translate(s_.split()) == want, (s_, mp.translate(toks))

def _diagrams(root):
    """every control point of the small cell -> (its rules, its canonical diagram), built in one table over the cell's own vocabulary"""
    import canon as C
    comp, mp = small_cell(root)
    if "canon" not in _cell:
        by_prev = collections.defaultdict(list)
        for p, l, o in comp["rules"]: by_prev[p].append((l, o))
        tab = C.Table(C.Model(comp["slot_vocab"]))
        _cell["canon"] = (tab, {prev: (rs, C.diagram(rs, table=tab)) for prev, rs in by_prev.items()})
    tab, D = _cell["canon"]
    return C, tab, D

def test_canon_is_canonical(root):
    """a control point's rules have one reduced ordered decision diagram over the cell's variables: the same object whatever order the
    rules come in, no variable tested twice on a path, and the paths are the rules in normal form — the region a path describes is
    covered by exactly the original rules whose outcome its leaf carries, a GAP by none, a CONFLICT by rules with the outcomes it lists"""
    C, tab, D = _diagrams(root)
    assert len(D) > 100
    for prev, (rs, d) in D.items():
        assert C.diagram(C.shuffled(rs, 1), table=tab) is d and C.diagram(list(reversed(rs)), table=tab) is d, f"{prev}: the diagram depends on the rules' order"
        def no_repeat(n, seen=()):
            if n.kind != "TEST": return True
            return n.var not in seen and all(no_repeat(b, seen + (n.var,)) for b in n.branches.values())
        assert no_repeat(d), f"{prev}: a variable tested twice on one path"
        forms = [(tab.normalize(lits, out), out) for lits, out in rs]
        for lits, leaf in C.paths(d):
            classes = C.path_classes(lits); outs = {}
            for nr, out in forms:
                if nr is None: continue
                f = nr[0]
                for var, true in classes.items(): f = C.substitute(f, var, tuple(sorted(true)))
                if f != C.F and C.satisfiable(f, tab): outs[repr(out)] = out       # some input of the region satisfies the rule: it covers part of it
            outs = sorted(outs.values(), key=repr)
            if leaf.kind == C.TERM: assert outs == [leaf.out], f"{prev}: path outcome differs from the rules"
            elif leaf.kind == C.GAP: assert outs == [], f"{prev}: a gap the rules cover"
            else: assert tuple(outs) == leaf.out, f"{prev}: conflict mismatch"

def test_canon_over_feasible_inputs(root):
    """the diagram ranges over feasible inputs, not over literals treated as independent: a token is never a determiner and a verb at
    once, so a rule asking for both has no path; the PP slots fill in order, so pp.count=1 and a filled pp2 together are no region;
    a derived predicate is tested through its base literals; a slot's values are what the rules can write, or nothing"""
    import canon as C
    tab = C.Table()                                                            # the universal model: the hand-written rules' vocabulary and the lexicon
    A, B, X = ("A", 0, ()), ("B", 0, ()), ("X", 0, ())
    d = C.diagram([((("w0.class=det", True), ("w0.class=end", True)), X), ((("w0.class=det", False),), B), ((("w0.class=det", True), ("w0.class=end", False)), A)], table=tab)
    assert C.complete(d) and C.consistent(d) and {l.out for l in C.leaves(d)} == {A, B}, "a token that is a determiner and the end at once"
    assert all(isinstance(v, frozenset) for v in tab.model.universe("w0")) and len(tab.model.universe("w0")) < len(tab.model.feats), "window values are the tokens' predicate sets, grouped"
    d = C.diagram([((("pp.count=0", True),), A), ((("pp.count=1", True),), A), ((("pp.count=2", True),), A), ((("pp.count=3", True),), B)], table=tab)
    assert C.complete(d) and C.consistent(d) and d.var == "pp.count", "the PP count is one variable, the PPs filling in order"
    d = C.diagram([((("d.neg", True),), A), ((("d.neg", False),), B)], table=tab)
    assert C.complete(d) and C.consistent(d) and all(n.var in ("neg", "adv1", "adv2", "subj.noun", "obj.noun") for n in C._nodes(d) if n.kind == "TEST"), "d.neg tested as itself"
    d = C.diagram([((("slot.cop_kind=adj", True),), A), ((("slot.cop_kind=np", True),), B)], table=tab)
    assert not C.complete(d) and d.var == "cop_kind" and len(d.branches) >= 3, "cop_kind: adj, np, and the other values or nothing"

def test_canon_dsl_roundtrip(root):
    """the text form of every diagram parses back to the same diagram, and the name round-trips"""
    C, tab, D = _diagrams(root)
    for prev, (rs, d) in D.items():
        name, back = C.parse(C.dump(d, str(prev)), table=tab)
        assert back is d and name == str(prev), f"{prev}: text form does not round-trip"
    text = C.dump(D[("NP1", "cop")][1], "NP1@cop")
    assert text.startswith("NP1@cop:\n") and "w0 ⟨" in text and "→ (" in text, text

def test_canon_control_points_total(root):
    """every control point of the cell is complete and consistent over its feasible inputs — every input reaches exactly one outcome,
    a refusal included: the hand-written modules by their otherwise-rules (a table position refuses a value outside its table), the
    induced control points by the refusals the compiler materialises where a guard fails or the traces disagreed, and the return
    states by its closing refusal for a return address no caller serves"""
    C, tab, D = _diagrams(root)
    bad = [prev for prev, (rs, d) in D.items() if not (C.complete(d) and C.consistent(d))]
    assert not bad, f"{len(bad)} of {len(D)} control points are not total, e.g. {bad[0]}"
    induced = {prev for prev in D if isinstance(prev[0], tuple) or prev[0] in ("START", "INNER")}
    assert len(induced) >= 25 and any(out == (PM.FAIL, 0, ()) for prev in induced for lits, out in D[prev][0]), "no refusal materialised at the induced control points"
    assert any(prev[0] in ("RET", "RETM") and out == (PM.FAIL, 0, ()) and lits and all(not t and PM.is_return_key(q) for q, t in lits) for prev, (rs, d) in D.items() for lits, out in rs), "no closing refusal at a return state"
    assert any(prev[0] == "M" and out == (PM.FAIL, 0, ()) and lits and all(not t and q.startswith("w0.surf=") for q, t in lits) for prev, (rs, d) in D.items() for lits, out in rs), "no otherwise at a table position"

def test_layers_roundtrip(root):
    toks = R.merge_multiword("a lot of people came .".split())
    assert toks[0] == "a lot"
    assert {c for c, v in R.classify("rest")} >= {"verb", "noun"}
    assert R.translate("i like the big dog .".split()) == "私は大きい犬が好きだ。"

def test_feedback_edges(root):
    assert R.translate("she took a rest .".split()) == "彼女は休んだ。"            # the noun's use-verb rewrote the verb
    assert R.translate("i rest .".split()) == "私は休む。" and R.translate("i need a rest .".split()) == "私は休息が必要だ。"   # the same token read by the parser's state
    comp, mp = small_cell(root)
    assert any(n[0] == "usepair" for n in comp["encoder"]["layers"][0][0].names)

def test_verb_hits_are_layer_zero(root):
    """the table rows that join the frame's verb with the window — the idiom tails, the particle after a verb, the use nouns — are
    units of the cell: one AND unit per row at layer 1 over the verb's unit and the window's table keys (threshold the number of
    keys), ORed at layer 2 into the hit predicates the rules read; the input carries no hit the verb decides, so the runner hands
    the cell the window alone and the cell still reads `she took a rest` as the rules do"""
    comp, mp = small_cell(root); enc = comp["encoder"]; S, X = enc["S"], enc["X"]; L1, L2 = enc["layers"][0][0], enc["layers"][1][0]
    rows = [n for n in L1.names if n[0] == "hitrow"]; hits = {n[1] for n in L2.names if n[0] == "pred" and PM.verb_hit(n[1])}
    assert rows and "hit=use" in hits and not any(n[0] == "pred" and PM.verb_hit(n[1]) for n in X.names)
    assert set(rows) <= set(PM.Layer0Lexical().hit_units(comp))
    W1, b1 = enc["layers"][0][1], enc["layers"][0][2]; W2 = enc["layers"][1][1]
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

def test_context_is_one_frame_per_clause(root):
    """the context is one frame per clause, in the order the clauses closed.  At a that-clause a rule induced from the traces writes
    the link into the working frame and raises flush, so the frame goes to the context whole; OPEN — a state per opening site —
    then clears every slot by its own gate, marks the new frame embedded and keeps the sentence block; CLOSE flushes the last
    frame.  The decoder's attention starts on the first frame; at the link it moves down, the clause prints (が on its subject,
    plain forms, と) and the attention moves back up, the outer frame continuing at its tail.  A question mark read after a
    that-clause reaches the decoder as the sentence block of the encoder's final state.  A that-clause inside a that-clause is
    one more frame under the same rules; a that-clause after a copula opens inside the copula module, which refuses it; and an
    attention that leaves the context is a refusal"""
    comp, mp = small_cell(root); enc = comp["encoder"]; V = comp["slot_vocab"]
    opening = [r for r in enc["rules"] if r[2][0][:2] == ("OPEN", "that")]; sites = {r[2][0][2] for r in opening}
    assert len(sites) >= 2 and all(("link", "that", ("const",)) in out[2] and PM.FLUSH in out[2] and out[1] == 0 for prev, lits, out in opening)
    for sid in sites:
        opens = [r for r in enc["rules"] if r[0] == ("OPEN", "that", sid)]; inner = [r for r in opens if r[2][0] == ("INNER", sid)]
        assert len(inner) == 2 and len(opens) == 5 and {dict(lits)["flag.question"] for _, lits, _ in inner} == {True, False}
        for prev, lits, (cp, k, acts) in inner:
            assert dict(lits)["slot.link=that"] is True and ("clause", "embedded", ("const",)) in acts
            assert (("sent.question", True, ("const",)) in acts) == dict(lits)["flag.question"]
            assert {sl for sl, v, src in acts if src[0] == "clear"} == {sl for sl in V if sl != "clause" and sl not in PM.SENT_SLOTS}, "OPEN does not clear exactly the frame"
        assert [out for _, lits, out in opens if dict(lits)["slot.link=that"] is False] == [(PM.FAIL, 0, ())]
        assert all(out == (PM.FAIL, 0, ()) for _, lits, out in opens if any(t and q in ("slot.clause=before", "slot.clause=after") for q, t in lits)), "a that-clause opened inside a subordinate clause"
    assert (("CLOSE",), (("slot.clause=after", False), ("slot.clause=before", False)), (("DONE",), 0, (PM.FLUSH,))) in enc["rules"]
    def walk(sentence):
        toks = R.merge_multiword(sentence.split()); frames, final = mp.encode(toks); path = []; at = [0]
        def on_step(which, s, x, s2):
            if not path or path[-1] != at[0]: path.append(at[0])
            at[0] += int(s2[mp.down_unit] > 0.5) - int(s2[mp.up_unit] > 0.5)
        out = "".join(PM.output_lexicon(u) for u in mp.decode(frames, final, on_step=on_step))
        assert out == R.translate(sentence.split()) == mp.translate(toks) == R.generate(mp.parse(toks)), (sentence, out)
        return [mp.frame_dict(f) for f in frames], final, path, out
    frames, final, path, out = walk("he said that he was tired .")
    assert out == "彼は彼が疲れていたと言った。" and path == [0, 1, 0] and len(frames) == 2
    assert frames[0].get("link") == "that" and "clause" not in frames[0] and frames[0]["verb"] == ("say", "言う", "g") and frames[0]["subj.noun"] == "彼" and frames[0]["past"] is True
    assert frames[1].get("clause") == "embedded" and "link" not in frames[1] and "verb" not in frames[1] and frames[1]["cop_kind"] == "adj"
    frames, final, path, out = walk("he said that he was tired ?")                 # the question mark comes after the outer frame has left
    assert out.endswith("言いましたか。") and "question" not in frames[0] and frames[1].get("sent.question") is True and all(final[e] > 0.5 for d, e in mp.handoff)
    frames, final, path, out = walk("he said that she thought that he was tired .")
    assert len(frames) == 3 and path == [0, 1, 2, 1, 0] and [f.get("clause") for f in frames] == [None, "embedded", "embedded"] and [f.get("link") for f in frames] == ["that", "that", None]
    try: mp.translate(R.merge_multiword("i 'm sure that he will come .".split())); assert False, "a copula's that-clause was read"
    except PR.ParseFail: pass
    toks = R.merge_multiword("he said that he was tired .".split()); frames, final = mp.encode(toks)
    try: mp.decode(frames[:1], final); assert False, "the decoder printed a link with no frame under it"
    except PR.ParseFail as e: assert "attention" in str(e)

def test_subordinate_clause_in_both_orders(root):
    """a subordinate clause is a frame of its own beside its main clause's, in the order the two were read, and is printed first
    either way.  Read first, it opens on its conjunction — OPEN reads the conjunction into the frame and marks it, nothing is
    flushed — and closes where the main clause begins: OPEN flushes it, TURN clears and marks the main clause's frame.  Read
    second, it opens where the PP machine of the main clause returns at the conjunction (its second exit): OPEN marks the main
    frame and flushes it, TURN clears, reads the conjunction and marks the new frame, keeping the sentence block.  The decoder
    starts on the first frame, steps over a subordinate clause that came first, prints the main clause's front adverb, moves to
    the subordinate frame — up or down — prints it with が on its subject, plain forms and its conjunction's ending, and moves
    back for the rest of the main clause.  A question mark read after the subordinate clause reaches the decoder in the sentence
    block.  What the frames cannot hold is refused by the encoder: a second subordinate clause, one that opens when the main
    clause has begun with an adverb; unless is read but not printed"""
    comp, mp = small_cell(root); enc = comp["encoder"]; V = comp["slot_vocab"]
    def walk(sentence):
        toks = R.merge_multiword(sentence.split()); frames, final = mp.encode(toks); path = []; at = [0]
        def on_step(which, s, x, s2):
            if not path or path[-1] != at[0]: path.append(at[0])
            at[0] += int(s2[mp.down_unit] > 0.5) - int(s2[mp.up_unit] > 0.5)
        out = "".join(PM.output_lexicon(u) for u in mp.decode(frames, final, on_step=on_step))
        assert out == R.translate(sentence.split()) == mp.translate(toks) == R.generate(mp.parse(toks)), (sentence, out)
        return [mp.frame_dict(f) for f in frames], final, path, out
    marks = lambda F: {k: F[k] for k in PM.MARKS if k in F}
    frames, final, path, out = walk("when i was a child , i loved reading .")
    assert out == "私が子供だったとき私は読むのが好きだった。" and path == [0, 1, 0, 1], path
    assert [marks(F) for F in frames] == [{"clause": "before", "conj": "when"}, {"sublink": "before"}] and frames[0]["cop_kind"] == "np" and frames[1]["want"] == "love"
    frames, final, path, out = walk("i loved reading when i was a child .")
    assert out == "私が子供だったとき私は読むのが好きだった。" and path == [0, 1, 0], path
    assert [marks(F) for F in frames] == [{"sublink": "after"}, {"clause": "after", "conj": "when"}] and frames[1]["cop_kind"] == "np" and frames[0]["want"] == "love"
    frames, final, path, out = walk("if it rains tomorrow , i will stay at home .")
    assert out == "明日雨が降ったら私は家に泊まる。" and frames[0]["conj"] == "if" and frames[0].get("past") is True and "past" not in frames[1]
    frames, final, path, out = walk("i went out although it was raining .")
    assert out == "雨が降っていたが、私は出かけた。" and path == [0, 1, 0]
    frames, final, path, out = walk("i went out when it was raining ?")            # the question mark comes after the main frame has left
    assert out.endswith("出かけましたか。") and "question" not in frames[0] and frames[1].get("sent.question") is True and all(final[e] > 0.5 for d, e in mp.handoff)
    # the clause rules, per kind of boundary
    by_kind = collections.defaultdict(set)
    for prev, lits, out in enc["rules"]:
        if out[0][0] == "OPEN": by_kind[out[0][1]].add(out[0][2]); assert out[1] == 0 and (out[2] == () or out[0][1] == "that"), "an induced boundary row that writes"
    assert by_kind["before"] and by_kind["main"] and by_kind["after"]
    rules_at = lambda st: [(lits, out) for prev, lits, out in enc["rules"] if prev == st]
    for sid in by_kind["before"]:
        (lits, (cp, k, acts)), = [r for r in rules_at(("OPEN", "before", sid)) if r[1][0] != PM.FAIL]
        assert lits == (("d.clause.lead", True),) and cp == ("INNER", sid) and k == 1 and PM.FLUSH not in acts
        assert set(acts) == {("clause", "before", ("const",)), ("conj", None, ("copy", "conj")), ("sub", True, ("const",))}
    import numpy as np
    def step_from(state, frame, window):
        """the control point the encoder reaches in one step from `state` with `frame` as its working frame and `window` ahead"""
        E = mp.enc; s = np.zeros(E.nS); s[E.S[("ctrl", state)]] = 1.0
        for slot, v in frame.items(): s[E.S[("slot", slot, v)]] = 1.0
        return E.control(mp.step_encoder(s, mp.input_vector(window, 0)))
    for sid in by_kind["main"]:
        opens = rules_at(("OPEN", "main", sid)); go = [r for r in opens if r[1][0] != PM.FAIL]
        assert go and all(dict(lits)["slot.clause=before"] is True and dict(lits)["d.clause.close"] is True and cp == ("TURN", sid) and k == 0 and PM.FLUSH in acts for lits, (cp, k, acts) in go)
        # the clause closes at a comma, or at the `or not` of whether, and nowhere else: the rule parser's tail refuses any other token there
        sub = {"clause": "before", "conj": "when"}
        assert step_from(("OPEN", "main", sid), sub, [",", "i"]) == ("TURN", sid) and step_from(("OPEN", "main", sid), sub, ["painted", ","]) == PM.FAIL
        assert step_from(("OPEN", "main", sid), dict(sub, conj="whether"), ["or", "not", ","]) == ("TURN", sid) and step_from(("OPEN", "main", sid), sub, ["or", "not", ","]) == PM.FAIL
        (lits, (cp, k, acts)), = rules_at(("TURN", sid))
        assert lits == () and cp == ("INNER", sid) and k == 0 and ("sublink", "before", ("const",)) in acts
        assert {sl for sl, v, src in acts if src[0] == "clear"} == set(V) - {"sublink"}, "TURN does not clear exactly the frame"
    for sid in by_kind["after"]:
        (lits, (cp, k, acts)), = [r for r in rules_at(("OPEN", "after", sid)) if r[1][0] != PM.FAIL]
        assert lits == (("d.clause.trail", True),) and cp == ("TURN", sid) and k == 0 and set(acts) == {("sublink", "after", ("const",)), PM.FLUSH}
        turns = rules_at(("TURN", sid)); assert len(turns) == 2 and {dict(lits)["flag.question"] for lits, _ in turns} == {True, False}
        for lits, (cp, k, acts) in turns:
            assert cp == ("INNER", sid) and k == 1 and {("clause", "after", ("const",)), ("conj", None, ("copy", "conj")), ("sub", True, ("const",))} <= set(acts)
            assert (("sent.question", True, ("const",)) in acts) == dict(lits)["flag.question"]
            assert {sl for sl, v, src in acts if src[0] == "clear"} == set(V) - {"clause", "conj", "sub"} - set(PM.SENT_SLOTS)
        # the boundary is reached from the PP machine's second exit and nowhere else, by its return address alone
        into = [(prev, lits) for prev, lits, out in enc["rules"] if out[0] == ("OPEN", "after", sid)]
        assert into and all(prev[0] == "MRET" and prev[1] == "pp" and prev[3] == "sub" and lits == () for prev, lits in into)
    exits = [(lits, out) for prev, lits, out in enc["rules"] if prev == ("RETM", "pp", "sub") and out[0] != PM.FAIL]
    assert exits and all(lits and all(PM.is_return_key(q) for q, t in lits) for lits, out in exits)
    import pp_module as PPM
    assert sum(1 for st, lits, out in PPM.rules() if out[0] == "RETSUB") == 1 and any(st == "PP0" and out[0] == "RET" and any("flag.sub" in c for l in lits if not l.startswith("¬") for c in PPM.derived().get(l, [])) for st, lits, out in PPM.rules())
    assert set(PM.SUBORD_TRAIL) == {w for w in R.SUBORD if w not in R.PREPS and w not in R.COORD} and "while" not in PM.SUBORD_TRAIL and "if" in PM.SUBORD_TRAIL
    # the PP machine on a main clause's frame: its second exit at a conjunction with one reading; before reads as the preposition
    # (and the clause's verb is then refused), while as a coordinator; in a subordinate clause's frame it returns at the comma
    went = {"verb": ("go", "行く", "g"), "subj.noun": "彼", "subj.cls": "h", "past": True}
    sim = lambda window, frame: PM.simulate_module("pp", window, 0, frame, None)
    assert sim(["when", "it", "rained", "."], went)[::3] == (0, "sub") and sim(["if", "it", "rained", "."], went)[::3] == (0, "sub")
    assert sim(["before", "the", "rain", "started", "."], went) is None and sim(["while", "it", "rained", "."], went) is None
    assert sim(["when", "it", "rained", "."], dict(went, sublink="before")) is None and sim(["when", "it", "rained", "."], dict(went, clause="embedded")) is None
    assert sim([",", "he", "came", "."], dict(went, sub=True, clause="before", conj="when"))[::3] == (0, "") and sim(["when", "he", "came", "."], dict(went, sub=True, clause="after", conj="if"))[::3] == (0, "")
    # the front adverb of the main clause comes before the subordinate clause, whichever side its frame is on
    import copy
    main = PR.Clause(); main.subj, main.subj_cls = "彼", "h"; main.verb = ("come", "来る", "k"); main.past = True; main.front = "実は、"
    sub = PR.Clause(); sub.subj, sub.subj_cls = "私", "h"; sub.copula = ("np", "子供", "h"); sub.past = True
    for order in ("before", "after"):
        m = copy.deepcopy(main); m.subords = [(order, "when", sub)]; m.pre_clause = R.subord_form(sub, "when")
        assert "".join(GM.PrintRules().render(m)) == R.generate(copy.deepcopy(m)) == "実は、私が子供だったとき彼は来た。"
    for sid in by_kind["before"]:                                                                  # a clause that opens when its main clause has begun; when before an auxiliary is the question word
        assert step_from(("OPEN", "before", sid), {}, ["when", "i", "was"]) == ("INNER", sid) and step_from(("OPEN", "before", sid), {}, ["when", "did", "he"]) == PM.FAIL
        assert step_from(("OPEN", "before", sid), {"time": sorted(V["time"])[0]}, ["when", "i", "was"]) == PM.FAIL and step_from(("OPEN", "before", sid), {"front": sorted(V["front"])[0]}, ["when", "i", "was"]) == PM.FAIL
        assert step_from(("OPEN", "before", sid), {"sublink": "before"}, ["when", "i", "was"]) == PM.FAIL and step_from(("OPEN", "before", sid), {}, ["and", "i", "was"]) == PM.FAIL
    for sid in {out[0][2] for prev, lits, out in enc["rules"] if out[0][:2] == ("OPEN", "that")}:      # a that-clause inside a subordinate clause
        assert step_from(("OPEN", "that", sid), {"link": "that"}, ["he"]) == ("INNER", sid)
        assert step_from(("OPEN", "that", sid), {"link": "that", "clause": "before", "conj": "when"}, ["he"]) == PM.FAIL and step_from(("OPEN", "that", sid), {"link": "that", "clause": "after", "conj": "when"}, ["he"]) == PM.FAIL
    # refusals: a second subordinate clause; a main clause that began before its subordinate clause; unless is read, not printed
    for s in ("when i was a child , i loved reading when it was raining .", "yesterday when i was a child , i loved reading ."):
        toks = R.merge_multiword(s.split())
        try: mp.translate(toks); assert False, f"translated: {s}"
        except PR.ParseFail: pass
    toks = R.merge_multiword("unless it rains , i will go , too .".split())
    assert R.generate(mp.parse(toks)) == R.translate(toks) and mp.frame_dict(mp.encode(toks)[0][0]).get("neg") is True
    try: mp.translate(toks); assert False, "unless was printed"
    except PR.ParseFail as e: assert "PRINT" in str(e)
    # a context out of order is a refusal: the subordinate frame on the other side of its main clause
    toks = R.merge_multiword("i loved reading when i was a child .".split()); frames, final = mp.encode(toks)
    try: mp.decode(frames[::-1], final); assert False, "printed frames in the wrong order"
    except PR.ParseFail: pass

def test_subordinate_frame_holds_its_conjunction(root):
    """what a conjunction changes in its clause (rules_ja.subord_adjust: the past of if, once and after, the negative of unless, the
    potential for may and no will under so that) is written into the frame by the rule that flushes it — adjust_rules mirrors
    subord_adjust for every conjunction of the table and every modal — so the frame the decoder reads is the clause generate()
    prints; and the printing rules give every conjunction its ending after the clause's finite form, as subord_form does, refusing
    unless, which slices the predicate's string"""
    import copy
    modals = sorted(set(R.MODALS.values()))
    rules = PM.adjust_rules(("X",), (("slot.clause=after", True),), ("Y",), [PM.FLUSH]); pr = GM.PrintRules(); n = 0
    for conj in sorted(R.SUBORD):
        for modal in [None] + modals:
            for past in (False, True):
                for neg in (False, True):
                    c = PR.Clause(); c.subj, c.subj_cls = "彼", "h"; c.verb = ("go", "行く", "g"); c.modal, c.past, c.neg = modal, past, neg
                    F = GM.flatten(c); F.update({"clause": "after", "conj": conj}); frame = dict(F)
                    P = PM.slot_predicates(frame)
                    hits = [out for st, lits, out in rules if all((q in P) == t for q, t in lits)]
                    assert len(hits) == 1 and hits[0][0] == ("Y",) and PM.FLUSH in hits[0][2], (conj, modal, hits)
                    PM.apply_actions(frame, hits[0][2], [], 0)
                    want = copy.deepcopy(c); R.subord_adjust(want, conj); W = GM.flatten(want); W.update({"clause": "after", "conj": conj})
                    assert frame == W, (conj, modal, past, neg, frame, W)
                    # the two clauses as a context: the main clause 彼は来た with this one after it
                    main = PR.Clause(); main.subj, main.subj_cls = "彼", "h"; main.verb = ("come", "来る", "k"); main.past = True
                    main.subords = [("after", conj, c)]; main.pre_clause = R.subord_form(c, conj)
                    try: got = "".join(pr.render(main))
                    except GM.Refuse: assert conj == "unless", conj; continue
                    assert got == R.generate(copy.deepcopy(main)), (conj, modal, past, neg, got); n += 1
    assert n > 500
    c = PR.Clause(); c.verb = ("walk", "歩く", "g"); main = PR.Clause(); main.subj, main.subj_cls = "彼", "h"; main.verb = ("come", "来る", "k")
    main.subords = [("before", "while", c)]; main.pre_clause = R.subord_form(c, "while")
    assert main.pre_clause == "歩きながら" and "".join(pr.render(main)) == R.generate(copy.deepcopy(main)) == "歩きながら彼は来る。"       # (the stem mode: test_clause_kind_decides_the_ending)

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
    """the item order: WRONG < REFUSED < STIFF < RIGHT, UNKNOWN below RIGHT and incomparable with the rest; the other chains total.
    A verdict is right, stiff or wrong, a stiff or wrong one names its cause, and the frontier groups the judged outputs by it"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("lattice", os.path.join(root, "harness", "lattice.py")); LT = importlib.util.module_from_spec(spec); spec.loader.exec_module(LT)
    assert LT.cmp("sentence", "WRONG", "REFUSED") == "up" and LT.cmp("sentence", "REFUSED", "RIGHT") == "up" and LT.cmp("sentence", "WRONG", "RIGHT") == "up"
    assert LT.cmp("sentence", "RIGHT", "REFUSED") == "down" and LT.cmp("sentence", "RIGHT", "UNKNOWN") == "down"
    assert LT.cmp("sentence", "UNKNOWN", "REFUSED") == "incomparable" and LT.cmp("sentence", "REFUSED", "UNKNOWN") == "incomparable" and LT.cmp("sentence", "UNKNOWN", "WRONG") == "incomparable"
    assert LT.cmp("sentence", "UNKNOWN", "RIGHT") == "up" and LT.cmp("matrix", "GAP", "AGREE") == "up" and LT.cmp("matrix", "AGREE", "MISFIRE") == "down" and LT.cmp("negative", "REFUSED", "ACCEPTED") == "down"
    # a stiff output: better than a wrong one or a refusal, worse than natural Japanese
    assert LT.cmp("sentence", "WRONG", "STIFF") == "up" and LT.cmp("sentence", "REFUSED", "STIFF") == "up" and LT.cmp("sentence", "STIFF", "RIGHT") == "up"
    assert LT.cmp("sentence", "STIFF", "REFUSED") == "down" and LT.cmp("sentence", "RIGHT", "STIFF") == "down" and LT.cmp("sentence", "UNKNOWN", "STIFF") == "incomparable" and LT.cmp("sentence", "STIFF", "UNKNOWN") == "incomparable"
    V = {"k": {"bad": "wrong", "ok": "right", "so-so": "stiff"}}; U = {"u": "ungrammatical"}
    assert LT.sentence_level("ref", "ref", "k", V, U) == "RIGHT" and LT.sentence_level("", "ref", "k", V, U) == "REFUSED" and LT.sentence_level("bad", "ref", "k", V, U) == "WRONG"
    assert LT.sentence_level("ok", "ref", "k", V, U) == "RIGHT" and LT.sentence_level("new", "ref", "k", V, U) == "UNKNOWN" and LT.sentence_level("so-so", "ref", "k", V, U) == "STIFF"
    Lq = {"sentence dev 1: a .": ("sentence", "WRONG", "x"), "sentence dev 2: b .": ("sentence", "STIFF", "y"), "sentence dev 3: c .": ("sentence", "RIGHT", "z")}
    Fq = LT.frontier(Lq, {}, {"dev 1: a .": {"x": "idiom"}, "dev 2: b .": {"y": "pronoun"}})
    assert set(Fq) == {"sentence output wrong: idiom", "sentence output stiff: pronoun"}, Fq
    adj = json.load(open(os.path.join(root, "harness", "adjudications.json"), encoding="utf-8"))["sentences"]
    assert all(("cause" in a) == (a["verdict"] in ("stiff", "wrong")) and a.get("cause", LT.CAUSES[0]) in LT.CAUSES for a in adj) and {a["verdict"] for a in adj} >= {"right", "stiff", "wrong"}
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
