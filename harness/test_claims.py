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
        extra = ("she took a rest .", "i like the big dog .", "he said that he was tired .", "he said that he was tired ?", "he said that she thought that he was tired .")
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
        if prev[0] in ("OPEN", "CLOSE"): kinds["flush"] += 1
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
    """every printing rule is (position, literals, (next position, token)): the literals are base predicates of the flat frame or
    derived predicates whose DNF is over base literals, a token is one of the four kinds of morpheme or a move of the attention,
    every next position is a position, DONE or FAIL, and the rules reach the sentence end from P0"""
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
            assert tok is None or tok[0] in ("c", "s", "vf", "af") or tok in (("h", "down"), ("h", "up")), tok
            nxts.add(nxt)
            if nxt not in reach and not G.is_terminal(nxt): reach.add(nxt); frontier.append(nxt)
    assert "DONE" in nxts and reach == positions, f"{len(positions - reach)} positions unreachable from P0, {len(reach - positions)} targets without rules"
    assert "".join(pr.render(F)) == R.generate(PR.parse_bounded(R.merge_multiword("i like the big dog .".split())))

def test_encoder_and_decoder_share_only_the_context(root):
    """the decoder is a second cell and nothing passes between the two but the context: the frames the encoder flushed and the
    sentence block of its final state.  No decoder unit reads a token, no encoder unit emits; a printing rule reads only the
    frame under the attention and the sentence block, and writes no slot; and the decoder alone, given the frames a Clause
    spells, prints what the rule generator prints"""
    import numpy as np
    comp, mp = small_cell(root); enc, dec = comp["encoder"], comp["decoder"]
    assert not any(n[0] in ("out", "emit", "down", "up") for n in enc["S"].names) and not any(n[0] in ("pend", "shift", "flush") for n in dec["S"].names)
    for prev, lits, (cp, k, acts) in dec["rules"]:
        assert k == 0 and all(src in (PM.PRINT_SRC, PM.HEAD_SRC) for slot, v, src in acts), "a printing rule that consumes a token or writes a slot"
    for prev, lits, (cp, k, acts) in enc["rules"]: assert not any(src in (PM.PRINT_SRC, PM.HEAD_SRC) for slot, v, src in acts), "a reading rule that prints"
    # layer 1 and layer 3 of the decoder read the frame (input), the sentence block and the printing position (state), and the constant
    D = dec["S"]; frame_or_one = len(dec["X"])
    for L, W, b in (dec["layers"][0], dec["layers"][2]):
        for u, ref, w in W:
            if ref[0] == "S": assert D.names[ref[1]][0] == "ctrl" or D.names[ref[1]][1] in PM.SENT_SLOTS, D.names[ref[1]]
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
    for i, (_, name, alt, v) in forms:
        assert b5.get(i) == -1.0 and sorted(into5[i], key=str) == sorted([(("L4", L4[("gform", name, alt)]), 1.0), (("X", X[("slot", "verb", v)]), 1.0)], key=str)
    into6 = collections.defaultdict(list)
    for tgt, src, w in dec["W6"]: into6[tgt].append((src, w))
    for i, (_, slot, idx, v) in reads: assert into6[S[("out", "s", slot, idx, v)]] == [(("L5", i), 1.0)]
    crashes = [i for i, n in enumerate(L5.names) if n[0] in ("noread", "noform", "noaform")]
    assert crashes and all((("L5", i), 1.0) in into6[S[("ctrl", PM.FAIL)]] for i in crashes)
    for i in crashes:
        ins = dict(into5[i]); assert sum(1 for src, w in ins.items() if w == 1.0 and src[0] == "L4") == 1 and all(w == -1.0 for src, w in ins.items() if src[0] == "X")

def test_output_is_one_hot_morphemes(root):
    """y(t): at every step of the decoder exactly one output unit is on when emit is, none when it is not; every output unit is a
    constant, a slot's value, or a form of a verb or an adjective, and the output lexical layer gives each a string"""
    comp, mp = small_cell(root); S = comp["decoder"]["S"]
    assert {n[1] for n in S.names if n[0] == "out"} == {"c", "s", "vf", "af"}
    for n in S.names:
        if n[0] == "out": assert isinstance(PM.output_lexicon(n), str)
    toks = R.merge_multiword("she took a rest .".split())
    frames, final = mp.encode(toks); seen = []
    def on_step(which, s, x, s2):
        on = [S.names[i] for i in mp.out_units if s2[i] > 0.5]
        assert which == "decoder" and len(on) == (1 if s2[mp.emit_unit] > 0.5 else 0), len(on)
        seen.append(on)
    units = mp.decode(frames, final, on_step=on_step)
    assert [u for on in seen for u in on] == units and len(seen) > len(units) and "".join(PM.output_lexicon(u) for u in units) == "彼女は休んだ。"

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
    opening = [r for r in enc["rules"] if r[2][0][0] == "OPEN"]; sites = {r[2][0][1] for r in opening}
    assert len(sites) >= 2 and all(("link", "that", ("const",)) in out[2] and PM.FLUSH in out[2] and out[1] == 0 for prev, lits, out in opening)
    for sid in sites:
        opens = [r for r in enc["rules"] if r[0] == ("OPEN", sid)]; inner = [r for r in opens if r[2][0] == ("INNER", sid)]
        assert len(inner) == 2 and len(opens) == 3 and {dict(lits)["flag.question"] for _, lits, _ in inner} == {True, False}
        for prev, lits, (cp, k, acts) in inner:
            assert dict(lits)["slot.link=that"] is True and ("clause", "embedded", ("const",)) in acts
            assert (("sent.question", True, ("const",)) in acts) == dict(lits)["flag.question"]
            assert {sl for sl, v, src in acts if src[0] == "clear"} == {sl for sl in V if sl != "clause" and sl not in PM.SENT_SLOTS}, "OPEN does not clear exactly the frame"
        assert [out for _, lits, out in opens if dict(lits)["slot.link=that"] is False] == [(PM.FAIL, 0, ())]
    assert (("CLOSE",), (), (("DONE",), 0, (PM.FLUSH,))) in enc["rules"]
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
