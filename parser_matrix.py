"""The parser and the printer, compiled to sparse matrices: an encoder cell, a context, a decoder cell.

MatrixParser implements the parser interface (parser.ParserInterface) as two recurrent cells, each five sparse ReLU layers and
a readout per step over a one-hot state, with nothing between them but the context.

The encoder reads.  Its input is the lexical layer's output for a 4-token window; its state holds the control point, the
slots of the working frame, the pending preposition and two head units: `shift` tells the window to advance, `flush` hands
the working frame — the slot block of the state — to the context.  Two kinds of rules make it up:
  * clause-level transitions extracted from the rule parser: it is traced on the corpus, each token consumption becomes a
    row (control point, predicates, outcome), and per control point a decision tree over the predicates it read is
    induced and flattened into AND units.  Their guards are only as complete as the corpus.
  * hand-translated modules with complete guards (np_module, pp_module, predicate_module), spliced in as sub-automata: a k=0
    entry transition writes the return site and clears what the module fills; the module's return state hands back through
    an exit transition carrying the caller's effects.
Register writes are factored into copy (slot <- a reading of the current token), const, move / move_time (slot <- another
slot) and clear.  Whatever the encoder cannot represent (a register with no block, an inner clause it has no rules for)
becomes a FAIL row, so the machine refuses there instead of guessing; inside its scope it must agree with the rule parser.

The context is one frame per clause, in the order the clauses closed, and the sentence block of the encoder's final state.
A clause boundary writes a mark into the frame, flushes it, and clears the slots for the next clause to be read into
(clause_rules): a that-clause opening (link), a subordinate clause opening after its main clause or closing before it
(sublink; the subordinate frame carries its conjunction and what the conjunction changes in it); the end of the sentence
flushes the last frame.

The decoder prints (compile_decoder).  Its input is the frame under the attention head; its state holds the printing
position, the sentence block, and what a step puts out.  The printing rules of gen_module are its rules: one emits a token
— a constant morpheme (its own output unit), a slot read (a read gate ORed from the rules, one read unit per value, AND of
gate and the frame's value, mirroring the encoder's write units) or a form lookup (a form gate, one lookup unit per verb or
adjective value) — and raises `emit`, as the encoder raises `shift`; or it raises `down` or `up`, and the attention head
moves to the next or the previous frame.  The readout sums the read and lookup units into the output units y(t), one-hot,
and the print head appends the unit's string (output_lexicon).  A read with nothing in the block, or a form the verb has
not, fires a crash unit into FAIL.
"""
import sys, collections, itertools
import numpy as np
import scipy.sparse as sp
import rules_ja as R
import gen_module as GM
import parser_rules as PRM
from parser_rules import Parser, Clause, ParseFail, Overflow, parse_bounded

WINDOW = 4
NP_ROLES = ("subj", "obj", "iobj", "pp1", "pp2", "pp3", "cop", "cmp")        # cmp: the noun phrase of a than / as comparison
NP_SUB = ("det", "poss", "num", "adj1", "adj2", "noun", "cls")
FLAGS = ("past", "neg", "prog", "perf", "question", "imp", "please", "lets", "there", "tag", "coord", "sub", "neg_inner", "noless", "perf_had", "from_seen")
ENUMS = ("modal", "want", "degree", "obj_particle", "wh", "time", "adv1", "adv2", "front", "cop_kind", "cop_adj", "cop_adv", "cop_part",
         "pp1_part", "pp2_part", "pp3_part", "cmp_kind", "verb", "purpose", "causative")
UNSUPPORTED_REGS = ("infinitive", "control", "embedded", "purpose_clause", "pre_clause", "manner", "so_that", "howabout",
                    "exclaim", "fixed_out", "relative", "suffix", "folded")
OPEN = {"noun", "verb", "adj", "adv", "time", "num"}
import re as _re
IDENTITY_WORDS = {w for w in _re.findall(r'"([a-z][a-z\' ]*)"', open(__file__.replace("parser_matrix.py", "parser_rules.py"), encoding="utf-8").read())}   # words the parser tests by identity
CLOSED_CLASSES = {"prep", "det", "pron", "conj", "modal", "aux_be", "aux_do", "aux_have", "aux_be3", "wh", "not", "end", "please", "int", "yesno", "there", "let", "to"}

class OutOfScope(Exception):
    pass

# ----------------------------------------------------------------------------------------------- lexical layer output
def readings_of(tok):
    return R.classify(tok) if tok is not None else [("END", None)]

def window_features(toks, i):
    """what the lexical layer hands the parser for slots 0..3: per slot, the surface for closed-class tokens, the
    reading classes, and the class-level extras (verb classes, noun class, adjective kind, the adverb's Japanese, membership
    in the duration and hearsay tables)"""
    feats = []
    for k in range(WINDOW):
        t = toks[i + k] if i + k < len(toks) else None
        rs = readings_of(t); classes = tuple(sorted(set(c for c, _ in rs)))
        surface = t if (t is None or not (set(classes) & OPEN) or (set(classes) & CLOSED_CLASSES) or t in IDENTITY_WORDS) else "*"
        extra = []; first_verb = next((v for c, v in rs if c == "verb"), None)
        if first_verb: extra.append(("vform1", first_verb[1]))                      # the first verb reading's form: what parser_rules.verb_form reads
        for c, v in rs:
            if c == "verb":
                extra.append(("vc", tuple(sorted(cls for cls, s in R.VC.items() if v[0] in s)))); extra.append(("vform", v[1]))
            elif c == "noun": extra.append(("ncls", v[1]))
            elif c == "adj": extra.append(("akind", v[1]))
            elif c == "pron": extra.append(("prole", v[1]))
            elif c == "adv": extra.append(("adv", v))
        if t in R.DURATION: extra.append(("dur", "1"))
        if t in R.HEARSAY: extra.append(("hearsay", "1"))
        feats.append((surface, classes, tuple(sorted(set(extra), key=str))))
    return tuple(feats)

def window_sources(toks, i):
    """copy sources spanning more than one token, read at slot 0: a duration phrase (two years -> ２年間, in 50 minutes -> ５０分で)"""
    out = {}
    t0 = toks[i] if i < len(toks) else None; t1 = toks[i + 1] if i + 1 < len(toks) else None; t2 = toks[i + 2] if i + 2 < len(toks) else None
    n0 = dict(readings_of(t0)).get("num") if t0 else None
    if t1 in R.DURATION and (n0 or t0 in DUR_DETS): out["dur"] = (n0 or DUR_DETS[t0]) + R.DURATION[t1]
    if t0 == "in" and t1 and t2 in R.DURATION and t2 not in ("times", "time") and dict(readings_of(t1)).get("num"): out["dur_in"] = dict(readings_of(t1))["num"] + R.DURATION[t2] + "で"
    return out
DUR_DETS = {"a": "一", "an": "一", "several": "数", "many": "何", "few": "数", "some": "数"}

def window_source_vocab():
    return {"dur": sorted({n + d for n in list(R.NUMBERS.values()) + list(DUR_DETS.values()) for d in set(R.DURATION.values())}),
            "dur_in": sorted({n + d + "で" for n in R.NUMBERS.values() for d in set(R.DURATION.values())})}

def table_hits(toks, i, verb):
    """table lookups that join the frame's verb with the window: idiom / phrasal / tail / use rows"""
    hits = []
    t0 = toks[i] if i < len(toks) else None
    lemmas = [v[0] for c, v in readings_of(t0) if c == "verb"]
    for lem in lemmas:
        for k in (3, 2, 1):
            key = (lem, tuple(toks[i + 1:i + 1 + k]))
            if key in R.VERB_IDIOM and i + 1 + k <= len(toks): hits.append(("idiom", k))
        if i + 1 < len(toks) and (lem, toks[i + 1]) in R.PHRASAL: hits.append(("phrasal",))
    if verb:
        for k in (3, 2, 1):
            tt = tuple(toks[i:i + k])
            keyed = tuple("SELF" if (t in R.PRONOUNS and R.PRONOUNS[t][1] == "r") else t for t in tt)
            if (verb[0], keyed) in R.IDIOM_TAIL: hits.append(("tail", k))
        if t0 is not None and (verb[0], t0) in R.PARTICLE_AFTER: hits.append(("pafter",))
        if t0 in R.USE and verb[0] in R.USE[t0]: hits.append(("use",))
    if t0 in R.SOFTENER: hits.append(("softener",))
    if i + 1 < len(toks) and (t0, toks[i + 1]) in R.ADJ_PREP: hits.append(("adjprep",))
    return tuple(sorted(set(hits)))

VERB_HITS = ("hit=tail,", "hit=pafter", "hit=use")     # the hits the frame's verb decides: computed inside the cell (hit_rows), not handed to it
def verb_hit(q): return q.startswith(VERB_HITS)
def window_hits(toks, i):
    """the hits the window alone decides (idiom, phrasal, softener, adjprep): the lexical layer's part of the input"""
    return table_hits(toks, i, None)

def hit_rows(lemma):
    """the table rows of one verb lemma that the cell joins with the window: (hit predicate, ((slot, (kind, value)), ...)) — the
    idiom tails over slots 0..k-1 (a reflexive pronoun as prole=r), the particle after the verb and the use nouns in slot 0"""
    rows = []
    for (lem, tt) in R.IDIOM_TAIL:
        if lem == lemma: rows.append((f"hit=tail,{len(tt)}", tuple((j, ("prole", "r") if t == "SELF" else ("tbl", t)) for j, t in enumerate(tt))))
    for (lem, t0) in R.PARTICLE_AFTER:
        if lem == lemma: rows.append(("hit=pafter", ((0, ("tbl", t0)),)))
    for t0, d in R.USE.items():
        if lemma in d: rows.append(("hit=use", ((0, ("tbl", t0)),)))
    return rows

# ----------------------------------------------------------------------------------------------- frame as slots
def snapshot(c, skip=()):
    """the frame's registers as a flat dict of slot -> value (strings/ids), raising OutOfScope for a register the cell has no
    block for (those in `skip` are not checked: the trace reads the clause a that-clause leaves behind with skip=("embedded",))"""
    for r in UNSUPPORTED_REGS:
        if r in skip: continue
        v = getattr(c, r)
        if v not in (None, "", False, []) : raise OutOfScope(f"register {r}")
    s = {}
    for f in FLAGS: s[f] = bool(getattr(c, f, False))
    s["modal"] = c.modal; s["want"] = c.want; s["degree"] = c.degree or None; s["obj_particle"] = c.obj_particle
    s["wh"] = c.wh; s["time"] = c.time or None; s["purpose"] = c.purpose; s["causative"] = c.causative
    s["verb"] = c.verb
    advs = [a for a in (c.adv,) if a]   # adv is an accumulated string; split on known adverb boundaries is impossible, so it is one slot value
    s["adv1"] = c.adv or None; s["adv2"] = None; s["front"] = c.front or None
    s["subj"] = c.subj; s["subj_cls"] = c.subj_cls if c.subj else None; s["subj_num"] = c.subj_num if c.subj else None
    s["obj"] = c.obj; s["iobj"] = c.iobj
    for k in range(3):
        if k < len(c.pp): s[f"pp{k+1}_part"], s[f"pp{k+1}"] = c.pp[k][0], c.pp[k][1]
        else: s[f"pp{k+1}_part"] = s[f"pp{k+1}"] = None
    if len(c.pp) > 3: raise OutOfScope("four PPs")
    kind = c.copula[0] if c.copula else None
    s["cop_kind"] = kind; s["cop_adj"] = c.copula[1] if kind == "adj" else None; s["cop_adv"] = c.copula[1] if kind == "adv" else None
    s["cop_part"] = c.copula[1] if kind == "pp" else None
    s["cop"] = c.copula[1] if kind == "np" else (c.copula[2] if kind == "pp" else None)
    s["cmp_kind"] = c.compare[0] if c.compare else None; s["cmp"] = c.compare[1] if c.compare else None
    if kind not in (None, "adj", "np", "pp", "adv", "wh"): raise OutOfScope(f"copula kind {kind}")
    return s

FLUSH = ("flush", None, ("flush",))    # the action of a rule that hands the working frame to the context: the flush unit, as shift is for a token
SENT_SLOTS = ("sent.question",)        # the sentence block: registers that outlive a clause boundary and reach the decoder with the encoder's final state

INNER_KINDS = ("embedded", "before", "after")      # Clause.inner_of of the inner clauses that get a frame of their own
MARKS = ("clause", "conj", "link", "sublink")      # the slots that say what a frame is to its neighbours in the context

def view_clause(p, st):
    """the clause whose registers the encoder's working frame holds (st: the trace's state).  The main clause until an inner
    clause with a frame of its own opens: a that-clause (Clause.inner_of == "embedded"), then that clause and from then on the
    innermost one — nothing of an outer clause is read after its that-clause; a subordinate clause before its main clause
    ("before"), until the parser is back in the main clause; a subordinate clause after it ("after"), and from then on that
    clause — the main clause's frame has left.  Raises OutOfScope for any other inner frame (control, purpose, a coordinated
    clause, a question clause)"""
    c = p.c; kind = getattr(c, "inner_of", None)
    if c is not p.main and kind not in INNER_KINDS: raise OutOfScope("inner frame")
    if kind == "after": st["sticky"] = c
    if c is p.main and st.get("sticky") is not None: c = st["sticky"]
    while c.embedded is not None: c = c.embedded
    return c

def clause_view(p, c, st, left=False):
    """the working frame as the trace sees it: the registers of the clause in view and its marks — for an inner clause its kind
    (`clause`) and, for a subordinate one, its conjunction (`conj`); for a main clause whose subordinate clause has closed
    before it, `sublink` — and the sentence block: sent.question, the main clause's question flag, in a frame read after the
    main clause's (the main frame took its own copy with it into the context, and the question mark is read after it left).
    left: the clause as it is left behind at a boundary (its that-clause is the next frame, not a register).  A main clause's
    pre_clause is no register when it is exactly the string of the one subordinate clause the trace saw closed (st["subord"])"""
    skip = ("embedded",) if left else ()
    if c is p.main and c.pre_clause and c.pre_clause == st.get("subord"): skip += ("pre_clause",)
    s = snapshot(c, skip)
    kind = getattr(c, "inner_of", None) if c is not p.main else None
    s["clause"] = kind
    s["conj"] = c.conj if kind in ("before", "after") else None
    s["sublink"] = "before" if c is p.main and any(o == "before" for o, _, _ in c.subords) else None
    s["sent.question"] = bool(p.main.question) if c is not p.main else False
    return s

def edge_kind(p, old, new, st):
    """the kind of clause boundary between two clauses in view: "that" (a that-clause opens in `old`), "before" / "after" (a
    subordinate clause opens before / after the main clause `old`), "main" (the subordinate clause that came first has closed
    and the main clause begins).  Raises OutOfScope for a boundary the frames cannot hold: a that-clause inside a subordinate
    clause, a second subordinate clause, one that opens when the main clause has already begun"""
    k_old, k_new = getattr(old, "inner_of", None), getattr(new, "inner_of", None)
    if k_new == "embedded":
        if k_old in ("before", "after"): raise OutOfScope("a that-clause inside a subordinate clause")
        return "that"
    if k_new in ("before", "after"):
        if old is not p.main: raise OutOfScope("a subordinate clause inside a clause")
        if p.main.subords: raise OutOfScope("a second subordinate clause")
        if k_new == "before" and clause_view(p, p.main, st) != st["initial"]: raise OutOfScope("a subordinate clause after the start of its main clause")
        return k_new
    if new is p.main and k_old == "before": return "main"
    raise OutOfScope("clause boundary")

# ----------------------------------------------------------------------------------------------- tracing the rule parser
PMETH = set(n for n in dir(Parser) if callable(getattr(Parser, n)) and not n.startswith("__"))

def trace(toks, choices=()):
    """run the rule parser and return the list of consumption events with before-state and register diffs"""
    p = Parser(list(toks), list(choices))
    events = []            # dicts: i (token index consumed), k (how many), cp, snap_before, np spans
    np_spans = []          # (i0, i1, ja, cls, depth_of_c)
    st = {"last_i": 0, "line": 0, "cp": None, "frames": (), "sticky": None, "subord": None, "left": None, "edge": None}
    def chain_of(frame):
        """the Parser-method frames on the stack, outermost first, as (method, current line)"""
        names = []; f = frame
        while f is not None:
            if f.f_code.co_name in PMETH and f.f_locals.get("self") is p: names.append((f.f_code.co_name, f.f_lineno))
            f = f.f_back
        return tuple(reversed(names))
    def frames_of(frame):
        """the same frames as (method, frame id, parameters of a module method), outermost first"""
        out = []; f = frame
        while f is not None:
            if f.f_code.co_name in PMETH and f.f_locals.get("self") is p:
                out.append((f.f_code.co_name, id(f), {k: f.f_locals.get(k) for k in MODULE_PARAMS.get(f.f_code.co_name, ())}))
            f = f.f_back
        return tuple(reversed(out))
    def stack_of(frame): return tuple(n for n, _ in chain_of(frame))
    def tracer(frame, event, arg):
        if frame.f_code.co_filename != R.__file__ and not frame.f_code.co_filename.endswith("parser_rules.py"): return tracer
        if event == "line":
            if p.i != st["last_i"]:
                # the consumption happened in the code of the previous line event: that frame's module owns it (the change is noticed
                # one statement later, which may already be inside a method called on the same line)
                # a conjunction consumed by sub_clause, noticed before the method has made the clause it opens: the event is the
                # opening of a subordinate clause, before or after its main clause as the caller said
                opens = frame.f_locals.get("order") if frame.f_code.co_name == "sub_clause" and "inner" not in frame.f_locals else None
                events.append({"i": st["last_i"], "k": p.i - st["last_i"], "cp": (stack_of(frame), st["line"]), "snap": st["snap"], "outer": st["outer"], "edge": st["edge"], "opens": opens,
                               "chain": chain_of(frame), "owner": owner_of(st["frames"]), "frames": st["frames"], "win_reads": set(reads["win"]), "reg_reads": set(reads["reg"])})
                reads["win"].clear(); reads["reg"].clear()
                st["last_i"] = p.i; st["event_vc"] = st["vc"]
            st["line"] = frame.f_lineno; st["frames"] = frames_of(frame)
            if type(p.c) is Clause: p.c.__class__ = TracedClause           # an inner clause's register reads count like the main clause's
            try:
                vc = view_clause(p, st); st["snap"] = clause_view(p, vc, st)
                # the clause in view is not the one the last event was read into: a clause boundary of kind `edge`, and `outer` is the
                # clause left behind as it stands now — the frame the encoder flushes
                if vc is not st["event_vc"]:
                    st["edge"] = edge_kind(p, st["event_vc"], vc, st); st["outer"] = clause_view(p, st["event_vc"], st, left=True)
                    if st["edge"] == "after" and st["left"] is None: st["left"] = left_view(p, st)
                else: st["outer"] = st["edge"] = None
                if st["left"] is not None and left_view(p, st) != st["left"]: raise OutOfScope("the main clause changes after its subordinate clause")
                st["vc"] = vc
            except OutOfScope as e: st["snap"], st["outer"], st["edge"] = e, None, None
        return tracer
    def left_view(p, st):
        """the main clause a subordinate clause follows, as its frame left: nothing but the question mark may reach it afterwards"""
        return {k: v for k, v in clause_view(p, p.main, st, left=True).items() if k != "question"}
    reads = {"win": set(), "reg": set()}
    orig_peek = Parser.peek
    def peek_wrapped(self, k=0):
        reads["win"].add(self.i + k); return orig_peek(self, k)
    Parser.peek = peek_wrapped
    class TracedClause(Clause):
        def __getattribute__(self, name):
            if not name.startswith("_") and name != "depth": reads["reg"].add(name)
            return object.__getattribute__(self, name)
    p.main.__class__ = TracedClause
    orig_np = Parser.parse_np
    def np_wrapped(self, role=None):
        i0 = self.i
        ja, cls = orig_np(self, role=role)
        np_spans.append((i0, self.i, ja, cls, self.c is self.main or getattr(self.c, "inner_of", None) in INNER_KINDS))      # in a clause with a frame
        heads.append((i0, self.i, self.last_head))
        return ja, cls
    heads = []
    Parser.parse_np = np_wrapped
    orig_subord = PRM.subord_form
    def subord_wrapped(inner, conj):
        """the string of a subordinate clause, noted for the views (st["subord"]); what generate() reads of the clause on the way
        is no read of the parser's"""
        saved = (set(reads["win"]), set(reads["reg"]))
        try: out = orig_subord(inner, conj)
        finally:
            for key, old in zip(("win", "reg"), saved): reads[key].clear(); reads[key].update(old)
        st["subord"] = out
        return out
    PRM.subord_form = subord_wrapped
    st["vc"] = st["event_vc"] = p.main; st["snap"] = st["initial"] = initial = clause_view(p, p.main, st); st["outer"] = None
    sys.settrace(tracer); err = None
    try:
        frame = p.parse()
    except ParseFail as e:
        err = e; frame = None
    finally:
        sys.settrace(None); Parser.parse_np = orig_np; Parser.peek = orig_peek; PRM.subord_form = orig_subord
    p.fail_info = None; _last_parser[0] = p
    p.last_cp = events[-1]["cp"] if events else ("START",)
    p.events_ok = all(not isinstance(e["snap"], (OutOfScope, type(None))) for e in events)
    p.pend_at_end = None
    p.events, p.np_spans, p.heads = events, np_spans, heads
    for j in range(p.i):
        if j < len(p.toks) and "prep" in dict(readings_of(p.toks[j])): p.pend_at_end = p.toks[j]
    if err is not None:
        try: fs = clause_view(p, view_clause(p, st), st)          # the view where the parser refused: inside an inner clause, that clause's
        except OutOfScope as e2: fs = e2
        p.fail_info = (err, fs, set(reads["win"]), set(reads["reg"]), p.i)
        tb = err.__traceback__; p.fail_stack = []; p.fail_frames = []
        while tb is not None:
            if tb.tb_frame.f_code.co_name in PMETH and tb.tb_frame.f_locals.get("self") is p: p.fail_stack.append(tb.tb_frame.f_code.co_name); p.fail_frames.append(id(tb.tb_frame))
            tb = tb.tb_next
        if events: events[0]["snap"] = initial
        raise err
    if events: events[-1]["win_reads"] |= reads["win"]; events[-1]["reg_reads"] |= reads["reg"]
    if events: events[0]["snap"] = initial          # writes made before the first consumption belong to the first step
    try:
        final = clause_view(p, view_clause(p, st), st)        # with an inner clause read last: the last frame, that clause's, with the sentence block
        if st["left"] is not None and left_view(p, st) != st["left"]: raise OutOfScope("the main clause changes after its subordinate clause")
    except OutOfScope as e: final = e                 # the finished frame holds a register the cell has no block for
    return events, final, np_spans, p

# ----------------------------------------------------------------------------------------------- the hand-translated clause modules
# method of the rule parser -> module name; the module's rules (pp_module.py, predicate_module.py) replace the extracted rules
# for the consumption events the method's activations own
MODULE_METHODS = {"predicate": "pred", "copula_predicate": "cop", "tail_pp_adv": "pp"}
MODULE_START = {"pred": ("M", "pred", "PR0"), "cop": ("M", "cop", "CP0"), "pp": ("M", "pp", "PP0")}
MODULE_PARAMS = {"copula_predicate": ("allow_empty",)}          # a method's parameter the module reads as the slot param.<name>
STRUCTURAL = {"parse", "pre", "after_wh", "question_inversion", "predicate", "copula_predicate", "verb_phrase", "tail", "tail_pp_adv", "end",
              "sub_clause", "coord", "embedded_question", "embedded_after", "predicate_noend", "parse_np", "parse_np_rest_after_poss"}
NP_METHODS = ("parse_np", "parse_np_rest_after_poss")

def owner_of(frames):
    """the module activation that owns a consumption event — (module, frame id, params) — or None: the innermost structural method
    that is a module method, with no other structural method between it and the event; a noun phrase inside a module belongs to
    the module, a noun phrase anywhere else is the noun-phrase module's own span"""
    for name, fid, params in reversed([f for f in frames if f[0] in STRUCTURAL]):
        if name in NP_METHODS: continue
        if name in MODULE_METHODS: return (MODULE_METHODS[name], fid, params)
        return None
    return None

_modules = {"on": False, "by_prev": None, "derived": None}       # set by build_hybrid: the compiled rules the simulation steps

def failed_in_module(fail_stack):
    """the parser's refusal happened in a clause module's own code: the innermost structural method on the failing stack (a noun
    phrase inside the module counts as the module's) is a module method"""
    structural = [n for n in fail_stack if n in STRUCTURAL]
    while structural and structural[-1] in NP_METHODS: structural.pop()
    return bool(structural) and structural[-1] in MODULE_METHODS

def clause_module_rules():
    """the three modules' rules and derived predicates in the compiler's format: states ("M", module, name), RET -> ("RETM", module),
    FAIL:<why> -> FAIL, a tuple target (a noun-phrase entry) as it is"""
    import pp_module as PPM, predicate_module as PDM
    rules = []; derived = {}
    for mod, mrules, D in (("pp", PPM.rules(), PPM.derived()), ("pred", PDM.pred_rules(), PDM.pred_derived()), ("cop", PDM.cop_rules(), PDM.cop_derived())):
        derived.update(D)
        for state, lits, (nxt, k, acts) in mrules:
            prev = state if isinstance(state, tuple) else ("M", mod, state)
            if isinstance(nxt, tuple): tgt = nxt
            elif nxt == "RET": tgt = ("RETM", mod, "")                               # (not ("RET", mod): "cop" is also a noun-phrase role)
            elif nxt == "RETCOP": tgt = ("RETM", mod, "cop")                         # the predicate module's second exit: the copula follows
            elif nxt == "RETSUB": tgt = ("RETM", mod, "sub")                         # the PP machine's second exit: a subordinate clause follows
            elif nxt.startswith("FAIL:"): tgt = FAIL
            else: tgt = ("M", mod, nxt)
            rules.append((prev, tuple((l[1:], False) if l.startswith("¬") else (l, True) for l in lits), (tgt, k, norm_actions(acts))))
    return rules, derived

def snapshot_to_frame(snap, classes):
    """a trace snapshot (the register view) -> a flat frame for the simulation: flags and enums as slots, every noun-phrase role as a
    stand-in head noun (its whole string) with the class the trace recorded for it"""
    frame = {}
    for f in FLAGS:
        if snap.get(f): frame[f] = True
    for k in ("modal", "want", "degree", "obj_particle", "wh", "time", "adv1", "front", "cop_kind", "cop_adj", "cop_adv", "cop_part", "pp1_part", "pp2_part", "pp3_part", "cmp_kind", "verb", "purpose", "causative"):
        v = snap.get(k)
        if v not in (None, "", False): frame[k] = v
    for role in NP_ROLES:
        s = snap.get(role)
        if s: frame[f"{role}.noun"] = s; frame[f"{role}.cls"] = snap.get("subj_cls") if role == "subj" else classes.get(role, "t")
    if snap.get("subj_num") and snap.get("subj"): frame["subj.number"] = snap["subj_num"]
    for k in MARKS + SENT_SLOTS:                                                  # the frame's marks and the sentence block, as they are
        if snap.get(k) not in (None, "", False): frame[k] = snap[k]
    return frame

def step_rules(by_prev, derived, state, toks, i, frame, pend):
    """the unique rule of `state` whose literals hold on the flat frame and the window at i -> (cp, k, actions); raises ParseFail"""
    P = predicates(toks, i, frame_to_snapshot(frame), set(range(i, i + WINDOW)), ALL_REGS | {"ret", "mret"}, pend, heads_of(frame)) | slot_predicates(frame)
    P.discard("agree.bad")                       # read as the derived predicate, from the slot and the window
    hits = [out for lits, out in by_prev.get(state, []) if all(eval_rule_lit(q, t, P, derived, frame) for q, t in lits)]
    if len(hits) != 1: raise ParseFail(f"RULES: {len(hits)} rules fire at {state} token {i}", i)
    return hits[0]

def exit_kind(mod, events, last):
    """which exit the parser took from a module activation, from the trace: the predicate module returns to the copula ("cop") or to
    the verb phrase ("") — by what the innermost predicate frame of the next event called (an outer clause's predicate is on the
    stack of every event of a clause opened inside it); the PP machine returns at a subordinating conjunction ("sub": the next
    event is the conjunction, consumed by sub_clause called from the tail) or at the end of its clause (""); the copula module
    has one exit"""
    if last + 1 >= len(events): return ""
    if mod == "pp": return "sub" if events[last + 1]["cp"][0][-2:] == ("tail_pp_adv", "sub_clause") else ""
    if mod != "pred": return ""
    chain = [n for n, _ in events[last + 1]["chain"]]
    inner = max((j for j, n in enumerate(chain) if n == "predicate"), default=None)
    return "cop" if inner is not None and chain[inner + 1:inner + 2] == ["copula_predicate"] else ""

def simulate_module(mod, toks, i0, frame, pend, bound=400):
    """step a module's rules (and the noun-phrase module's, when it hands over) from its start on a flat frame; -> (i, frame, pend,
    exit kind) at its return, or None when it refuses or the rules do not decide"""
    by_prev, derived = _modules["by_prev"], _modules["derived"]
    state, i, frame = MODULE_START[mod], i0, dict(frame)
    for _ in range(bound):
        if isinstance(state, tuple) and state[:2] == ("RETM", mod): return i, frame, pend, state[2]
        if state == FAIL: return None
        try: cp, k, actions = step_rules(by_prev, derived, state, toks, i, frame, pend)
        except ParseFail: return None
        try: apply_actions(frame, actions, toks, i)
        except ParseFail: return None
        for j in range(i, i + k):
            if j < len(toks) and "prep" in dict(readings_of(toks[j])): pend = toks[j]
        i += k; state = cp
    return None

# ----------------------------------------------------------------------------------------------- predicates and induction
ENUM_REGS = ("modal", "want", "cop_kind", "degree", "obj_particle", "wh", "purpose", "causative", "subj_cls", "time", "adv1", "front", "cmp_kind", "ret", "mret")
# subject-verb agreement as one predicate over the subject's number and the current token (parser_rules.agrees, mirrored):
# true exactly when no verb reading of the token fits the subject.  Compiled as AND units over the slot and the window
# and read by the FAIL rules that agreement_fail_rows adds at every finite-verb consumption.
AGREE_DNF = [["slot.subj.number=3sg", "w0.vform=base", "¬w0.vform=past", "¬w0.vform=3sg", "¬w0.class=aux_be"],
             ["slot.subj.number=non3sg", "w0.vform=3sg", "¬w0.vform=base", "¬w0.vform=past", "¬w0.class=aux_be"],
             ["slot.subj.number=3sg", "w0.class=aux_be", "w0.surf=are"],
             ["slot.subj.number=non3sg", "w0.class=aux_be", "w0.surf=is"]]
def agree_bad(P):
    """evaluate AGREE_DNF over a predicate set that holds the window predicates and slot.subj.number=<v>"""
    return any(all((l[1:] not in P) if l.startswith("¬") else (l in P) for l in conj) for conj in AGREE_DNF)
NOUN_EN = collections.defaultdict(set)         # Japanese noun -> the English tokens that read as it (for the USE feedback predicate)
for _tok in R.LEX:
    for _c, _v in R.classify(_tok):
        if _c == "noun": NOUN_EN[_v[0]].add(_tok)
def use_pred(verb, head_ja):
    """the semantic feedback edge: the head noun carries a use-verb for the frame's verb (take a rest -> 休む)"""
    return bool(verb) and any(verb[0] in R.USE.get(en, {}) for en in NOUN_EN.get(head_ja, ()))

def build_use_table():
    """(head noun in Japanese, verb lemma) -> (the verb after the feedback, particle on the noun, stative): tables/use.tbl read
    through the lexicon, so that the feedback is one lookup unit per pair in the cell instead of a constant in every rule
    that happens to fire on it.  Two English nouns with the same Japanese and different feedback would collide; the first in
    sorted order wins and the other pairs stay constants in their rules (hybrid_rows folds only an exact match)."""
    table = {}
    for en in sorted(R.USE):
        for lemma, (uja, uconj, upart, stative) in R.USE[en].items():
            for ja in sorted({v[0] for c, v in R.classify(en) if c == "noun"}):
                table.setdefault((ja, lemma), ((lemma + "+" + en, uja, uconj), upart, stative))
    return table
USE_TABLE = build_use_table()

def fold_use(post, role, head_ja, verb):
    """the post-NP effects of the USE feedback (verb <- the pair's verb, the particle, the drop of the noun for "-", prog when
    stative), when they are exactly what USE_TABLE prescribes for this head and verb, folded into one action (role, None, ("use",))"""
    hit = USE_TABLE.get((head_ja, verb[0])) if head_ja and verb else None
    if hit is None: return post
    new_verb, upart, stative = hit
    expected = {("verb", new_verb, ("const",))}
    optional = set()
    if upart == "-": expected |= {(sl, None, ("clear",)) for sl in np_slots(role)}; optional = {("obj_particle", None, ("clear",))}
    elif role.startswith("pp"): expected.add((f"{role}_part", (upart, upart), ("const",)))      # came to a decision: the particle of that PP
    else: expected.add(("obj_particle", upart, ("const",)))
    if stative: expected.add(("prog", True, ("const",)))
    have = set(post)
    if not expected <= have: return post
    return [a for a in post if a not in expected and a not in optional] + [(role, None, ("use",))]

def predicates(toks, i, before, win_idx, reg_rd, pend_prep, heads=None):
    """binary predicates available to guard the transition at token i: only over what the parser read"""
    P = set()
    for role, head in (heads or {}).items():
        if head and use_pred(before.get("verb"), head): P.add(f"use.{role}")
    for role in NP_ROLES:                                             # a clock-time noun phrase (at 5 -> ５時に)
        v = before.get(role)
        if isinstance(v, str) and v.endswith("時"): P.add(f"np.{role}.time")
    allw = window_features(toks, i)
    for j in sorted(win_idx):
        k = j - i
        if not (0 <= k < WINDOW): continue
        surface, classes, extra = allw[k]
        for c in classes: P.add(f"w{k}.class={c}")
        if surface != "*": P.add(f"w{k}.surf={surface}")
        for e in extra:
            if e[0] == "vc":
                for cls in e[1]: P.add(f"w{k}.vc={cls}")
            else: P.add(f"w{k}.{e[0]}={e[1]}")
    for h in table_hits(toks, i, before["verb"]): P.add("hit=" + ",".join(map(str, h)))
    if i + 2 < len(toks) and toks[i] == toks[i + 2]: P.add("w0=w2")                     # longer and longer
    for rname in reg_rd:
        if rname == "subj":
            if before.get("subj") is not None: P.add("reg.subj")
            if before.get("subj") == "それ": P.add("subj.sore")
        elif rname in ("obj", "iobj", "cop", "cmp", "compare"):
            key = "cmp" if rname == "compare" else rname
            if before.get(key) is not None: P.add(f"reg.{key}")
        elif rname == "pp":
            n = len([k for k in ("pp1", "pp2", "pp3") if before.get(k)]); P.add(f"pp.count={n}")
        elif rname == "copula":
            if before.get("cop_kind"): P.add(f"reg.cop_kind={before['cop_kind']}"); P.add("reg.cop_kind")
        elif rname == "verb":
            vb = before["verb"]
            if vb:
                P.add("verb.set"); P.add("reg.verb"); P.add(f"verb.conj={vb[2]}")
                if " " in vb[0]: P.add("verb.phrasal")
                if "+" in vb[0]: P.add("verb.use")
                for cls, s in R.VC.items():
                    if R.lemma_of(vb) in s: P.add(f"verb.vc={cls}")
        elif rname in FLAGS:
            if before.get(rname): P.add(f"flag.{rname}")
        elif rname in ENUM_REGS or rname == "adv":
            key = {"adv": "adv1"}.get(rname, rname); v = before.get(key)
            if v not in (None, "", False): P.add(f"reg.{rname}={'set' if rname in ('time', 'adv', 'adv1', 'front') else v}")
    if pend_prep: P.add(f"pend_prep={pend_prep}")
    if before.get("subj_num") and "subj_num" in reg_rd:
        full = predicates(toks, i, before, {i}, set(), None) | {f"slot.subj.number={before['subj_num']}"}
        if agree_bad(full): P.add("agree.bad")
    return P

FAIL = ("FAIL",)
ALL_REGS = {"subj", "obj", "iobj", "cop", "cmp", "pp", "copula", "verb", "adv", "time", "subj_num"} | set(FLAGS) | set(ENUM_REGS)
_last_parser = [None]

# ----------------------------------------------------------------------------------------------- the hybrid: parse_np as a hand-translated module
NP_SUBS = ("det", "poss", "num", "adj1", "adj2", "noun", "noun2", "cls", "mono", "compound")
def np_slots(role):
    """every sub-slot of a noun-phrase role, its of-phrase's included"""
    return [f"{role}.{sub}" for sub in NP_SUBS] + [f"{role}of.{sub}" for sub in NP_SUBS]

def np_site(chain):
    """the return address of a noun phrase: (caller methods, line of the parse_np call) for the outermost parse_np frame"""
    for idx, (name, line) in enumerate(chain):
        if name == "parse_np":
            return (tuple(n for n, _ in chain[:idx]), chain[idx - 1][1] if idx > 0 else 0)
    return None

def site_id(site):
    return ">".join(site[0][1:] or ("parse",)) + ":" + str(site[1])

def views_of(events, snaps):
    """(befores, afters): the view before and after each event.  afters[k] is the view before event k+1, except at a clause
    boundary: the view after the event before the first event of the next clause is the view of the clause left behind (the
    next view begins with the boundary row, which the cell takes before that event)"""
    befores = snaps[:len(events)]; afters = list(snaps[1:len(events) + 1])
    for k, e in enumerate(events):
        if e.get("outer") is not None and k > 0: afters[k - 1] = e["outer"]
    return befores, afters

def np_span_info(events, snaps, np_spans, final_snap):
    """the outermost noun phrases of the clause in view: (first event, last event, role, ja, cls, site, old value of the role, where it moved)"""
    befores, afters = views_of(events, snaps)
    spans = sorted(set((i0, i1) for (i0, i1, ja, cls, main) in np_spans if main))
    outer = [(i0, i1) for (i0, i1) in spans if not any(j0 <= i0 and i1 <= j1 and (j0, j1) != (i0, i1) for (j0, j1) in spans)]
    ja_of = {(i0, i1): (ja, cls) for (i0, i1, ja, cls, main) in np_spans if main}
    def ev_of_tok(j):
        hits = [k for k, e in enumerate(events) if e["i"] <= j < e["i"] + e["k"]]
        if not hits: raise OutOfScope("noun phrase token outside the consumption events")
        return hits[0]
    info = []
    for (i0, i1) in outer:
        if i1 <= i0: raise OutOfScope("empty noun phrase")
        ja, cls = ja_of[(i0, i1)]
        first, last = ev_of_tok(i0), ev_of_tok(i1 - 1)
        role = None; dropped = False
        for ei in range(last, len(events)):
            before, after = befores[ei], afters[ei]
            for r in NP_ROLES:
                if after.get(r) == ja and before.get(r) != ja: role = r; break
            if role: break
        if role is None and afters[last].get("verb") and "+" in afters[last]["verb"][0]:
            role, dropped = "obj", True                       # take a rest -> 休む: the use-verb absorbs the object (USE particle "-")
        if role is None: raise OutOfScope(f"noun phrase {ja!r} not written to a slot")
        site = np_site(events[first]["chain"])
        if site is None: raise OutOfScope("noun phrase without a parse_np frame")
        old = befores[first].get(role); moved_to = None
        if old is not None:
            for r2 in NP_ROLES:
                if r2 != role and afters[last].get(r2) == old and befores[first].get(r2) != old: moved_to = r2
        info.append((first, last, role, ja, cls, site, old, moved_to, dropped))
    return info

def register_actions(before, after, tok, allow_copy=True):
    """the non-NP register diffs between two snapshots as actions; copies from the consumed token's readings when allowed"""
    rs = dict(readings_of(tok)) if allow_copy else {}
    actions = []
    for slot in before:
        if slot in NP_ROLES or slot in ("subj_cls", "subj_num"): continue        # written by the noun-phrase module
        if before[slot] == after.get(slot): continue
        v = after.get(slot)
        if v in (None, False, "", 0): actions.append((slot, None, ("clear",))); continue
        src = ("const",)
        if slot == "verb" and "verb" in rs and v[0] == rs["verb"][0] and v[1] == rs["verb"][2] and v[2] == rs["verb"][3]: src = ("copy", "verb")
        elif slot == "verb" and "adj" in rs and v == ("make+" + tok, R.adj_causative(*rs["adj"]), "i"): src = ("copy", "adj_caus")
        elif slot == "adv1" and "adv" in rs and v == rs["adv"] and before[slot] in (None, ""): src = ("copy", "adv")
        elif slot == "front" and "adv" in rs and v == rs["adv"] + "、" and before[slot] in (None, ""): src = ("copy", "adv_front")
        elif slot == "time" and "time" in rs and v == rs["time"]: src = ("copy", "time")
        elif slot == "wh" and "wh" in rs and v == rs["wh"]: src = ("copy", "wh")
        elif slot == "cop_adj" and "adj" in rs and v == rs["adj"]: src = ("copy", "adj_pair")
        elif slot == "cop_adv" and "adv" in rs and v == rs["adv"]: src = ("copy", "adv")
        elif slot == "modal" and "modal" in rs and v == rs["modal"][0]: src = ("copy", "modal")
        actions.append((slot, v, src))
    return actions

def np_register_actions(before, after, tok, skip_roles):
    """NP-string registers written without parse_np (go swimming -> obj 水泳; than today -> pp 今日) and clears"""
    rs = dict(readings_of(tok)); actions = []
    for r in NP_ROLES:
        if r in skip_roles or after.get(r) == before.get(r): continue
        v = after.get(r)
        if v is None:
            actions += [(sl, None, ("clear",)) for sl in np_slots(r)]; continue
        # a slot acquiring another slot's old string is a move (give him a book: obj -> iobj)
        src_role = next((r2 for r2 in NP_ROLES if r2 != r and before.get(r2) == v), None)
        if src_role:
            actions += [(sl, None, ("move", sl2)) for sl, sl2 in zip(np_slots(r), np_slots(src_role))]; continue
        if "time" in rs and v == rs["time"]: src = ("copy", "time")
        elif "loc" in rs and v == rs["loc"]: src = ("copy", "loc")
        elif "noun" in rs and v == rs["noun"][0]: src = ("copy", "noun")
        elif "pron" in rs and v == rs["pron"][0]: src = ("copy", "pron")
        else: src = ("const",)
        actions.append((f"{r}.noun", v, src)); actions.append((f"{r}.cls", (after.get("subj_cls") or "h") if r == "subj" else "t", ("const",)))   # the class the parser recorded
    return actions

def norm_actions(actions):
    return tuple(sorted(((slot, None if src[0] in ("copy", "move", "move_time") else v, src) for slot, v, src in actions), key=str))

def hybrid_rows(toks):
    """transition rows for the hybrid machine.  Outer transitions are extracted from the trace as before; each main-frame
    noun phrase becomes a hand-over to np_module: an entry row (k=0 into ("NP0", role), the return site written to `ret`,
    the role's slots cleared or moved) and an exit row (from ("RET", role): the post-NP effects, into the caller's continuation).
    Returns (rows, roles used); raises ParseFail / OutOfScope when the sentence is outside the cell's scope."""
    events, final, np_spans, p = trace(toks)
    if isinstance(final, OutOfScope): raise final
    for e in events:
        if isinstance(e["snap"], OutOfScope): raise e["snap"]
    snaps = [e["snap"] for e in events] + [final]
    rows, roles, prev, pend = transition_rows(toks, events, snaps, np_spans, final)
    if prev is None: return rows, roles                   # a clause module refused inside: the sentence ends in the module's refusal
    P_end = frozenset(predicates(toks, len(toks), final, set(range(len(toks), len(toks) + WINDOW)), ALL_REGS, pend))
    rows.append((prev, P_end, frozenset(), (("CLOSE",), 0, ())))          # CLOSE flushes the last frame and is DONE (clause_rules)
    return rows, roles

def module_spans(events):
    """the activations of the clause modules: maximal runs of consecutive events one activation owns, as (first, last, module,
    params); an activation that consumed no token of its own (the copula method before a progressive verb, the predicate method
    before a plain verb) is on the stack of the event that follows it and gets an empty span (last = first - 1) before that event"""
    if not _modules["on"]: return []
    spans = []; seen = set()
    for k, e in enumerate(events):
        o = e.get("owner")
        for name, fid, params in e.get("frames", ()):
            if name in MODULE_METHODS and fid not in seen and (o is None or o[1] != fid) and not any(x.get("owner") and x["owner"][1] == fid for x in events[k:]):
                seen.add(fid); spans.append([k, k - 1, MODULE_METHODS[name], params, fid])
        if o is None: continue
        if spans and spans[-1][1] == k - 1 and spans[-1][4] == o[1]: spans[-1][1] = k
        else: seen.add(o[1]); spans.append([k, k, o[0], o[2], o[1]])
    return [(a, b, m, prm) for a, b, m, prm, fid in spans]

def module_site(chain, method):
    """the return address of a module activation: (caller methods, line of the call), from the event's chain.  The activation's
    own frame is the innermost of its name: the predicate of a that-clause runs inside the outer clause's predicate (predicate >
    verb_phrase > predicate_noend > predicate), and the two activations return to different sites"""
    for idx in range(len(chain) - 1, -1, -1):
        if chain[idx][0] == method: return (tuple(n for n, _ in chain[:idx]), chain[idx - 1][1] if idx > 0 else 0)
    return None

def transition_rows(toks, events, snaps, np_spans, final):
    """the rows of the consumption events given (a whole parse, or the prefix of one up to the point where it stops being
    representable or is refused): -> (rows, roles used, the control state after the last event, the pending preposition).
    A noun phrase outside the clause modules and every activation of a clause module become hand-overs: an entry row into the
    module and an exit row from its RET, carrying what the caller wrote after the module returned."""
    import np_module as NPM
    mspans = module_spans(events); in_mspan = {k: n for n, (first, last, *_) in enumerate(mspans) for k in range(first, last + 1)}
    empty_before = {first: n for n, (first, last, *_) in enumerate(mspans) if last < first}; emitted_empty = set()
    own_np = [sp for sp in np_spans if not any(events[k]["i"] <= sp[0] < events[k]["i"] + events[k]["k"] and k in in_mspan for k in range(len(events)))]
    info = np_span_info(events, snaps, own_np, final)
    try: classes = {role: cls for (first, last, role, ja, cls, *_) in np_span_info(events, snaps, np_spans, final)}       # the class of every noun phrase, for the simulation's stand-ins
    except OutOfScope: classes = {}
    in_span = {k: n for n, (first, last, *_) in enumerate(info) for k in range(first, last + 1)}
    befores, afters = views_of(events, snaps)
    rows = []; roles = set(); prev = ("START",); pend = None
    if not hasattr(hybrid_rows, "module_log"): hybrid_rows.module_log = collections.Counter()
    noreads = {"win_reads": set(), "reg_reads": set()}
    # the clause boundaries: event index of the boundary row -> (kind, the view of the clause left behind, whether the clause rules
    # consume the event themselves).  A boundary is noticed at the first event of the next clause; a subordinate clause opens one
    # event earlier, at its conjunction, which the clause rules read (clause_rules): that event has no row of its own, and it is a
    # boundary by itself — also in a parse that dies on the token after it
    edges = {}; opened = set()
    for k, x in enumerate(events):
        if x.get("opens"): edges[k] = (x["opens"], befores[k], True)
        elif x.get("outer") is not None and x["edge"] not in ("before", "after"): edges[k] = (x["edge"], x["outer"], False)
    ei = 0
    while ei < len(events):
        e = events[ei]
        in_mod = ei in in_mspan or (ei in empty_before and empty_before[ei] not in emitted_empty)
        if in_mod and edges:
            first, last = (mspans[in_mspan[ei]] if ei in in_mspan else mspans[empty_before[ei]])[:2]
            if any(first < b <= last for b in edges):         # the clause opened inside a module activation (the copula's that-clause): the module refuses it
                hybrid_rows.module_log["that-clause inside a module"] += 1
                return rows, roles, None, pend
        if ei in edges and ei not in opened and not (edges[ei][2] and in_mod):       # (at a conjunction, the PP machine the main clause is in returns first)
            # the clause boundary, at a site of its own: the row from the traces only goes to the site's OPEN state, and the clause
            # rules do the rest by hand — mark and flush the frame of the clause left behind, clear the working frame, mark the new one
            # — down to the site's INNER state, where the next clause is read.  A that-clause's row writes the link and flushes itself
            opened.add(ei); kind, left, absorbed = edges[ei]
            site = np_site(e["chain"]); sid = site_id(site) if site else ">".join(e["cp"][0][1:]) + ":" + str(e["cp"][1])    # where the clause opens: its rows are that site's own
            e2 = events[ei + 1] if ei + 1 < len(events) else noreads
            P_read = frozenset(predicates(toks, e["i"], left, set(e["win_reads"]) | set(e2["win_reads"]), set(e["reg_reads"]) | set(e2["reg_reads"]), pend))
            P_all = frozenset(predicates(toks, e["i"], left, set(range(e["i"], e["i"] + WINDOW)), ALL_REGS, pend))
            if kind == "after": P_read = frozenset()          # reached from the PP machine's second exit alone, and OPEN reads the conjunction itself (d.clause.trail)
            rows.append((prev, P_all, P_read, (("OPEN", kind, sid), 0, norm_actions([("link", "that", ("const",)), FLUSH] if kind == "that" else []))))
            prev = ("INNER", sid)
            if absorbed:                                      # the conjunction: read by the clause rule on the way to INNER
                if "prep" in dict(readings_of(toks[e["i"]])): pend = toks[e["i"]]
                ei += 1; continue
        if in_mod:
            empty = ei not in in_mspan
            if empty: emitted_empty.add(empty_before[ei])
            first, last, mod, params = mspans[in_mspan[ei]] if not empty else mspans[empty_before[ei]]
            method = next(m for m, name in MODULE_METHODS.items() if name == mod)
            site = module_site(events[first]["chain"], method); sid = (mod + "@" + site_id(site)) if site else mod
            acts = [("mret", sid, ("const",))] + [(f"param.{k}", True, ("const",)) for k, v in (params or {}).items() if v]
            e1 = events[first]; e2 = events[first + 1] if first + 1 < len(events) else noreads
            P_read = frozenset(predicates(toks, e1["i"], befores[first], set(e1["win_reads"]) | set(e2["win_reads"]), set(e1["reg_reads"]) | set(e2["reg_reads"]), pend))
            P_all = frozenset(predicates(toks, e1["i"], befores[first], set(range(e1["i"], e1["i"] + WINDOW)), ALL_REGS, pend))
            rows.append((prev, P_all, P_read, (MODULE_START[mod], 0, norm_actions(acts))))
            roles |= {"pp1", "pp2", "pp3", "cmp", "cop"}
            i1 = events[last]["i"] + events[last]["k"] if not empty else e1["i"]
            frame0 = snapshot_to_frame(befores[first], classes); frame0["mret"] = sid
            for k, v in (params or {}).items():
                if v: frame0[f"param.{k}"] = True
            sim = simulate_module(mod, toks, e1["i"], frame0, pend); kind = exit_kind(mod, events, last)
            if sim is None or sim[0] != i1 or sim[3] != kind:
                hybrid_rows.module_log[f"{mod} module differs" if sim else f"{mod} module refuses"] += 1
                return rows, roles, None, pend                       # the module decides for itself from here: no row may follow (prev None = cut)
            hybrid_rows.module_log[f"{mod} module agrees"] += 1
            i_exit, frame_exit, pend, _ = sim
            exit_view = frame_to_snapshot(frame_exit); exit_view["mret"] = sid
            after_view = afters[last] if not empty else befores[first]          # (an empty activation: the view does not move)
            post = register_actions(exit_view, after_view, None, allow_copy=False)
            post += np_register_actions(exit_view, after_view, toks[i1] if i1 < len(toks) else None, set())      # (the diff clears mret: the next snapshot has none)
            heads = heads_of(frame_exit)
            en = events[last + 1] if last + 1 < len(events) else noreads
            P_read = frozenset(predicates(toks, i1, exit_view, set(en["win_reads"]), set(en["reg_reads"]) | {"mret"}, pend, heads))
            P_all = frozenset(predicates(toks, i1, exit_view, set(range(i1, i1 + WINDOW)), ALL_REGS | {"mret"}, pend, heads))
            post_state = ("MRET", mod, sid, kind)
            if kind == "sub": P_read = frozenset(q for q in P_read if is_return_key(q))       # the module read the conjunction itself: the hand-over is keyed by its return address alone
            rows.append((("RETM", mod, kind), P_all, P_read, (post_state, 0, norm_actions(post))))
            prev = post_state; ei = last + 1
            continue                                    # (an empty span: ei = first, the event that follows the activation, is next)
        if ei in in_span:
            first, last, role, ja, cls, site, old, moved_to, dropped = info[in_span[ei]]
            roles.add(role); sid = site_id(site)
            # entry
            acts = [("ret", sid, ("const",))]
            if old is not None:
                if moved_to: acts += [(sl, None, ("move", sl2)) for sl, sl2 in zip(np_slots(moved_to), np_slots(role))]
                acts += [(sl, None, ("clear",)) for sl in np_slots(role)]
            e1 = events[first]; e2 = events[first + 1] if first + 1 < len(events) else noreads
            P_read = frozenset(predicates(toks, e1["i"], befores[first], set(e1["win_reads"]) | set(e2["win_reads"]), set(e1["reg_reads"]) | set(e2["reg_reads"]), pend))
            P_all = frozenset(predicates(toks, e1["i"], befores[first], set(range(e1["i"], e1["i"] + WINDOW)), ALL_REGS, pend))
            rows.append((prev, P_all, P_read, (("NP0", role), 0, norm_actions(acts))))
            # exit: the module's slots as it leaves them (its own run), then the post-NP effects of the rule parser
            el = events[last]; i1 = el["i"] + el["k"]
            regs = {"verb": befores[first].get("verb"), "cop_kind": befores[first].get("cop_kind"), "subj": befores[first].get("subj") if role != "subj" else None}
            mod = NPM.run_module(role, toks, e1["i"], regs, pend)
            heads = {}
            if mod[0] == "RET":
                slots = mod[2]; heads = {role: slots.get(f"{role}.noun2") or slots.get(f"{role}.noun")}
                hybrid_rows.module_log["module agrees" if (NPM.np_string(slots, role) == ja and mod[1] == i1) else "module differs"] += 1
                if NPM.np_string(slots, role) != ja or mod[1] != i1: hybrid_rows.module_log[("differs", " ".join(toks[e1["i"]:i1]), ja, NPM.np_string(slots, role))] += 1
            else: hybrid_rows.module_log["module refuses: " + mod[1].split(" at ")[0][:30]] += 1
            snap_ret = dict(befores[last]); snap_ret[role] = ja; snap_ret["ret"] = sid
            if role == "subj": snap_ret["subj_cls"] = cls
            if moved_to: snap_ret[moved_to] = old
            post = register_actions(befores[last], afters[last], None, allow_copy=False)
            post += np_register_actions(snap_ret, afters[last], toks[i1] if i1 < len(toks) else None, {role})
            if dropped: post += [(sl, None, ("clear",)) for sl in np_slots(role)]
            post = fold_use(post, role, heads.get(role), befores[first].get("verb"))
            post.append(("ret", None, ("clear",)))
            for j in range(e1["i"], i1):
                if "prep" in dict(readings_of(toks[j])): pend = toks[j]
            en = events[last + 1] if last + 1 < len(events) else noreads
            P_read = frozenset(predicates(toks, i1, snap_ret, set(en["win_reads"]), set(en["reg_reads"]) | {"ret"}, pend, heads))
            P_all = frozenset(predicates(toks, i1, snap_ret, set(range(i1, i1 + WINDOW)), ALL_REGS | {"ret"}, pend, heads))
            post_state = ("NPRET", sid, role)
            rows.append((("RET", role), P_all, P_read, (post_state, 0, norm_actions(post))))
            prev = post_state; ei = last + 1
            continue
        before, after = befores[ei], afters[ei]; tok = toks[e["i"]]
        actions = register_actions(before, after, tok) + np_register_actions(before, after, tok, set())
        nxt_e = events[ei + 1] if ei + 1 < len(events) else noreads
        win_idx = set(e["win_reads"]) | set(nxt_e["win_reads"]); reg_rd = set(e["reg_reads"]) | set(nxt_e["reg_reads"])
        P_read = frozenset(predicates(toks, e["i"], before, win_idx, reg_rd, pend))
        P_all = frozenset(predicates(toks, e["i"], before, set(range(e["i"], e["i"] + WINDOW)), ALL_REGS, pend))
        rows.append((prev, P_all, P_read, (e["cp"], e["k"], norm_actions(actions))))
        rows += agreement_fail_rows(prev, P_all, P_read, tok, e["reg_reads"])
        prev = e["cp"]
        for j in range(e["i"], e["i"] + e["k"]):
            if "prep" in dict(readings_of(toks[j])): pend = toks[j]
        ei += 1
    return rows, roles, prev, pend

def agreement_fail_rows(prev, P_all, P_read, tok, reg_reads):
    """the agreement check materialised: where the rule parser evaluated subject-verb agreement (it read subj_num, which
    it does only at a finite verb or at is/are), the refusing branch as a FAIL row — the same predicates plus agree.bad.
    The corpus holds no violation, so without this row the induced guard could never learn the refusal."""
    if "subj_num" not in reg_reads or "agree.bad" in P_all: return []
    return [(prev, frozenset(set(P_all) | {"agree.bad"}), frozenset(set(P_read) | {"agree.bad"}), (FAIL, 0, ()))]

def np_entry_event(events, np_spans, k):
    """if event k is inside a main-frame noun phrase, the index of that phrase's first event; else k"""
    e = events[k]
    if not any(n == "parse_np" for n, _ in e["chain"]): return k
    outer = [(i0, i1) for (i0, i1, ja, cls, main) in np_spans if main and i0 <= e["i"] < i1]
    if not outer: return None
    i0, i1 = max(outer, key=lambda s: s[1] - s[0])
    hits = [j for j, x in enumerate(events) if x["i"] <= i0 < x["i"] + x["k"]]
    return hits[0] if hits else None

def prefix_rows(toks, events, np_spans, cut, final_snap):
    """the transition rows of events[:cut] (a parse that stops being representable, or is refused, at event cut), with the
    noun phrases completed before the cut; -> (rows, roles, the control state after event cut-1, the pending preposition).
    Raises OutOfScope when a noun phrase of the prefix cannot be handed over."""
    if cut == 0: return [], set(), ("START",), None
    consumed = events[cut - 1]["i"] + events[cut - 1]["k"]
    spans = [sp for sp in np_spans if sp[1] <= consumed]
    snaps = [x["snap"] for x in events[:cut]] + [final_snap]
    return transition_rows(toks, events[:cut], snaps, spans, final_snap)

def hybrid_fail_rows(toks):
    """rows for a sentence the hybrid cannot parse to the end: the transitions the rule parser took before the point where
    it refused or the parse stopped being representable (an inner clause, a register the cell has no block for) — they are
    its transitions like any other — and a FAIL row at that point (outside parse_np: inside it the module refuses by itself)"""
    try:
        events, final, spans, p = trace(toks)
    except ParseFail as e:
        p = _last_parser[0]
        err, fs, win_rd, reg_rd, i = p.fail_info
        if isinstance(fs, OutOfScope) or not p.events_ok: return []
        events, spans = p.events, p.np_spans
        cut = len(events)                                   # events inside the unfinished noun phrase are the module's, not ours
        for k, e in enumerate(events):
            if any(n == "parse_np" for n, _ in e["chain"]) and not any(main and i0 <= e["i"] < i1 for (i0, i1, ja, cls, main) in spans): cut = k; break
        if _modules["on"] and failed_in_module(p.fail_stack):           # the parser died in a clause module's own code: the module refuses by itself from its entry
            for k, e in enumerate(events):
                if e.get("owner") and e["owner"][1] in p.fail_frames: cut = min(cut, k); break
        if p.forks:                                         # this is the default reading; from its first fork on, the path is the choice's, not the window's
            fpos = p.forks[0][0]
            cut = min(cut, next((k for k, e in enumerate(events) if e["i"] + e["k"] > fpos), cut))
            cut = np_entry_event(events, spans, cut) if cut < len(events) else cut
            if cut is None: return []
        final_snap = events[cut]["snap"] if cut < len(events) else fs
        if not isinstance(final_snap, dict): return []
        try: rows, roles, prev, pend = prefix_rows(toks, events, spans, cut, final_snap)
        except OutOfScope: return []
        if prev is None or cut < len(events) or "parse_np" in p.fail_stack or (_modules["on"] and failed_in_module(p.fail_stack)): return rows
        P_all = predicates(toks, i, fs, set(range(i, i + WINDOW)), ALL_REGS, p.pend_at_end)
        P_read = predicates(toks, i, fs, win_rd, reg_rd, p.pend_at_end)
        return rows + [(prev, frozenset(P_all), frozenset(P_read), (FAIL, 0, ()))]
    snaps = [x["snap"] for x in events] + [final]
    cut = None
    for ei, e in enumerate(events):
        if isinstance(e["snap"], OutOfScope) or e["snap"] is None: cut = ei - 1; break
    if cut is None and isinstance(final, OutOfScope): cut = len(events) - 1
    if cut is None:
        # parsed and representable, but a noun phrase could not be handed over (e.g. its string went to no register)
        try: np_span_info(events, snaps, spans, final); return []
        except OutOfScope: return []
    if cut < 0: return []
    cut = np_entry_event(events, spans, cut)              # a cut inside a noun phrase moves to the phrase's entry
    if cut is None or not isinstance(events[cut]["snap"], dict): return []
    if _modules["on"] and events[cut].get("owner"):       # the frame left the scope inside a clause module: the module refuses by itself from its entry
        fid = events[cut]["owner"][1]
        while cut > 0 and events[cut - 1].get("owner") and events[cut - 1]["owner"][1] == fid: cut -= 1
        try: return prefix_rows(toks, events, spans, cut, events[cut]["snap"])[0]
        except OutOfScope: return []
    try: rows, roles, prev, pend = prefix_rows(toks, events, spans, cut, events[cut]["snap"])
    except OutOfScope: return []
    if prev is None: return rows
    e = events[cut]
    nxt_e = events[cut + 1] if cut + 1 < len(events) else {"win_reads": set(), "reg_reads": set()}
    P_all = predicates(toks, e["i"], e["snap"], set(range(e["i"], e["i"] + WINDOW)), ALL_REGS, pend)
    P_read = predicates(toks, e["i"], e["snap"], set(e["win_reads"]) | set(nxt_e["win_reads"]), set(e["reg_reads"]) | set(nxt_e["reg_reads"]), pend)
    return rows + [(prev, frozenset(P_all), frozenset(P_read), (FAIL, 0, ()))]

def module_rules(roles):
    """np_module's rules and derived predicates for the roles in use (each with its of-phrase role), in the compiler's rule format"""
    import np_module as NPM
    rules = []; derived = {}
    for role in sorted(roles):
        for r in (role, NPM.of_role(role)):
            D = NPM.derived(r)
            for name, conjs in D.items(): derived[f"{name}@{r}"] = conjs
            def lit(l, r=r):
                neg = l.startswith("¬"); q = l[1:] if neg else l
                if q.startswith("d."): q = f"{q}@{r}"
                return (q, not neg)
            for state, lits, (nxt, k, acts) in NPM.rules_for(r):
                if isinstance(nxt, tuple): tgt = nxt                                   # into the of-phrase's module
                elif nxt == "RET": tgt = ("RET", r)
                elif nxt.startswith("FAIL:"): tgt = FAIL
                else: tgt = (nxt, r)
                rules.append(((state, r), tuple(lit(l) for l in lits), (tgt, k, norm_actions(acts))))
        rules.append(((("RET", NPM.of_role(role))), (), (("RET", role), 0, ())))           # the of-phrase's return is the phrase's return
    return rules, derived

def eval_rule_lit(q, truth, P, derived, frame=None):
    if q in derived:
        v = any(all(eval_rule_lit(l[1:] if l.startswith("¬") else l, not l.startswith("¬"), P, derived, frame) for l in conj) for conj in derived[q])
    elif q in P: v = True
    elif frame is not None and (q.startswith("flag.") or GM.family_of(q)): v = GM.base_pred(q, frame)     # the printer's predicate families, on the flat frame
    else: v = False
    return v == truth

def run_rules(rules, derived, toks, bound_per_token=12):
    """interpret the hybrid rule set (induced outer rules + module rules) on a sentence — the check of the rules before the matrix"""
    by_prev = collections.defaultdict(list)
    for prev, lits, out in rules: by_prev[prev].append((lits, out))
    state = ("START",); i = 0; frame = {}; pend = None
    for _ in range(bound_per_token * (len(toks) + 2)):
        if state == ("DONE",): return frame
        if state == FAIL: raise ParseFail(f"RULES: FAIL at token {i}", i)
        cp, k, actions = step_rules(by_prev, derived, state, toks, i, frame, pend)
        if cp == ("CONFLICT",) or (isinstance(cp, tuple) and cp and cp[0] == "CONFLICT"): raise ParseFail("RULES: conflict", i)
        apply_actions(frame, actions, toks, i)
        for j in range(i, i + k):
            if j < len(toks) and "prep" in dict(readings_of(toks[j])): pend = toks[j]
        i += k; state = cp
        if i > len(toks): raise ParseFail("RULES: ran past the end", i)
    raise ParseFail("RULES: did not finish", i)

def build_hybrid(sentences, verbose=False, modules=True):
    """rows from the corpus (outer transitions + NP hand-overs + the clause modules' hand-overs + FAIL rows), induced trees, the
    modules' rules -> the compiled machine: the encoder cell, the decoder cell, and what they share (the rules of both, the
    derived predicates, the slot vocabulary of the frame that passes between them)"""
    rows = []; roles = set(); scope = collections.Counter()
    _modules["on"] = modules
    if modules:
        crules, cderived = clause_module_rules()
        mrules0, mderived0 = module_rules(set(NP_ROLES))
        by_prev = collections.defaultdict(list)
        for prev, lits, out in crules + mrules0: by_prev[prev].append((lits, out))
        _modules["by_prev"] = by_prev; _modules["derived"] = {**cderived, **mderived0, "agree.bad": AGREE_DNF}
    for toks in sentences:
        toks = list(toks)
        try:
            Parser(list(toks), []).parse()
            r, rl = hybrid_rows(toks); rows += r; roles |= rl; scope["in scope"] += 1
        except (ParseFail, OutOfScope) as e:
            scope["refused" if isinstance(e, ParseFail) else "out of scope"] += 1
            try: rows += hybrid_fail_rows(toks)
            except Exception as e2: scope["fail-row error " + type(e2).__name__] += 1
        except Exception as e:
            scope["extraction error " + type(e).__name__] += 1
    trees = induce(rows)
    mrules, derived = module_rules(roles)
    derived["agree.bad"] = AGREE_DNF
    if modules:
        crules, cderived = clause_module_rules(); mrules = crules + mrules; derived.update(cderived)
    enc = compile_encoder(trees, extra_rules=mrules, derived=derived)
    dec = compile_decoder(enc["slot_vocab"])                                  # the printing rules, over the frame the encoder leaves
    if verbose: print("hybrid build:", dict(scope), "| data rows", len(rows), "| module rules", len(mrules), "| roles", sorted(roles))
    return {"encoder": enc, "decoder": dec, "rules": enc["rules"] + dec["rules"], "derived": {**enc["derived"], **dec["derived"]},
            "slot_vocab": enc["slot_vocab"], "src_vocab": enc["src_vocab"], "unresolved": enc["unresolved"] + dec["unresolved"], "scope": scope, "rows": rows}

GUARD_PREFIXES = ("w0.class=", "verb.set", "verb.vc=", "verb.phrasal", "verb.use", "flag.", "reg.", "hit=", "pp.count", "use.", "np.")   # the leaf guard: classes and registers the parser reads — not surfaces, the later window slots, or the conjugation class it never branches on
def induce(rows):
    """per previous control point, a decision tree over predicates whose leaves are outcomes; returns {prev: tree}.
    A leaf also carries the guard the rule parser itself applied: the class and register predicates (GUARD_PREFIXES) that
    every row reaching the leaf read as true (an `end` transition keeps w0.class=end even when no training row ever had
    another token there; an ing_comp transition keeps verb.vc=ing_comp), so that an induced rule never fires on a kind of
    token or state the parser was not seen to fire on; surfaces and the later window slots are left to the tree."""
    by_prev = collections.defaultdict(list); read_by_prev = collections.defaultdict(set)
    for prev, P_all, P_read, outcome in rows: by_prev[prev].append((P_all, P_read, outcome)); read_by_prev[prev] |= P_read
    def guard(items):
        common = set.intersection(*(set(R_) for _, R_, _ in items)) if items else set()
        return tuple(sorted(q for q in common if q.startswith(GUARD_PREFIXES)))
    def build(items, used, cands_all):
        outcomes = set(o for _, _, o in items)
        if len(outcomes) == 1: return ("leaf", next(iter(outcomes)), guard(items))
        cands = sorted(cands_all - used)
        best = None
        for q in cands:
            yes = [it for it in items if q in it[0]]; no = [it for it in items if q not in it[0]]
            if not yes or not no: continue
            score = (len(set(o for _, _, o in yes)) + len(set(o for _, _, o in no)), abs(len(yes) - len(no)))
            if best is None or score < best[0]: best = (score, q, yes, no)
        if best is None: return ("conflict", sorted(outcomes, key=str), len(items))
        _, q, yes, no = best
        return ("split", q, build(yes, used | {q}, cands_all), build(no, used | {q}, cands_all))
    trees = {}
    for prev, items in by_prev.items():
        t = build(items, frozenset(), read_by_prev[prev])
        if has_conflict(t):                                    # the read predicates do not separate: fall back to everything true here
            t = build(items, frozenset(), set().union(*(P for P, _, _ in items)))
        trees[prev] = t
    return trees

def has_conflict(tree):
    if tree[0] == "leaf": return False
    if tree[0] == "conflict": return True
    return has_conflict(tree[2]) or has_conflict(tree[3])

def tree_rules(tree, path=(), keys=None):
    """flatten a tree into (literals, outcome) rules; literals are (predicate, truth), the leaf's guard appended — and the refusals
    the tree left implicit made explicit, so that every induced control point decides every input (pinned by
    test_canon_control_points_total): where a guard literal fails, and where the traces disagreed (a conflict leaf), the rule is
    FAIL.  A return state (RET, RETM) is shared by every caller, each keyed by its return address, so a failing address guard is
    not negated (that would fire on another caller's address) but spelled out over the other addresses this tree knows; an address
    none of the callers serves is closed by the compiler (close_returns)"""
    if keys is None: keys = sorted(_return_keys(tree))
    if tree[0] == "leaf":
        seen = {q for q, t in path}; fixed = {q for q, t in path if t and is_return_key(q)}
        guard = sorted((q for q in tree[2] if q not in seen), key=lambda q: (not is_return_key(q), q))      # the return address first
        rules = [(path + tuple((q, True) for q in guard), tree[1])]
        for j, q in enumerate(guard):
            before = path + tuple((g, True) for g in guard[:j])
            if is_return_key(q):
                if not fixed: rules += [(before + ((k, True),), (FAIL, 0, ())) for k in keys if k != q]
            else: rules.append((before + ((q, False),), (FAIL, 0, ())))
        return rules
    if tree[0] == "conflict": return [(path, (FAIL, 0, ()))]
    _, q, yes, no = tree
    return tree_rules(yes, path + ((q, True),), keys) + tree_rules(no, path + ((q, False),), keys)

def _return_keys(tree):
    if tree[0] == "leaf": return {q for q in tree[2] if is_return_key(q)}
    if tree[0] == "conflict": return set()
    return ({tree[1]} if is_return_key(tree[1]) else set()) | _return_keys(tree[2]) | _return_keys(tree[3])

def frame_to_snapshot(frame):
    """the slot dict as the snapshot() view (so predicates read it the same way)"""
    s = {f: bool(frame.get(f)) for f in FLAGS}
    for r in ("modal", "want", "obj_particle", "wh", "time", "purpose", "causative", "verb", "front", "cop_kind", "cop_adj", "cop_adv", "cop_part", "pp1_part", "pp2_part", "pp3_part", "cmp_kind", "ret", "mret"):
        s[r] = frame.get(r)
    s["adv1"] = ((frame.get("adv1") or "") + (frame.get("adv2") or "")) or None; s["adv2"] = None       # the parser's one adverb string
    s["degree"] = ((frame.get("degree") or "") + (frame.get("degree2") or "")) or None
    for role in NP_ROLES:
        s[role] = np_string(frame, role)
    s["subj_cls"] = frame.get("subj.cls") if s["subj"] else None
    s["subj_num"] = (frame.get("subj.number") if frame.get("subj.number") != "none" else None) if s["subj"] else None
    for k in MARKS: s[k] = frame.get(k)
    for k in SENT_SLOTS: s[k] = bool(frame.get(k))
    return s

def np_string(frame, role):
    """det poss [of-phrase の] num adj1 adj2 noun [の noun2] [mono], or None when the role is empty"""
    if frame.get(f"{role}.noun") is None: return None
    s = (frame.get(f"{role}.det") or "") + (frame.get(f"{role}.poss") or "")
    if frame.get(f"{role}of.noun"): s += np_string(frame, f"{role}of") + "の"       # the top of the mountain -> 山の頂上
    s += "".join(frame.get(f"{role}.{sub}") or "" for sub in ("num", "adj1", "adj2", "noun"))
    if frame.get(f"{role}.noun2"): s += "の" + frame[f"{role}.noun2"]        # noun-noun compound
    if frame.get(f"{role}.mono"): s += frame[f"{role}.mono"]                  # tom 's -> トムのもの
    return s

def heads_of(frame):
    """the head noun of each filled NP role (the second noun of a compound), for the USE feedback predicate"""
    return {role: frame.get(f"{role}.noun2") or frame.get(f"{role}.noun") for role in NP_ROLES if frame.get(f"{role}.noun")}

def slot_predicates(frame):
    P = set()
    for slot, v in frame.items():
        if v not in (None, "", False): P.add(f"slot.{slot}"); P.add(f"slot.{slot}={v}")
    return P

def apply_actions(frame, actions, toks, i):
    tok = toks[i] if i < len(toks) else None; rs = collections.defaultdict(list)
    for c, v in readings_of(tok): rs[c].append(v)
    wsrc = window_sources(toks, i)
    for slot, v, src in actions:
        if src[0] == "copy" and src[1] in ("dur", "dur_in") and src[1] not in wsrc: raise ParseFail(f"TABLE: no {src[1]} phrase at {tok!r}", i)
        if src[0] == "copy" and src[1] not in ("ncls", "nnum", "dur", "dur_in") and not rs[{"num_time": "num", "adj_pair": "adj", "adj_pair_plain": "adj", "adj_mono": "adj", "adj_caus": "adj", "pron_poss": "pron", "adv_front": "adv", "adv_too": "adv"}.get(src[1], src[1])]:
            raise ParseFail(f"TABLE: nothing to copy into {slot} from {src[1]} at {tok!r}", i)       # the machine's crash: the rule fired on a token without that reading
    old = dict(frame)                                   # moves and clears read the state before this step, as the matrix does
    for slot, v, src in actions:
        if src[0] == "flush": continue                      # the frame goes to the context: nothing in it changes
        if src[0] == "use":                                 # the feedback lookup: slot is the role whose head noun carries the verb
            head = old.get(f"{slot}.noun2") or old.get(f"{slot}.noun"); verb = old.get("verb")
            hit = USE_TABLE.get((head, verb[0])) if head and verb else None
            if hit is None: raise ParseFail(f"TABLE: no use pair for {head!r} and {verb and verb[0]!r} at {tok!r}", i)
            new_verb, upart, stative = hit
            frame["verb"] = new_verb
            if upart == "-":
                frame.pop("obj_particle", None)
                for sl in np_slots(slot): frame.pop(sl, None)
            elif slot.startswith("pp"): frame[f"{slot}_part"] = (upart, upart)
            else: frame["obj_particle"] = upart
            if stative: frame["prog"] = True
            continue
        if src[0] == "clear": frame.pop(slot, None)
        elif src[0] == "const": frame[slot] = v
        elif src[0] in ("move", "move_time"):              # the gate clears the slot; the write lands only if the source slot is filled
            if old.get(src[1]) is not None: frame[slot] = old.get(src[1]) + "時" if src[0] == "move_time" else old.get(src[1])
            else: frame.pop(slot, None)
        elif src[0] == "copy":
            cls = src[1]
            if cls == "ncls": frame[slot] = rs["noun"][-1][1] if rs["noun"] else "t"
            elif cls == "noun": frame[slot] = rs["noun"][-1][0]
            elif cls == "pron": frame[slot] = rs["pron"][0][0]
            elif cls == "pron_poss": frame[slot] = rs["pron"][0][0] if rs["pron"][0][1] == "p" else rs["pron"][0][0] + "の"
            elif cls == "det": frame[slot] = rs["det"][0]
            elif cls == "num": frame[slot] = rs["num"][0]
            elif cls == "num_time": frame[slot] = rs["num"][0] + "時"
            elif cls == "adj": ja, kind = rs["adj"][-1]; frame[slot] = ja + ("な" if kind == "na" and not ja.endswith("な") else "")
            elif cls == "adj_pair": frame[slot] = rs["adj"][-1]
            elif cls == "adj_mono": ja, kind = rs["adj"][-1]; frame[slot] = ja + ("な" if kind == "na" and not ja.endswith("な") else "") + "もの"
            elif cls == "adj_caus": ja, kind = rs["adj"][-1]; frame[slot] = ("make+" + tok, R.adj_causative(ja, kind), "i")
            elif cls == "adv": frame[slot] = rs["adv"][-1]
            elif cls == "adv_front": frame[slot] = rs["adv"][-1] + "、"
            elif cls == "adv_too": frame[slot] = "あまりに" + rs["adv"][-1]
            elif cls == "adj_pair_plain": ja, kind = rs["adj"][-1]; frame[slot] = (ja[2:] if ja.startswith("より") else ja, kind)
            elif cls in ("dur", "dur_in"): frame[slot] = wsrc[cls]
            elif cls == "time": frame[slot] = rs["time"][-1]
            elif cls == "loc": frame[slot] = rs["loc"][-1]
            elif cls == "wh": frame[slot] = rs["wh"][-1]
            elif cls == "modal": frame[slot] = rs["modal"][-1][0]
            elif cls == "conj": frame[slot] = rs["conj"][0]
            elif cls == "nnum": frame[slot] = R.number_of(tok) or "none"
            elif cls == "verb":
                lem, form, ja, conj = rs["verb"][0]; frame[slot] = (lem, ja, conj)

def frame_to_clause(frame):
    c = Clause()
    for f in FLAGS: setattr(c, f, bool(frame.get(f)))
    for r in ("modal", "want", "obj_particle", "wh", "purpose", "causative", "verb"): setattr(c, r, frame.get(r))
    c.degree = (frame.get("degree") or "") + (frame.get("degree2") or ""); c.time = frame.get("time") or ""
    c.adv = (frame.get("adv1") or "") + (frame.get("adv2") or ""); c.front = frame.get("front") or ""
    c.subj = np_string(frame, "subj"); c.subj_cls = frame.get("subj.cls") or "h"
    c.subj_num = frame.get("subj.number") if frame.get("subj.number") != "none" else None
    c.obj = np_string(frame, "obj"); c.iobj = np_string(frame, "iobj")
    c.pp = [(frame[f"pp{k}_part"], np_string(frame, f"pp{k}")) for k in (1, 2, 3) if frame.get(f"pp{k}.noun") is not None]
    c.compare = (frame["cmp_kind"], np_string(frame, "cmp")) if frame.get("cmp_kind") and frame.get("cmp.noun") is not None else None
    kind = frame.get("cop_kind")
    if kind == "adj": c.copula = ("adj", frame.get("cop_adj"))
    elif kind == "np": c.copula = ("np", np_string(frame, "cop"), frame.get("cop.cls") or "t")
    elif kind == "pp": c.copula = ("pp", frame.get("cop_part"), np_string(frame, "cop"))
    elif kind == "adv": c.copula = ("adv", frame.get("cop_adv"))
    elif kind == "wh": c.copula = ("wh",)
    return c

def frames_to_clause(frames):
    """the frames of a context -> the rule parser's Clause.  A frame marked embedded is the that-clause of the frame before it;
    of the others one is the main clause and at most one a subordinate clause, on the side its mark says, which the main clause
    holds as the rule parser does — the clause in subords and its string in pre_clause (rules_ja.subord_form: what the
    conjunction changes is in the frame already, and changing it again changes nothing); the sentence block of the last frame
    — the question mark — belongs to the main clause"""
    clauses = [frame_to_clause(F) for F in frames]; tops = []
    for k, (F, c) in enumerate(zip(frames, clauses)):
        if F.get("clause") == "embedded":
            if k == 0: raise ParseFail("MATRIX: a that-clause with no clause before it")
            clauses[k - 1].embedded = c; c.inner_of = "embedded"
        else: tops.append((F, c))
    mains = [c for F, c in tops if not F.get("clause")]; subs = [(F, c) for F, c in tops if F.get("clause")]
    if len(mains) != 1 or len(subs) > 1: raise ParseFail(f"MATRIX: {len(mains)} main clauses and {len(subs)} subordinate clauses in the context")
    main = mains[0]
    for F, c in subs:
        order, conj = F["clause"], F.get("conj")
        if conj not in R.SUBORD or tops.index((F, c)) != (0 if order == "before" else 1): raise ParseFail(f"MATRIX: a subordinate frame out of place or without its conjunction ({conj!r})")
        c.inner_of, c.conj = order, conj; main.subords.append((order, conj, c)); main.pre_clause = R.subord_form(c, conj)
    if frames[-1].get("sent.question"): main.question = True
    return main

# ============================================================================================== the matrices
class Index:
    """names -> contiguous unit indices"""
    def __init__(self): self.names = []; self.pos = {}
    def add(self, name):
        if name not in self.pos: self.pos[name] = len(self.names); self.names.append(name)
        return self.pos[name]
    def __len__(self): return len(self.names)
    def __contains__(self, name): return name in self.pos
    def __getitem__(self, name): return self.pos[name]

COPY_SOURCES = ("noun", "ncls", "pron", "pron_poss", "det", "num", "num_time", "adj", "adj_pair", "adj_pair_plain", "adj_mono", "adj_caus", "adv", "adv_front", "adv_too", "time", "wh", "modal", "verb", "nnum", "loc", "dur", "dur_in", "conj")

def source_values(tok):
    """the copy-source readings of a token, as (source, value) pairs — what the lexical layer emits for window slot 0"""
    out = []
    for c, v in readings_of(tok):
        if c == "noun": out.append(("noun", v[0])); out.append(("ncls", v[1]))
        elif c == "pron": out.append(("pron", v[0])); out.append(("pron_poss", v[0] if v[1] == "p" else v[0] + "の"))
        elif c == "det": out.append(("det", v))
        elif c == "num": out.append(("num", v)); out.append(("num_time", v + "時"))
        elif c == "adj":
            out.append(("adj", v[0] + ("な" if v[1] == "na" and not v[0].endswith("な") else ""))); out.append(("adj_pair", v))
            out.append(("adj_mono", v[0] + ("な" if v[1] == "na" and not v[0].endswith("な") else "") + "もの")); out.append(("adj_pair_plain", (v[0][2:] if v[0].startswith("より") else v[0], v[1])))
            out.append(("adj_caus", ("make+" + tok, R.adj_causative(v[0], v[1]), "i")))
        elif c == "adv": out.append(("adv", v)); out.append(("adv_front", v + "、")); out.append(("adv_too", "あまりに" + v))
        elif c == "loc": out.append(("loc", v))
        elif c == "time": out.append(("time", v))
        elif c == "wh": out.append(("wh", v))
        elif c == "modal": out.append(("modal", v[0]))
        elif c == "conj": out.append(("conj", v))
        elif c == "verb": out.append(("verb", (v[0], v[2], v[3])))
    if any(c in ("noun", "pron") for c, v in readings_of(tok)): out.append(("nnum", R.number_of(tok) or "none"))   # the head's grammatical number
    # the rule parser takes the LAST reading of a class (dict() semantics) for noun/adj/adv/time/wh, the FIRST for pron/det/num/verb
    last = {}; first = {}
    for s, v in out:
        last[s] = v; first.setdefault(s, v)
    pick = {s: (last[s] if s in ("noun", "ncls", "adj", "adj_pair", "adj_pair_plain", "adj_mono", "adj_caus", "adv", "adv_front", "adv_too", "loc", "time", "wh") else first[s]) for s in last}
    return pick

PRINT_SRC = ("emit",)          # the action of a printing rule that emits: ("out", token, PRINT_SRC)
HEAD_SRC = ("attend",)         # the action of a printing rule that moves the attention: ("att", "down" | "up", HEAD_SRC)

def print_state(pos):
    """a printing position as a control state of the decoder; a FAIL:<reason> exit is the cell's FAIL"""
    return FAIL if isinstance(pos, str) and pos.startswith("FAIL:") else ("P", pos)

SUBORD_LEAD = sorted(R.SUBORD)                                                             # the conjunctions that open a clause before the main clause (parser_rules.pre)
SUBORD_TRAIL = sorted(w for w in R.SUBORD if w not in R.PREPS and w not in R.COORD)        # … and after it: the PP machine's second exit (pp_module: a preposition or while reads otherwise)

def adjust_rules(state, guard, target, acts):
    """the rules that close a subordinate clause's frame at `state`, under the literals `guard`: what its conjunction changes in
    the clause (rules_ja.subord_adjust, mirrored: the past of if, once and after, the negative of unless, the potential for may
    and no will under so that) is written in the step that does `acts`; one rule per case, exclusive by the one-hot of conj and
    of modal (pinned by test_subordinate_frame_holds_its_conjunction)"""
    G = tuple(guard); acts = list(acts); lit = lambda q, t=True: (q, t)
    cases = [((lit(f"slot.conj={c}"),), [("past", True, ("const",))]) for c in ("if", "once", "after")]
    cases += [((lit("slot.conj=unless"),), [("neg", True, ("const",))]),
              ((lit("slot.conj=so that"), lit("slot.modal=may")), [("modal", "can", ("const",))]),
              ((lit("slot.conj=so that"), lit("slot.modal=will")), [("modal", None, ("clear",))]),
              ((lit("slot.conj=so that"), lit("slot.modal=may", False), lit("slot.modal=will", False)), []),
              (tuple(lit(f"slot.conj={c}", False) for c in ("if", "once", "after", "unless", "so that")), [])]
    return [(state, G + lits, (target, 0, norm_actions(extra + acts))) for lits, extra in cases]

def clause_rules(slots, sites=()):
    """the encoder's fixed rules at the edges of a clause, over `slots` (every slot of its frame) and the `sites` — (kind, site) —
    where the traces open a clause: -> (rules, the derived predicates they read).  The row induced from the traces only says
    that a boundary of that kind is taken there, into the site's OPEN state — one state per site, as a noun phrase's return has
    one; these rules mark and flush the frame of the clause left behind, clear every slot by its own gate for the next clause
    to be read into, mark the new frame, and arrive at the site's INNER state, where that clause is read.
      that    a that-clause opens: the induced row has written the link and raised the flush unit; OPEN clears, marks the new
              frame embedded and keeps the sentence block — a question flag of the clause left behind stays as sent.question,
              since the question mark is read after its frame has gone.  Inside a subordinate clause it is refused (its frame
              would leave without what its conjunction changes, and lie between the two frames the decoder walks)
      before  a subordinate clause opens before its main clause, on its conjunction (d.clause.lead: a subordinator, not the
              question word when, the working frame still empty — parser_rules.pre): OPEN reads the conjunction into conj and
              marks the frame; nothing is flushed
      main    that clause has closed, where the rule parser's tail leaves a subordinate clause and its main clause may begin
              (d.clause.close: at a comma, or at the `or not` of whether — the tail's return there consumes nothing, so no
              trace shows the PP machine deciding it): OPEN writes what its conjunction changes and flushes the frame; TURN
              clears everything and marks the main clause's frame with sublink = before
      after   a subordinate clause opens after its main clause, on its conjunction (d.clause.trail): OPEN marks the main
              clause's frame with sublink = after and flushes it; TURN clears, reads the conjunction and marks the new frame,
              keeping the sentence block
    CLOSE, at the end of every sentence, flushes the last frame — a subordinate clause read after its main clause with what
    its conjunction changes — and is DONE; a subordinate clause with no main clause after it is refused"""
    slots = sorted(set(slots) | set(MARKS) | {"sub"} | set(SENT_SLOTS))
    def clears(keep): return [(sl, None, ("clear",)) for sl in slots if sl not in keep]
    def subord(w, lead): return [f"w0.class=conj", f"w0.surf={w}"] + ([f"¬w1.class={c}" for c in ("aux_be", "aux_do", "aux_have", "modal")] if lead and w == "when" else [])
    derived = {"d.clause.lead": [subord(w, True) + ["¬w0.class=time", "¬w0.class=int", "¬w0.class=yesno", "¬slot.front", "¬slot.time", "¬flag.please", "¬slot.clause", "¬slot.sublink"] for w in SUBORD_LEAD],
               "d.clause.trail": [subord(w, False) + ["¬slot.clause", "¬slot.sublink"] for w in SUBORD_TRAIL],
               "d.clause.close": [["w0.surf=,"], ["slot.conj=whether", "w0.surf=or", "w1.class=not"]]}
    T = lambda q: (q, True); F = lambda q: (q, False)
    read_conj = [("conj", None, ("copy", "conj")), ("sub", True, ("const",))]
    rules = adjust_rules(("CLOSE",), (T("slot.clause=after"),), ("DONE",), [FLUSH])
    rules += [(("CLOSE",), (T("slot.clause=before"),), (FAIL, 0, ())),
              (("CLOSE",), (F("slot.clause=after"), F("slot.clause=before")), (("DONE",), 0, (FLUSH,)))]
    for kind, sid in sorted(sites):
        O, U, I = ("OPEN", kind, sid), ("TURN", sid), ("INNER", sid)
        if kind == "that":
            opened = clears(("clause",) + SENT_SLOTS) + [("clause", "embedded", ("const",))]
            main = (T("slot.link=that"), F("slot.clause=before"), F("slot.clause=after"))
            rules += [(O, main + (T("flag.question"),), (I, 0, norm_actions(opened + [("sent.question", True, ("const",))]))),
                      (O, main + (F("flag.question"),), (I, 0, norm_actions(opened))),
                      (O, (T("slot.link=that"), T("slot.clause=before")), (FAIL, 0, ())), (O, (T("slot.link=that"), T("slot.clause=after")), (FAIL, 0, ())),
                      (O, (F("slot.link=that"),), (FAIL, 0, ()))]
        elif kind == "before":
            rules += [(O, (T("d.clause.lead"),), (I, 1, norm_actions([("clause", "before", ("const",))] + read_conj))), (O, (F("d.clause.lead"),), (FAIL, 0, ()))]
        elif kind == "main":
            rules += adjust_rules(O, (T("slot.clause=before"), T("d.clause.close")), U, [FLUSH])
            rules += [(O, (F("slot.clause=before"),), (FAIL, 0, ())), (O, (T("slot.clause=before"), F("d.clause.close")), (FAIL, 0, ()))]
            rules += [(U, (), (I, 0, norm_actions(clears(("sublink",)) + [("sublink", "before", ("const",))])))]
        elif kind == "after":
            rules += [(O, (T("d.clause.trail"),), (U, 0, norm_actions([("sublink", "after", ("const",)), FLUSH]))), (O, (F("d.clause.trail"),), (FAIL, 0, ()))]
            opened = clears(("clause", "conj", "sub") + SENT_SLOTS) + [("clause", "after", ("const",))] + read_conj
            rules += [(U, (T("flag.question"),), (I, 1, norm_actions(opened + [("sent.question", True, ("const",))]))),
                      (U, (F("flag.question"),), (I, 1, norm_actions(opened)))]
        else: raise ValueError(f"no clause rules for a boundary of kind {kind}")
    return rules, derived

def decoder_rules():
    """gen_module's printing rules in the compiler's rule format — a token to emit as the action ("out", token, PRINT_SRC), a move
    of the attention as ("att", move, HEAD_SRC): -> (rules, derived predicates)"""
    pr = GM.PrintRules(); rules = []
    for pos, lits, (nxt, tok) in pr.rules():
        acts = () if tok is None else (("att", tok[1], HEAD_SRC),) if tok[0] == "h" else (("out", tok, PRINT_SRC),)
        rules.append((("P", pos), tuple((l[1:], False) if l.startswith("¬") else (l, True) for l in lits), (print_state(nxt), 0, acts)))
    return rules, pr.derived()

def form_ok(verb, name, alt):
    """the verb has this form (gen_module.vform does not refuse it); the alternative forms belong to 'understand' alone"""
    if alt and verb[0] != "understand": return False
    try: GM.vform(verb, name, alt); return True
    except GM.Refuse: return False

def output_lexicon(unit):
    """the output lexical layer: an output unit of the state -> the string the print head appends"""
    kind = unit[1]
    if kind == "c": return unit[2]
    if kind == "s":
        _, _, slot, idx, v = unit; v = v if idx is None else v[idx]
        return "" if v == GM.NOPART else str(v)
    if kind == "vf": _, _, name, alt, v = unit; return GM.vform(v, name, alt)
    if kind == "af": _, _, strip, v = unit; return GM.adj_stem(v, strip)
    raise ValueError(unit)

def implied(sets, by_superset):
    """drop a literal another literal of the same conjunction implies: a positive one whose units are a superset of another positive
    one's (B => A), a negative one whose units are a subset of another negative one's (not B => not A)"""
    keep = []
    for i, a in enumerate(sets):
        if any(j != i and (b < a if by_superset else a < b) or (b == a and j < i) for j, b in enumerate(sets)): continue
        keep.append(a)
    return keep

def is_return_key(q): return q.startswith(("reg.ret=", "reg.mret="))

def close_returns(rules):
    """at a return state — ("RET", role) of the noun-phrase module, ("RETM", module, exit) of a clause module — every rule is keyed by
    the return address it serves, and the hand-written modules and the induced hand-overs each add theirs, so no single one can
    write the otherwise: the compiler closes the state with a refusal for an address none of them serves (a state with an
    unconditional return is total already)"""
    by_prev = collections.defaultdict(list)
    for prev, lits, out in rules:
        if isinstance(prev, tuple) and prev[0] in ("RET", "RETM"): by_prev[prev].append(lits)
    extra = []
    for prev, litss in sorted(by_prev.items(), key=str):
        keys = set()
        for lits in litss:
            ks = [q for q, t in lits if t and is_return_key(q)]
            if not ks: keys = None; break
            keys |= set(ks)
        if keys: extra.append((prev, tuple((q, False) for q in sorted(keys)), (FAIL, 0, ())))
    return extra

MOVE_FN = {"move": lambda v: v, "move_time": lambda v: v + "時"}

def vocabulary(rules):
    """the values every slot can hold under a rule set: the constants the rules write, every value the lexicon can produce for
    the copy sources they use, what the moves carry over (to a depth of three), and what the USE lookup rewrites; -> (slot
    vocabulary, source vocabulary, the copy pairs, the move pairs, the rules that apply the USE feedback per role)"""
    slot_vocab = collections.defaultdict(set)
    for _, _, (cp, k, actions) in rules:
        for slot, v, src in actions:
            if src[0] == "const" and v is not None: slot_vocab[slot].add(v)
    src_vocab = collections.defaultdict(set)
    for tok in list(R.LEX) + list(R.PRONOUNS) + list(R.DETS) + list(R.NUMBERS) + list(R.WH) + list(R.MODALS) + list(R.VERB_FORMS) + list(R.SUBORD) + list(R.COORD):
        for s, v in source_values(tok).items(): src_vocab[s].add(v)
    for s, vals in window_source_vocab().items(): src_vocab[s] |= set(vals)
    for s in src_vocab: src_vocab[s] = sorted(src_vocab[s], key=str)
    copy_pairs = set(); move_pairs = set()
    for _, _, (cp, k, actions) in rules:
        for slot, v, src in actions:
            if src[0] == "copy": copy_pairs.add((slot, src[1])); slot_vocab[slot] |= set(src_vocab[src[1]])
            if src[0] in MOVE_FN: move_pairs.add((slot, src[1], src[0]))
    for _ in range(3):
        for slot, src_slot, fn in list(move_pairs): slot_vocab[slot] |= {MOVE_FN[fn](v) for v in slot_vocab.get(src_slot, ())}
    use_rules = collections.defaultdict(list)                                   # role -> the rules that apply the feedback on it
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        for slot, v, src in actions:
            if src[0] == "use": use_rules[slot].append(ri)
    for role in use_rules:
        nouns = set(slot_vocab.get(f"{role}.noun", [])) | set(slot_vocab.get(f"{role}.noun2", []))
        lemmas = {vv[0] for vv in slot_vocab.get("verb", [])}
        for (ja, lemma), (new_verb, upart, stative) in USE_TABLE.items():
            if ja not in nouns or lemma not in lemmas: continue
            slot_vocab["verb"].add(new_verb)
            if upart != "-": slot_vocab[f"{role}_part" if role.startswith("pp") else "obj_particle"].add((upart, upart) if role.startswith("pp") else upart)
            if stative: slot_vocab["prog"].add(True)
    return slot_vocab, src_vocab, copy_pairs, move_pairs, use_rules

def universal_vocabulary():
    """every slot's values over the hand-written rules of every module — the noun-phrase module for every role, the clause modules,
    the clause rules — and the lexicon's copy sources: the universe a diagram (canon.py) ranges over when it has no particular cell"""
    rules = module_rules(set(NP_ROLES))[0] + clause_module_rules()[0]
    rules += clause_rules(sorted(vocabulary(rules)[0]))[0]
    return vocabulary(rules)[0]

# ---------------------------------------------------------------------------------------------- the layers both cells share
def slot_values(q, slot_vocab):
    """a literal over one slot of the frame -> (the slot, its values that make the literal true): slot.<s> (any value),
    slot.<s>=<v>, flag.<f>, and the printer's families (a test on one slot's value); None when q is not such a literal"""
    if q.startswith("slot.") and "=" in q:
        slot, v = q[5:].split("=", 1); return slot, [u for u in slot_vocab.get(slot, []) if str(u) == v]
    if q.startswith(("slot.", "flag.")): return q[5:], list(slot_vocab.get(q[5:], []))
    fam = GM.family_of(q)
    if fam: return fam[0], [u for u in slot_vocab.get(fam[0], []) if fam[1](u, fam[2])]
    return None

def conjunction_units(derived, edges, L1, W1, b1):
    """layer 1 of either cell: one AND unit per conjunction of a derived predicate's DNF, deduplicated by content — every positive
    literal on and no negative one.  edges(name, q) gives the units whose OR is the base literal q, or None when no value of the
    cell makes it true: a positive literal then kills the conjunction, a negative one holds vacuously.  A literal another literal
    implies adds nothing, so the weights stay ±1.  -> (the conjunction units of each predicate, the conjunctions that are dead)"""
    conj_units = {}; terms = collections.defaultdict(list); dead = 0
    for name, conjs in derived.items():
        for conj in conjs:
            key = tuple(sorted(conj))
            if key not in conj_units:
                pos, neg_, ok = [], [], True
                for l in conj:
                    neg = l.startswith("¬"); q = l[1:] if neg else l
                    es = edges(name, q)
                    if es is None:
                        if not neg: ok = False; break            # a positive literal the cell cannot see: the conjunction is never true
                        continue                                  # a negative literal on an unseen predicate is vacuously true
                    (neg_ if neg else pos).append(frozenset(es))
                if not ok: conj_units[key] = None; dead += 1
                else:
                    u = L1.add(("conj", key))
                    pos, neg_ = implied(pos, True), implied(neg_, False)       # a literal another literal implies adds nothing: weights stay ±1
                    acc = collections.defaultdict(float)
                    for refs in pos:
                        for ref in refs: acc[ref] += 1.0
                    for refs in neg_:
                        for ref in refs: acc[ref] -= 1.0
                    for ref, w in acc.items():
                        if w: W1.append((u, ref, w))
                    b1[u] = -(len(pos) - 1)                       # AND: all positives on and no negative; with no positive, 1 unless a negative is on
                    conj_units[key] = u
            if conj_units[key] is not None: terms[name].append(conj_units[key])
    return terms, dead

def nor_units(derived, terms, L2, W2, b2):
    """layer 2 of either cell: the NOT of each derived predicate — one minus the sum of its conjunction units"""
    for name in derived:
        u = L2.add(("nor", name)); b2[u] = 1.0
        for t in terms.get(name, []): W2.append((u, ("L1", t), -1.0))

def rule_units(rules, derived, edges, ctrl, silencer, L2, L3, W3, b3, unresolved, quiet):
    """layer 3 of either cell: a rule is the AND of its control point and its literals — a derived predicate through its NOT unit
    (positive wants NOT = 0, negative wants NOT = 1), a base literal through the units edges(q) gives.  A positive literal no
    value of the cell makes true silences the rule with a large negative weight from the constant input, and is counted as
    unresolved unless the rule's control point is of a `quiet` kind (a hand-written rule on a value nothing writes is dead by
    design); a negative one holds vacuously"""
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        u = L3.add(("rule", ri)); npos = 1
        W3.append((u, ctrl(prev), 1.0))
        pos, neg, unseen = [], [], False
        for q, truth in lits:
            if q in derived:
                nu = ("L2", L2[("nor", q)])
                if truth: W3.append((u, nu, -1.0))
                else: W3.append((u, nu, 1.0)); npos += 1
                continue
            es = edges(q)
            if es is None:
                if truth:
                    unseen = True
                    if prev[0] not in quiet: unresolved[q] += 1
                continue
            (pos if truth else neg).append(frozenset(es))
        pos, neg = implied(pos, True), implied(neg, False)
        acc = collections.defaultdict(float)
        for refs in pos:
            for ref in refs: acc[ref] += 1.0
        for refs in neg:
            for ref in refs: acc[ref] -= 1.0
        for ref, w in acc.items():
            if w: W3.append((u, ref, w))                        # a unit in a positive and a negative literal cancels to nothing: AND(P, not N) over one-hot blocks
        if unseen: W3.append((u, silencer, -5.0))
        npos += len(pos)
        b3[u] = -(npos - 1)

# ---------------------------------------------------------------------------------------------- the encoder cell
def compile_encoder(trees, extra_rules=(), derived=None):
    """trees (per previous control point) + hand-written rules -> the encoder cell: five sparse ReLU layers over a one-hot state —
    the control point, the slots of the working frame, the pending preposition, the shift and flush units — and the lexical
    input, then the new state.  `derived` maps a predicate name to a DNF over base literals (np_module.derived)."""
    derived = dict(derived or {})
    rules = []   # (prev, literals, (cp, k, actions))
    for prev, t in trees.items():
        for lits, out in tree_rules(t):
            if out[0] == "CONFLICT": continue
            rules.append((prev, lits, out))
    rules += list(extra_rules)
    rules += close_returns(rules)
    # ---- vocabularies; the clause rules range over every slot the frame has, and what they write joins the vocabulary
    slot_vocab = vocabulary(rules)[0]
    crules, cderived = clause_rules(sorted(slot_vocab), {out[0][1:] for _, _, out in rules if out[0][0] == "OPEN"})
    rules += crules; derived.update(cderived)
    slot_vocab, src_vocab, copy_pairs, move_pairs, use_rules = vocabulary(rules)
    # ---- state index: control, slots, pend_prep, shift
    S = Index()
    ctrl_states = {("START",), ("DONE",), FAIL} | set(trees) | set(out[0] for _, _, out in rules) | set(prev for prev, _, _ in rules)
    skip_states = []
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        for j in range(1, k): skip_states.append(("SKIP", ri, j))
    for c in sorted(ctrl_states, key=str): S.add(("ctrl", c))
    for c in skip_states: S.add(("ctrl", c))
    for slot in sorted(slot_vocab):
        for v in sorted(slot_vocab[slot], key=str): S.add(("slot", slot, v))
    preps = sorted(w for w in R.PREPS)
    for w in preps: S.add(("pend", w))
    S.add(("shift",)); S.add(("flush",))
    def slot_units(slot): return [S[("slot", slot, v)] for v in slot_vocab.get(slot, [])]
    # ---- input index: window predicates, copy sources, prep-surface of slot 0
    X = Index()
    def is_window(q): return q.startswith("w") and q[1].isdigit() or (q.startswith("hit=") and not verb_hit(q))     # a verb hit is a unit of the cell
    for prev, lits, out in rules:
        for q, truth in lits:
            if is_window(q): X.add(("pred", q))
    for name, conjs in derived.items():
        for conj in conjs:
            for l in conj:
                q = l[1:] if l.startswith("¬") else l
                if is_window(q): X.add(("pred", q))
    for s in COPY_SOURCES:
        for v in src_vocab.get(s, []): X.add(("src", s, v))
    for w in preps: X.add(("prep0", w))
    X.add(("one",))
    # base literals resolved on the state and input directly (one-hot blocks: a sum over a block is 0/1)
    def base_edges(q):
        """-> the units whose OR is a base literal, or None if no value of this cell makes it true"""
        if is_window(q): return [("X", X[("pred", q)])] if ("pred", q) in X else None
        if q == "reg.verb": return [("S", u) for u in slot_units("verb")] or None
        if q == "reg.cop_kind": return [("S", u) for u in slot_units("cop_kind")] or None
        if q == "reg.subj": return [("S", u) for u in slot_units("subj.noun")] or None
        if q.startswith("pend_prep="): return [("S", S[("pend", q[10:])])] if ("pend", q[10:]) in S else None
        if q.startswith("reg.ret="): return [("S", S[("slot", "ret", q[8:])])] if ("slot", "ret", q[8:]) in S else None
        if q.startswith("reg.mret="): return [("S", S[("slot", "mret", q[9:])])] if ("slot", "mret", q[9:]) in S else None
        if q.startswith("reg.") and "=" in q:
            e, v = q[4:].split("=", 1); e = {"adv": "adv1"}.get(e, e)
            if v == "set": return [("S", u) for u in slot_units(e)] or None
            return [("S", S[("slot", e, u)]) for u in slot_vocab.get(e, []) if str(u) == v] or None
        sv = slot_values(q, slot_vocab)                             # slot.<s>[=<v>], flag.<f>, a family: an OR over the slot's values that pass
        return ([("S", S[("slot", sv[0], u)]) for u in sv[1]] or None) if sv else None
    def conj_edges(name, q):
        if verb_hit(q): raise ValueError(f"{name}: a hit the verb decides is a layer-2 unit and cannot enter a layer-1 conjunction")
        return base_edges(q)
    # ---- layer 1: register predicates derived from the state (OR over slot units); derived conjunctions (AND over base literals);
    #      USE pairs (AND of verb unit and head-noun unit)
    L1 = Index(); W1 = []; b1 = {}
    def or_unit(name, src_idx):
        u = L1.add(name)
        for i in src_idx: W1.append((u, i, 1.0))
        return u
    for role in NP_ROLES:
        if slot_vocab.get(f"{role}.noun"): or_unit(("pred", f"reg.{role}"), slot_units(f"{role}.noun"))
    if ("slot", "subj.noun", "それ") in S: or_unit(("pred", "subj.sore"), [S[("slot", "subj.noun", "それ")]])
    for cls in ("h", "t", "a", "p"):
        if ("slot", "subj.cls", cls) in S: or_unit(("pred", f"reg.subj_cls={cls}"), [S[("slot", "subj.cls", cls)]])
    for f in FLAGS:
        if slot_vocab.get(f): or_unit(("pred", f"flag.{f}"), slot_units(f))
    for e in ("modal", "want", "degree", "obj_particle", "wh", "purpose", "causative", "cop_kind", "ret"):
        for v in slot_vocab.get(e, []):
            or_unit(("pred", f"reg.{e}={v}"), [S[("slot", e, v)]])
    if slot_vocab.get("cop_kind"): or_unit(("pred", "reg.cop_kind"), slot_units("cop_kind"))
    for e in ("time", "adv1", "front"):
        if slot_vocab.get(e): or_unit(("pred", f"reg.{e}=set"), slot_units(e))
    if slot_vocab.get("adv1"): or_unit(("pred", "reg.adv=set"), slot_units("adv1"))
    vunits = slot_units("verb")
    if vunits:
        or_unit(("pred", "verb.set"), vunits); or_unit(("pred", "reg.verb"), vunits)
        for cls, s in R.VC.items():
            idx = [S[("slot", "verb", v)] for v in slot_vocab["verb"] if R.lemma_of(v) in s]
            if idx: or_unit(("pred", f"verb.vc={cls}"), idx)
        for conj in set(v[2] for v in slot_vocab["verb"]):
            or_unit(("pred", f"verb.conj={conj}"), [S[("slot", "verb", v)] for v in slot_vocab["verb"] if v[2] == conj])
        or_unit(("pred", "verb.phrasal"), [S[("slot", "verb", v)] for v in slot_vocab["verb"] if " " in v[0]] or [])
        or_unit(("pred", "verb.use"), [S[("slot", "verb", v)] for v in slot_vocab["verb"] if "+" in v[0]] or [])
    for role in NP_ROLES:                                             # np.<role>.time: the head noun ends in 時 and nothing follows it
        tu = [S[("slot", f"{role}.noun", v)] for v in slot_vocab.get(f"{role}.noun", []) if isinstance(v, str) and v.endswith("時")]
        if tu:
            u = or_unit(("pred", f"np.{role}.time"), tu)
            for i in slot_units(f"{role}.noun2") + slot_units(f"{role}.mono"): W1.append((u, i, -1.0))
    f = [L1.add(("filled", f"pp{k}")) for k in (1, 2, 3)]
    for k in (1, 2, 3):
        for i in slot_units(f"pp{k}.noun"): W1.append((f[k-1], i, 1.0))
    for w in preps: or_unit(("pred", f"pend_prep={w}"), [S[("pend", w)]])
    # derived conjunctions: one AND unit per conjunction, deduplicated by content
    derived_terms, dead_conj = conjunction_units(derived, conj_edges, L1, W1, b1)
    # USE pairs: verb unit AND head-noun unit (head = noun2 if present else noun)
    use_pairs = collections.defaultdict(list)
    pairs = {(ja, vv) for en, d in R.USE.items() for lemma in d for ja in {v[0] for c, v in R.classify(en) if c == "noun"}
             for vv in slot_vocab.get("verb", []) if vv[0] == lemma}
    use_roles = {q[4:] for _, lits, _ in rules for q, t in lits if q.startswith("use.")} | set(use_rules)
    pair_units = collections.defaultdict(list)                                  # role -> (unit, ja, vv) for the lookup writes
    for role in sorted(use_roles):
        if not slot_vocab.get(f"{role}.noun") or not vunits: continue
        for ja, vv in sorted(pairs, key=str):
            if ("slot", f"{role}.noun2", ja) in S:
                u = L1.add(("usepair", role, "noun2", ja, vv)); W1.append((u, ("S", S[("slot", "verb", vv)]), 1.0)); W1.append((u, ("S", S[("slot", f"{role}.noun2", ja)]), 1.0)); b1[u] = -1.0
                use_pairs[role].append(u); pair_units[role].append((u, ja, vv))
            if ("slot", f"{role}.noun", ja) in S:
                u = L1.add(("usepair", role, "noun", ja, vv)); W1.append((u, ("S", S[("slot", "verb", vv)]), 1.0)); W1.append((u, ("S", S[("slot", f"{role}.noun", ja)]), 1.0)); b1[u] = -1.0
                if ("slot", f"{role}.compound", True) in S: W1.append((u, ("S", S[("slot", f"{role}.compound", True)]), -1.0))   # the head is noun2 then
                use_pairs[role].append(u); pair_units[role].append((u, ja, vv))
    # layer 0 inside the cell: the table rows that join the frame's verb with the window — one AND unit per row over the verb's
    # unit and the window's table keys (w<j>.tbl=<token>, w<j>.prole=r), ORed at layer 2 into hit=tail,k / hit=pafter / hit=use
    hit_units = collections.defaultdict(list)
    for vv in sorted(slot_vocab.get("verb", []), key=str):
        if "+" in vv[0]: continue                                               # a use-rewritten verb is past its tables
        for hit, keys in hit_rows(vv[0]):
            u = L1.add(("hitrow", hit, vv, keys)); W1.append((u, ("S", S[("slot", "verb", vv)]), 1.0))
            for j, (kind, val) in keys: W1.append((u, ("X", X.add(("pred", f"w{j}.{kind}={val}"))), 1.0))
            b1[u] = -float(len(keys)); hit_units[hit].append(u)
    # ---- layer 2: pp.count, NOT-derived (1 - sum of the conjunctions), use.<role> (OR of pairs), the verb hits (OR of rows)
    L2 = Index(); W2 = []; b2 = {}
    u = L2.add(("pred", "pp.count=0")); b2[u] = 1.0
    for k in range(3): W2.append((u, ("L1", f[k]), -1.0))
    for n in (1, 2):
        u = L2.add(("pred", f"pp.count={n}")); W2.append((u, ("L1", f[n-1]), 1.0)); W2.append((u, ("L1", f[n]), -1.0))
    u = L2.add(("pred", "pp.count=3")); W2.append((u, ("L1", f[2]), 1.0))
    nor_units(derived, derived_terms, L2, W2, b2)
    for role, units in use_pairs.items():
        u = L2.add(("pred", f"use.{role}"))
        for t in units: W2.append((u, ("L1", t), 1.0))
    for hit, units in sorted(hit_units.items()):
        u = L2.add(("pred", hit))
        for t in units: W2.append((u, ("L1", t), 1.0))
    # ---- layer 3: rules = AND(control, literals)
    L3 = Index(); W3 = []; b3 = {}
    def pred_ref(q):
        if ("pred", q) in L1: return ("L1", L1[("pred", q)])
        if ("pred", q) in L2: return ("L2", L2[("pred", q)])
        if ("pred", q) in X: return ("X", X[("pred", q)])
        return None
    unresolved = collections.Counter()
    def rule_edges(q):
        ref = pred_ref(q)
        return [ref] if ref is not None else base_edges(q)
    rule_units(rules, derived, rule_edges, lambda prev: ("S", S[("ctrl", prev)]), ("X", X[("one",)]), L2, L3, W3, b3, unresolved, quiet=("M", "OPEN", "TURN", "CLOSE"))
    # skip-state rules: SKIP(ri, j) -> SKIP(ri, j-1) or the target
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        for j in range(1, k):
            u = L3.add(("skiprule", ri, j)); W3.append((u, ("S", S[("ctrl", ("SKIP", ri, j))]), 1.0)); b3[u] = 0.0
    # ---- layer 4: next control, shift, gates
    L4 = Index(); W4 = []; b4 = {}
    def rule_target(ri, k, cp): return ("ctrl", ("SKIP", ri, k - 1)) if k > 1 else ("ctrl", cp)
    shift = L4.add(("shift",))
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        tgt = rule_target(ri, k, cp) if k >= 1 else ("ctrl", cp)
        u = L4.add(("next", tgt)); W4.append((u, ("L3", L3[("rule", ri)]), 1.0))
        if k >= 1: W4.append((shift, ("L3", L3[("rule", ri)]), 1.0))
        for j in range(1, k):
            nxt = ("ctrl", ("SKIP", ri, j - 1)) if j > 1 else ("ctrl", cp)
            u2 = L4.add(("next", nxt)); W4.append((u2, ("L3", L3[("skiprule", ri, j)]), 1.0)); W4.append((shift, ("L3", L3[("skiprule", ri, j)]), 1.0))
    # flush: OR of the rules that hand the working frame to the context, as shift is of the rules that consume a token
    flush = L4.add(("flush",))
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        if FLUSH in actions: W4.append((flush, ("L3", L3[("rule", ri)]), 1.0))
    # gates per slot (any write/clear) and per (slot, source) copy / (slot, from, fn) move
    gate_slot = {}; gate_copy = {}; gate_move = {}; const_w = collections.defaultdict(list)
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        for slot, v, src in actions:
            if src[0] == "flush": continue
            g = gate_slot.setdefault(slot, L4.add(("gate", slot))); W4.append((g, ("L3", L3[("rule", ri)]), 1.0))
            if src[0] == "copy":
                gc = gate_copy.setdefault((slot, src[1]), L4.add(("gcopy", slot, src[1]))); W4.append((gc, ("L3", L3[("rule", ri)]), 1.0))
            elif src[0] in MOVE_FN:
                gm = gate_move.setdefault((slot, src[1], src[0]), L4.add(("gmove", slot, src[1], src[0]))); W4.append((gm, ("L3", L3[("rule", ri)]), 1.0))
            elif src[0] == "const":
                const_w[(slot, v)].append(ri)
            elif src[0] == "use":                                            # the lookup rewrites the verb and the particle for every pair
                for target in ("verb", f"{slot}_part" if slot.startswith("pp") else "obj_particle"):
                    gt = gate_slot.setdefault(target, L4.add(("gate", target))); W4.append((gt, ("L3", L3[("rule", ri)]), 1.0))
    # the USE lookup gates: any feedback rule fired (guse); AND a stative pair is active (gusestat: prog <- True);
    # AND a "-" pair is active (gusedash: the role's noun phrase is dropped, take a rest -> 休む)
    guse = {}; gusestat = {}; gusedash = {}
    for role, ris in use_rules.items():
        g = L4.add(("guse", role)); guse[role] = g
        for ri in ris: W4.append((g, ("L3", L3[("rule", ri)]), 1.0))
        stat = [u for u, ja, vv in pair_units.get(role, []) if USE_TABLE[(ja, vv[0])][2]]
        dash = [u for u, ja, vv in pair_units.get(role, []) if USE_TABLE[(ja, vv[0])][1] == "-"]
        for name, units, store in (("gusestat", stat, gusestat), ("gusedash", dash, gusedash)):
            if not units: continue
            gu = L4.add((name, role)); b4[gu] = -1.0; store[role] = gu
            for ri in ris: W4.append((gu, ("L3", L3[("rule", ri)]), 1.0))
            for u in units: W4.append((gu, ("L1", u), 1.0))
    # pend_prep gate: a preposition in slot 0 AND this step consumes it (any rule with k >= 1, or a skip step)
    gp = L4.add(("gate", "pend")); b4[gp] = -1.0
    for w in preps: W4.append((gp, ("X", X[("prep0", w)]), 1.0))
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        if k >= 1: W4.append((gp, ("L3", L3[("rule", ri)]), 1.0))
        for j in range(1, k): W4.append((gp, ("L3", L3[("skiprule", ri, j)]), 1.0))
    # ---- layer 5: keep / write per slot value
    L5 = Index(); W5 = []; b5 = {}
    for slot in sorted(slot_vocab):
        g = gate_slot.get(slot)
        for v in slot_vocab[slot]:
            si = S[("slot", slot, v)]
            kp = L5.add(("keep", slot, v)); W5.append((kp, ("S", si), 1.0))
            if g is not None: W5.append((kp, ("L4", g), -1.0))
            if slot == "prog":
                for gu in gusestat.values(): W5.append((kp, ("L4", gu), -1.0))
            if "." in slot and slot.split(".")[0] in gusedash: W5.append((kp, ("L4", gusedash[slot.split(".")[0]]), -1.0))
            for (sl, s) in copy_pairs:
                if sl == slot and v in src_vocab.get(s, []) and ("src", s, v) in X:
                    wu = L5.add(("write", slot, s, v)); W5.append((wu, ("L4", gate_copy[(sl, s)]), 1.0)); W5.append((wu, ("X", X[("src", s, v)]), 1.0)); b5[wu] = -1.0
            for (sl, from_slot, fn) in move_pairs:
                if sl != slot: continue
                for v0 in slot_vocab.get(from_slot, []):
                    if MOVE_FN[fn](v0) == v:
                        wu = L5.add(("writem", slot, from_slot, fn, v0, v)); W5.append((wu, ("L4", gate_move[(sl, from_slot, fn)]), 1.0)); W5.append((wu, ("S", S[("slot", from_slot, v0)]), 1.0)); b5[wu] = -1.0
            if (slot, v) in const_w:
                wu = L5.add(("writec", slot, v))                       # a constant write is its own unit: OR of the rules that write it
                for ri in const_w[(slot, v)]: W5.append((wu, ("L3", L3[("rule", ri)]), 1.0))
    for w in preps:
        kp = L5.add(("keep", "pend", w)); W5.append((kp, ("S", S[("pend", w)]), 1.0)); W5.append((kp, ("L4", gp), -1.0))
        wu = L5.add(("write", "pend", w)); W5.append((wu, ("X", X[("prep0", w)]), 1.0)); W5.append((wu, ("L4", gp), 1.0)); b5[wu] = -1.0
    # the USE lookup writes: AND(the gate, the pair unit), one unit per target value; prog through gusestat; no pair -> FAIL
    for role, g in guse.items():
        by_target = collections.defaultdict(list)
        for u, ja, vv in pair_units.get(role, []):
            new_verb, upart, stative = USE_TABLE[(ja, vv[0])]
            by_target[("verb", new_verb)].append(u)
            if upart == "-": continue
            if role.startswith("pp"): by_target[(f"{role}_part", (upart, upart))].append(u)
            else: by_target[("obj_particle", upart)].append(u)
        for (slot, v), units in sorted(by_target.items(), key=str):
            if ("slot", slot, v) not in S: continue
            wu = L5.add(("writeu", slot, v, role)); W5.append((wu, ("L4", g), 1.0)); b5[wu] = -1.0
            for u in units: W5.append((wu, ("L1", u), 1.0))
        if role in gusestat and ("slot", "prog", True) in S:
            wu = L5.add(("writeu", "prog", True, role)); W5.append((wu, ("L4", gusestat[role]), 1.0))
        cu = L5.add(("nouse", role)); W5.append((cu, ("L4", g), 1.0))
        for u, ja, vv in pair_units.get(role, []): W5.append((cu, ("L1", u), -1.0))
    # the crash: a copy gate with no reading of its source in slot 0 -> FAIL (the rule fired on a token it was not written for)
    for (slot, s), gc in gate_copy.items():
        srcs = [X[("src", s, v)] for v in src_vocab.get(s, []) if ("src", s, v) in X]
        if not srcs: continue
        cu = L5.add(("nosrc", slot, s)); W5.append((cu, ("L4", gc), 1.0))
        for xi in srcs: W5.append((cu, ("X", xi), -1.0))
    # ---- layer 6: the new state = next control + shift + flush + keep + write
    W6 = []
    for i, name in enumerate(L4.names):
        if name[0] == "next": W6.append((S[name[1]], ("L4", i), 1.0))
        if name[0] in ("shift", "flush"): W6.append((S[name], ("L4", i), 1.0))
    for i, name in enumerate(L5.names):
        if name[0] in ("keep", "write", "writec", "writem"):
            key = ("pend", name[2]) if name[1] == "pend" else ("slot", name[1], name[-1])
            W6.append((S[key], ("L5", i), 1.0))
        if name[0] == "writeu": W6.append((S[("slot", name[1], name[2])], ("L5", i), 1.0))
        if name[0] in ("nosrc", "nouse"): W6.append((S[("ctrl", FAIL)], ("L5", i), 1.0))
    layers = [(L1, W1, b1), (L2, W2, b2), (L3, W3, b3), (L4, W4, b4), (L5, W5, b5)]
    return {"S": S, "X": X, "layers": layers, "W6": W6, "rules": rules, "slot_vocab": slot_vocab, "src_vocab": src_vocab,
            "unresolved": unresolved, "derived": derived, "dead_conj": dead_conj}

# ---------------------------------------------------------------------------------------------- the decoder cell
def compile_decoder(slot_vocab):
    """the printing rules -> the decoder cell, over the frame the encoder leaves (slot_vocab: its slots and their values).
    The input is the frame under the attention — one unit per value of every slot, in the encoder's order — and the constant.
    The state is the printing position, the sentence block (SENT_SLOTS: set from the encoder's final state, then kept), and what
    a step puts out: the emit unit with one output unit — a morpheme: a constant, a slot's value, a form of the verb or of the
    adjective — or an attention unit, down or up.  The same five layers as the encoder: conjunctions over the frame, their
    NOTs, the rules, the gates (next position, emit, down, up, a read gate per slot read, a form gate per stem form), and the
    read and lookup units, AND(gate, the frame's value), with the crash of a read that finds nothing"""
    rules, derived = decoder_rules()
    emits = [(ri, v) for ri, (prev, lits, (cp, k, actions)) in enumerate(rules) for slot, v, src in actions if src == PRINT_SRC]
    moves = [(ri, v) for ri, (prev, lits, (cp, k, actions)) in enumerate(rules) for slot, v, src in actions if src == HEAD_SRC]
    out_tokens = sorted({v for ri, v in emits}, key=str)
    const_out = sorted({tok[1] for tok in out_tokens if tok[0] == "c"})
    read_out = sorted({(tok[1], tok[2] if len(tok) > 2 else None) for tok in out_tokens if tok[0] == "s"}, key=str)
    form_out = sorted({(tok[1], tok[2]) for tok in out_tokens if tok[0] == "vf"})
    aform_out = sorted({tok[2] for tok in out_tokens if tok[0] == "af"})
    # ---- state: the printing position, the sentence block, y(t) — one output unit per morpheme — and the head units
    S = Index()
    for c in sorted({FAIL, ("P", "P0"), ("P", "DONE")} | {prev for prev, _, _ in rules} | {out[0] for _, _, out in rules}, key=str): S.add(("ctrl", c))
    for slot in SENT_SLOTS:
        for v in sorted(slot_vocab.get(slot, []), key=str): S.add(("slot", slot, v))
    for s_ in const_out: S.add(("out", "c", s_))
    for slot, idx in read_out:
        for v in slot_vocab.get(slot, []): S.add(("out", "s", slot, idx, v))
    for name, alt in form_out:
        for v in slot_vocab.get("verb", []):
            if form_ok(v, name, alt): S.add(("out", "vf", name, alt, v))
    for strip in aform_out:
        for v in slot_vocab.get("cop_adj", []): S.add(("out", "af", strip, v))
    for head in ("emit", "down", "up"): S.add((head,))
    # ---- input: the attended frame, unit for unit the encoder's slot block, and the constant
    X = Index()
    for slot in sorted(slot_vocab):
        for v in sorted(slot_vocab[slot], key=str): X.add(("slot", slot, v))
    X.add(("one",))
    def unit(slot, v): return ("S", S[("slot", slot, v)]) if slot in SENT_SLOTS else ("X", X[("slot", slot, v)])
    def edges(q):
        sv = slot_values(q, slot_vocab)
        if sv is None: raise ValueError(f"a printing rule reads {q}, which is not a literal over the frame")
        return [unit(sv[0], v) for v in sv[1]] or None
    # ---- layers 1-3: the conjunctions of the derived predicates, their NOTs, the rules
    L1 = Index(); W1 = []; b1 = {}; L2 = Index(); W2 = []; b2 = {}; L3 = Index(); W3 = []; b3 = {}
    terms, dead_conj = conjunction_units(derived, lambda name, q: edges(q), L1, W1, b1)
    nor_units(derived, terms, L2, W2, b2)
    unresolved = collections.Counter()
    rule_units(rules, derived, edges, lambda prev: ("S", S[("ctrl", prev)]), ("X", X[("one",)]), L2, L3, W3, b3, unresolved, quiet=("P",))
    # ---- layer 4: the next position; emit (OR of the rules that emit, as shift is of the rules that consume); down and up (the
    #      rules that move the attention); a read gate per slot read, a form gate per verb form and per adjective form; a constant
    #      morpheme is written by its rules directly (layer 5)
    L4 = Index(); W4 = []; b4 = {}
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        u = L4.add(("next", ("ctrl", cp))); W4.append((u, ("L3", L3[("rule", ri)]), 1.0))
    emit = L4.add(("emit",)); gate_read = {}; gate_form = {}; gate_aform = {}; const_out_w = collections.defaultdict(list)
    for ri, tok in emits:
        W4.append((emit, ("L3", L3[("rule", ri)]), 1.0))
        if tok[0] == "c": const_out_w[tok[1]].append(ri)
        elif tok[0] == "s":
            g = gate_read.setdefault((tok[1], tok[2] if len(tok) > 2 else None), L4.add(("gread", tok[1], tok[2] if len(tok) > 2 else None))); W4.append((g, ("L3", L3[("rule", ri)]), 1.0))
        elif tok[0] == "vf":
            g = gate_form.setdefault((tok[1], tok[2]), L4.add(("gform", tok[1], tok[2]))); W4.append((g, ("L3", L3[("rule", ri)]), 1.0))
        elif tok[0] == "af":
            g = gate_aform.setdefault(tok[2], L4.add(("gaform", tok[2]))); W4.append((g, ("L3", L3[("rule", ri)]), 1.0))
    for move in ("down", "up"):
        g = L4.add((move,))
        for ri, mv in moves:
            if mv == move: W4.append((g, ("L3", L3[("rule", ri)]), 1.0))
    # ---- layer 5: the sentence block kept; a read unit per value = AND(read gate, the frame's value), the mirror of the encoder's
    #      write unit; a lookup unit per verb or adjective value = AND(form gate, the value); a constant morpheme = OR of the rules
    #      that emit it; and the crash of a read with nothing in the block or a form the verb has not -> FAIL (the mirror of nosrc)
    L5 = Index(); W5 = []; b5 = {}
    for slot in SENT_SLOTS:
        for v in slot_vocab.get(slot, []):
            kp = L5.add(("keep", slot, v)); W5.append((kp, ("S", S[("slot", slot, v)]), 1.0))
    for (slot, idx), g in gate_read.items():
        for v in slot_vocab.get(slot, []):
            ru = L5.add(("read", slot, idx, v)); W5.append((ru, ("L4", g), 1.0)); W5.append((ru, unit(slot, v), 1.0)); b5[ru] = -1.0
        cu = L5.add(("noread", slot, idx)); W5.append((cu, ("L4", g), 1.0))
        for v in slot_vocab.get(slot, []): W5.append((cu, unit(slot, v), -1.0))
    for (name, alt), g in gate_form.items():
        valid = [v for v in slot_vocab.get("verb", []) if form_ok(v, name, alt)]
        for v in valid:
            lu = L5.add(("form", name, alt, v)); W5.append((lu, ("L4", g), 1.0)); W5.append((lu, unit("verb", v), 1.0)); b5[lu] = -1.0
        cu = L5.add(("noform", name, alt)); W5.append((cu, ("L4", g), 1.0))
        for v in valid: W5.append((cu, unit("verb", v), -1.0))
    for strip, g in gate_aform.items():
        for v in slot_vocab.get("cop_adj", []):
            lu = L5.add(("aform", strip, v)); W5.append((lu, ("L4", g), 1.0)); W5.append((lu, unit("cop_adj", v), 1.0)); b5[lu] = -1.0
        cu = L5.add(("noaform", strip)); W5.append((cu, ("L4", g), 1.0))
        for v in slot_vocab.get("cop_adj", []): W5.append((cu, unit("cop_adj", v), -1.0))
    for s_, ris in const_out_w.items():
        ou = L5.add(("outc", s_))
        for ri in ris: W5.append((ou, ("L3", L3[("rule", ri)]), 1.0))
    # ---- the readout: the next position, the head units, the sentence block, y(t)
    W6 = []
    for i, name in enumerate(L4.names):
        if name[0] == "next": W6.append((S[name[1]], ("L4", i), 1.0))
        if name[0] in ("emit", "down", "up"): W6.append((S[name], ("L4", i), 1.0))
    for i, name in enumerate(L5.names):
        if name[0] == "keep": W6.append((S[("slot", name[1], name[2])], ("L5", i), 1.0))
        if name[0] in ("noread", "noform", "noaform"): W6.append((S[("ctrl", FAIL)], ("L5", i), 1.0))
        if name[0] == "read": W6.append((S[("out", "s", name[1], name[2], name[3])], ("L5", i), 1.0))
        if name[0] == "form": W6.append((S[("out", "vf", name[1], name[2], name[3])], ("L5", i), 1.0))
        if name[0] == "aform": W6.append((S[("out", "af", name[1], name[2])], ("L5", i), 1.0))
        if name[0] == "outc": W6.append((S[("out", "c", name[1])], ("L5", i), 1.0))
    layers = [(L1, W1, b1), (L2, W2, b2), (L3, W3, b3), (L4, W4, b4), (L5, W5, b5)]
    return {"S": S, "X": X, "layers": layers, "W6": W6, "rules": rules, "derived": derived, "unresolved": unresolved, "dead_conj": dead_conj}

class Cell:
    """a compiled cell as sparse matrices: acts = [s; x], each of the five ReLU layers appended to acts, then the readout into s'"""
    def __init__(self, c):
        self.c = c; S, X = c["S"], c["X"]; self.S, self.X = S, X
        self.nS, self.nX = len(S), len(X)
        self.offsets = {"S": 0, "X": self.nS}; width = self.nS + self.nX
        self.mats = []
        for (L, W, b) in c["layers"]:
            rows_, cols_, vals_ = [], [], []
            for (u, ref, w) in W:
                if isinstance(ref, tuple): src, j = ref
                else: src, j = "S", ref
                rows_.append(u); cols_.append(self.offsets[src] + j); vals_.append(w)
            M = sp.csr_matrix((vals_, (rows_, cols_)), shape=(len(L), width)); M.sum_duplicates(); M.eliminate_zeros()
            bias = np.zeros(len(L))
            for u, v in b.items(): bias[u] = v
            self.mats.append((M, bias)); self.offsets["L" + str(len(self.mats))] = width; width += len(L)
        rows_, cols_, vals_ = [], [], []
        for (si, ref, w) in c["W6"]:
            rows_.append(si); cols_.append(self.offsets[ref[0]] + ref[1]); vals_.append(w)
        self.W6 = sp.csr_matrix((vals_, (rows_, cols_)), shape=(self.nS, width))
        self.width = width
        self.ctrl_idx = np.array([i for i, n in enumerate(S.names) if n[0] == "ctrl"]); self.fail_unit = S[("ctrl", FAIL)]
    def nnz(self): return sum(M.nnz for M, _ in self.mats) + self.W6.nnz
    def step(self, s, x):
        """one step: the five layers and the readout, acts = [s; x] -> s'"""
        acts = np.concatenate([s, x])
        for M, bias in self.mats:
            h = np.maximum(0.0, M @ acts + bias); acts = np.concatenate([acts, h])
        return np.maximum(0.0, self.W6 @ acts)
    def control(self, s):
        """the control point a state holds: its name; FAIL when the refusal unit is on; None when no rule fired; ("MULTI", n) when n did"""
        if s[self.fail_unit] > 0.5: return FAIL
        on = self.ctrl_idx[s[self.ctrl_idx] > 0.5]
        return self.S.names[on[0]][1] if len(on) == 1 else ("MULTI", len(on)) if len(on) else None

class MatrixParser:
    """the compiled machine: an encoder cell, the context it leaves, a decoder cell.
    The read head feeds the encoder one window per step and advances when its shift unit is on; when its flush unit is on, the
    slot block of its state — the frame of the clause it has just read — is appended to the context.  The decoder starts at the
    first printing position with the sentence block of the encoder's final state; its input is the frame under the attention
    head, which starts on the first frame and moves when the decoder's down or up unit is on; when its emit unit is on, the one
    output unit that is on is appended to the output.  Nothing else passes between the two cells"""
    name = "matrix"
    def __init__(self, compiled):
        self.c = compiled; self.enc = Cell(compiled["encoder"]); self.dec = Cell(compiled["decoder"])
        E, D = self.enc.S, self.dec.S
        self.frame_units = [n for n in E.names if n[0] == "slot"]                         # the coordinates of a frame: the encoder's slot block
        assert self.frame_units + [("one",)] == self.dec.X.names, "the decoder's input is not the encoder's frame and the constant"
        self.frame_idx = np.array([E[n] for n in self.frame_units]); self.frame_pos = {n: j for j, n in enumerate(self.frame_units)}
        self.shift_unit, self.flush_unit = E[("shift",)], E[("flush",)]
        self.emit_unit, self.down_unit, self.up_unit = D[("emit",)], D[("down",)], D[("up",)]
        self.out_units = [i for i, n in enumerate(D.names) if n[0] == "out"]
        self.handoff = [(D[n], E[n]) for n in D.names if n[0] == "slot"]                  # the sentence block: decoder register <- the encoder's final state
    def nnz(self): return self.enc.nnz() + self.dec.nnz()
    def step_encoder(self, s, x): return self.enc.step(s, x)
    def step_decoder(self, s, x): return self.dec.step(s, x)
    def lexical_input(self, toks, i):
        """the lexical layer's output for the window at i: class / surface / extra predicates of the four slots"""
        x = np.zeros(self.enc.nX); X = self.enc.X
        allw = window_features(toks, i)
        for k in range(WINDOW):
            surface, classes, extra = allw[k]
            for cq in [f"w{k}.class={c}" for c in classes] + ([f"w{k}.surf={surface}"] if surface != "*" else []):
                if ("pred", cq) in X: x[X[("pred", cq)]] = 1
            for e in extra:
                qs = [f"w{k}.vc={c}" for c in e[1]] if e[0] == "vc" else [f"w{k}.{e[0]}={e[1]}"]
                for q in qs:
                    if ("pred", q) in X: x[X[("pred", q)]] = 1
            tok = toks[i + k] if i + k < len(toks) else None                                 # the token as a table key, for the hit rows
            if tok is not None and ("pred", f"w{k}.tbl={tok}") in X: x[X[("pred", f"w{k}.tbl={tok}")]] = 1
        return x
    def input_vector(self, toks, i):
        """the whole input of one encoder step: the window's readings and table keys, the hits the window alone decides, the copy
        sources of slot 0, the preposition in slot 0, and the constant — nothing read from the state: the hits the frame's verb
        decides are units of the cell"""
        X = self.enc.X; x = self.lexical_input(toks, i)
        for h in window_hits(toks, i):
            q = "hit=" + ",".join(map(str, h))
            if ("pred", q) in X: x[X[("pred", q)]] = 1
        tok = toks[i] if i < len(toks) else None
        for sname, v in source_values(tok).items():
            if ("src", sname, v) in X: x[X[("src", sname, v)]] = 1
        for sname, v in window_sources(toks, i).items():
            if ("src", sname, v) in X: x[X[("src", sname, v)]] = 1
        if i + 2 < len(toks) and toks[i] == toks[i + 2] and ("pred", "w0=w2") in X: x[X[("pred", "w0=w2")]] = 1
        if tok in R.PREPS: x[X[("prep0", tok)]] = 1
        x[X[("one",)]] = 1
        return x
    def frame_dict(self, vec):
        """a frame of the context as slot -> value"""
        return {self.frame_units[j][1]: self.frame_units[j][2] for j in np.nonzero(vec > 0.5)[0]}
    def frame_vector(self, F):
        """slot -> value as a frame of the context: one unit per slot value; a value the cell's vocabulary does not have is refused"""
        vec = np.zeros(len(self.frame_units))
        for slot, v in F.items():
            if v in (None, False, ""): continue
            if ("slot", slot, v) in self.frame_pos: vec[self.frame_pos[("slot", slot, v)]] = 1.0
            elif slot in self.c["slot_vocab"] or slot in MARKS: raise ParseFail(f"PRINT: {v!r} is not a value of {slot} in this cell")     # (a mark the cell has no block for says what the frame is to its neighbours: never dropped)
        return vec
    def emitted(self, s):
        """the output unit a decoder state carries when its emit unit is on: [] when the step emitted nothing, else the one unit"""
        if s[self.emit_unit] <= 0.5: return []
        on = [self.dec.S.names[i] for i in self.out_units if s[i] > 0.5]
        if len(on) != 1: raise ParseFail(f"PRINT: {len(on)} output units on")
        return on
    def encode(self, toks, bound_per_token=12, on_step=None):
        """the encoder under the read head: one step per parser transition, the window advancing when `shift` is on (k=0
        transitions do not), the working frame appended to the context when `flush` is on — at a clause boundary and at the end;
        -> (the context: the frames in the order they were flushed, the encoder's final state)"""
        E = self.enc; s = np.zeros(E.nS); s[E.S[("ctrl", ("START",))]] = 1.0; i = 0; frames = []
        for _ in range(bound_per_token * (len(toks) + 2)):
            ctrl = E.control(s)
            if ctrl == ("DONE",): return frames, s
            if ctrl == FAIL: raise ParseFail(f"MATRIX: FAIL at token {i} ({toks[i] if i < len(toks) else 'END'!r})", i)
            if ctrl is None: raise ParseFail(f"MATRIX: no rule fired at token {i} ({toks[i] if i < len(toks) else 'END'!r})", i)
            if ctrl[0] == "MULTI": raise ParseFail(f"MATRIX: {ctrl[1]} rules fired at token {i}", i)
            if i > len(toks): raise ParseFail("MATRIX: ran past the end", i)
            x = self.input_vector(toks, i); s2 = self.step_encoder(s, x)
            if on_step: on_step("encoder", s, x, s2)
            s = s2
            if s[self.flush_unit] > 0.5: frames.append(np.array(s[self.frame_idx]))
            if s[self.shift_unit] > 0.5: i += 1
        raise ParseFail("MATRIX: did not reach DONE", i)
    def decode(self, frames, final=None, bound=600, on_step=None):
        """the decoder under the attention head: from the first printing position, with the sentence block of the encoder's final
        state, one step per printing rule on the frame the head is on; `down` and `up` move the head to the next and the previous
        frame, `emit` appends the one output unit that is on; -> the output units, in order"""
        D = self.dec; s = np.zeros(D.nS); s[D.S[("ctrl", ("P", "P0"))]] = 1.0; at = 0; out = []; one = np.ones(1)
        if final is not None:
            for d, e in self.handoff: s[d] = final[e]
        for _ in range(bound):
            ctrl = D.control(s)
            if ctrl == ("P", "DONE"): return out
            if ctrl == FAIL: raise ParseFail(f"PRINT: FAIL after {len(out)} morphemes")
            if ctrl is None: raise ParseFail(f"PRINT: no rule fired after {len(out)} morphemes")
            if ctrl[0] == "MULTI": raise ParseFail(f"PRINT: {ctrl[1]} rules fired after {len(out)} morphemes")
            if not 0 <= at < len(frames): raise ParseFail("PRINT: the attention left the context")
            x = np.concatenate([frames[at], one]); s2 = self.step_decoder(s, x)
            if on_step: on_step("decoder", s, x, s2)
            s = s2
            out += self.emitted(s)
            if s[self.down_unit] > 0.5: at += 1
            if s[self.up_unit] > 0.5: at -= 1
        raise ParseFail("PRINT: did not finish")
    def parse(self, toks, bound_per_token=12):
        """the parser interface: the Clause the context's frames spell (frames_to_clause)"""
        return frames_to_clause([self.frame_dict(f) for f in self.encode(list(toks), bound_per_token)[0]])
    def run(self, toks):
        """encode, then decode: -> (the context as slot dicts, the output units)"""
        frames, final = self.encode(list(toks))
        return [self.frame_dict(f) for f in frames], self.decode(frames, final)
    def translate(self, toks):
        """the whole machine on a sentence: the output lexical layer's strings of the emitted units, joined"""
        return "".join(output_lexicon(u) for u in self.run(toks)[1])
    def split_np(self, role, s):
        """a noun phrase's Japanese -> its sub-slots (det poss num adj1 adj2 noun [noun2] [mono]) from the cell's vocabulary of the
        role, the first decomposition that spells the string; None if none does"""
        V = self.c["slot_vocab"]; subs = ("det", "poss", "num", "adj1", "adj2")
        def opts(sub): return sorted((v for v in V.get(f"{role}.{sub}", []) if isinstance(v, str)), key=len, reverse=True)
        def head(rest):
            for noun in opts("noun"):
                if not rest.startswith(noun): continue
                tail = rest[len(noun):]
                if tail == "": return {"noun": noun}
                for n2 in opts("noun2"):
                    if tail.startswith("の" + n2):
                        t2 = tail[len(n2) + 1:]
                        if t2 == "": return {"noun": noun, "noun2": n2}
                        if t2 in opts("mono"): return {"noun": noun, "noun2": n2, "mono": t2}
                if tail in opts("mono"): return {"noun": noun, "mono": tail}
            return None
        def rec(k, rest):
            if k == len(subs): return head(rest)
            if subs[k] == "num" and not role.endswith("of"):                      # an of-phrase, "<inner>の", may precede the number
                for j, ch in enumerate(rest):
                    if ch != "の": continue
                    inner = self.split_np(role + "of", rest[:j]); r = rec(k, rest[j + 1:]) if inner is not None else None
                    if inner is not None and r is not None:
                        for sub, v in inner.items(): r["of:" + sub] = v
                        return r
            for v in opts(subs[k]):
                if rest.startswith(v):
                    r = rec(k + 1, rest[len(v):])
                    if r is not None: r[subs[k]] = v; return r
            return rec(k + 1, rest)
        return rec(0, s)
    def clause_context(self, clause):
        """the context a Clause would leave, for the decoder alone: its frames (gen_module.frames_of), every noun phrase split into
        the cell's sub-slots by its vocabulary; a flat frame, or a list of them, is taken as it is"""
        if isinstance(clause, dict): return [self.frame_vector(clause)]
        if isinstance(clause, list): return [self.frame_vector(F) for F in clause]
        frames = []
        for F in GM.frames_of(clause):
            for role in GM.NP_ROLES:
                if F.get(f"{role}.noun") is None: continue
                parts = self.split_np(role, (F.pop(f"{role}.poss", None) or "") + F.pop(f"{role}.noun"))
                if parts is None: raise ParseFail(f"PRINT: the {role} noun phrase is not spelled by this cell's vocabulary")
                for sub, v in parts.items(): F[f"{role}of.{sub[3:]}" if sub.startswith("of:") else f"{role}.{sub}"] = v
            frames.append(self.frame_vector(F))
        return frames
    def print_clause(self, clause):
        """the decoder on the context of a Clause: -> the output units"""
        return self.decode(self.clause_context(clause))

class Layer0Lexical:
    """the lexical layer as layer 0 of the cell: the units that compute the verb-conditioned table hits from the window's table
    keys and the verb block (the row units of layer 1 and the hit predicates of layer 2)"""
    def hit_units(self, comp):
        L1, L2 = comp["encoder"]["layers"][0][0], comp["encoder"]["layers"][1][0]
        return [n for n in L1.names if n[0] == "hitrow"] + [n for n in L2.names if n[0] == "pred" and verb_hit(n[1])]

class HandModules:
    """the three clause-level machines translated by hand (pp_module.py, predicate_module.py), in the module rule format"""
    def pp_rules(self):
        import pp_module as PPM; return PPM.rules()
    def predicate_rules(self):
        import predicate_module as PDM; return PDM.pred_rules()
    def copula_rules(self):
        import predicate_module as PDM; return PDM.cop_rules()
