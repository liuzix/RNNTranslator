"""The parser layer, compiled to sparse matrices.

MatrixParser implements the parser interface (parser.ParserInterface): a left-to-right machine that reads the lexical
layer's output for a 4-token window and updates a one-hot state vector with five sparse ReLU layers per step; a `shift`
unit tells the window when to advance, and the frame is read out of the state at the end of the sentence.

Two kinds of rules make up the cell:
  * clause-level transitions extracted from the rule parser: it is traced on the corpus, each token consumption becomes a
    row (control point, predicates, outcome), and per control point a decision tree over the predicates it read is
    induced and flattened into AND units.  Their guards are only as complete as the corpus.
  * hand-translated modules with complete guards (np_module for parse_np), spliced in as sub-automata: a k=0 entry
    transition writes the return site into `ret` and clears the role's slots; the module's RET state hands back through an
    exit transition carrying the post-NP effects.
Register writes are factored into copy (slot <- a reading of the current token), const, move / move_time (slot <- another
slot) and clear.  Whatever the cell cannot represent (an inner clause, a register with no block) becomes a FAIL row, so
the machine refuses there instead of guessing; inside its scope it must agree with the rule parser exactly.
"""
import sys, collections, itertools
import numpy as np
import scipy.sparse as sp
import rules_ja as R
from parser_rules import Parser, Clause, ParseFail, Overflow, parse_bounded

WINDOW = 4
NP_ROLES = ("subj", "obj", "iobj", "pp1", "pp2", "pp3", "cop", "cmp")        # cmp: the noun phrase of a than / as comparison
NP_SUB = ("det", "poss", "num", "adj1", "adj2", "noun", "cls")
FLAGS = ("past", "neg", "prog", "perf", "question", "imp", "please", "lets", "there", "tag", "coord", "sub", "neg_inner", "noless")
ENUMS = ("modal", "want", "degree", "obj_particle", "wh", "time", "adv1", "adv2", "front", "cop_kind", "cop_adj", "cop_adv", "cop_part",
         "pp1_part", "pp2_part", "pp3_part", "cmp_kind", "verb", "purpose", "causative")
UNSUPPORTED_REGS = ("infinitive", "control", "embedded", "purpose_clause", "pre_clause", "manner", "so_that", "howabout",
                    "exclaim", "fixed_out", "relative", "suffix")
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
    reading classes, and the class-level extras (verb classes, noun class, adjective kind)"""
    feats = []
    for k in range(WINDOW):
        t = toks[i + k] if i + k < len(toks) else None
        rs = readings_of(t); classes = tuple(sorted(set(c for c, _ in rs)))
        surface = t if (t is None or not (set(classes) & OPEN) or (set(classes) & CLOSED_CLASSES) or t in IDENTITY_WORDS) else "*"
        extra = []
        for c, v in rs:
            if c == "verb":
                extra.append(("vc", tuple(sorted(cls for cls, s in R.VC.items() if v[0] in s)))); extra.append(("vform", v[1]))
            elif c == "noun": extra.append(("ncls", v[1]))
            elif c == "adj": extra.append(("akind", v[1]))
            elif c == "pron": extra.append(("prole", v[1]))
        feats.append((surface, classes, tuple(sorted(set(extra)))))
    return tuple(feats)

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

# ----------------------------------------------------------------------------------------------- frame as slots
def snapshot(c):
    """the frame's registers as a flat dict of slot -> value (strings/ids), raising OutOfScope for inner frames"""
    for r in UNSUPPORTED_REGS:
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

# ----------------------------------------------------------------------------------------------- tracing the rule parser
PMETH = set(n for n in dir(Parser) if callable(getattr(Parser, n)) and not n.startswith("__"))

def trace(toks, choices=()):
    """run the rule parser and return the list of consumption events with before-state and register diffs"""
    p = Parser(list(toks), list(choices))
    events = []            # dicts: i (token index consumed), k (how many), cp, snap_before, np spans
    np_spans = []          # (i0, i1, ja, cls, depth_of_c)
    st = {"last_i": 0, "line": 0, "cp": None}
    def chain_of(frame):
        """the Parser-method frames on the stack, outermost first, as (method, current line)"""
        names = []; f = frame
        while f is not None:
            if f.f_code.co_name in PMETH and f.f_locals.get("self") is p: names.append((f.f_code.co_name, f.f_lineno))
            f = f.f_back
        return tuple(reversed(names))
    def stack_of(frame): return tuple(n for n, _ in chain_of(frame))
    def tracer(frame, event, arg):
        if frame.f_code.co_filename != R.__file__ and not frame.f_code.co_filename.endswith("parser_rules.py"): return tracer
        if event == "line":
            if p.i != st["last_i"]:
                events.append({"i": st["last_i"], "k": p.i - st["last_i"], "cp": (stack_of(frame), st["line"]), "snap": st["snap"],
                               "chain": chain_of(frame), "win_reads": set(reads["win"]), "reg_reads": set(reads["reg"])})
                reads["win"].clear(); reads["reg"].clear()
                st["last_i"] = p.i
            st["line"] = frame.f_lineno
            try: st["snap"] = snapshot(p.c) if p.c is p.main else None
            except OutOfScope as e: st["snap"] = e
        return tracer
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
        np_spans.append((i0, self.i, ja, cls, self.c is self.main))
        heads.append((i0, self.i, self.last_head))
        return ja, cls
    heads = []
    Parser.parse_np = np_wrapped
    st["snap"] = initial = snapshot(p.c)
    sys.settrace(tracer); err = None
    try:
        frame = p.parse()
    except ParseFail as e:
        err = e; frame = None
    finally:
        sys.settrace(None); Parser.parse_np = orig_np; Parser.peek = orig_peek
    p.fail_info = None; _last_parser[0] = p
    p.last_cp = events[-1]["cp"] if events else ("START",)
    p.events_ok = all(not isinstance(e["snap"], (OutOfScope, type(None))) for e in events)
    p.pend_at_end = None
    p.events, p.np_spans, p.heads = events, np_spans, heads
    for j in range(p.i):
        if j < len(p.toks) and "prep" in dict(readings_of(p.toks[j])): p.pend_at_end = p.toks[j]
    if err is not None:
        try: fs = snapshot(p.main)
        except OutOfScope as e2: fs = e2
        p.fail_info = (err, fs, set(reads["win"]), set(reads["reg"]), p.i)
        tb = err.__traceback__; p.fail_stack = []
        while tb is not None:
            if tb.tb_frame.f_code.co_name in PMETH and tb.tb_frame.f_locals.get("self") is p: p.fail_stack.append(tb.tb_frame.f_code.co_name)
            tb = tb.tb_next
        if events: events[0]["snap"] = initial
        raise err
    if events: events[-1]["win_reads"] |= reads["win"]; events[-1]["reg_reads"] |= reads["reg"]
    if events: events[0]["snap"] = initial          # writes made before the first consumption belong to the first step
    try: final = snapshot(frame)
    except OutOfScope as e: final = e                 # the finished frame holds a register the cell has no block for
    return events, final, np_spans, p

# ----------------------------------------------------------------------------------------------- predicates and induction
ENUM_REGS = ("modal", "want", "cop_kind", "degree", "obj_particle", "wh", "purpose", "causative", "subj_cls", "time", "adv1", "front", "cmp_kind", "ret")
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
    if upart == "-": expected |= {(f"{role}.{sub}", None, ("clear",)) for sub in NP_SUBS}; optional = {("obj_particle", None, ("clear",))}
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

def np_site(chain):
    """the return address of a noun phrase: (caller methods, line of the parse_np call) for the outermost parse_np frame"""
    for idx, (name, line) in enumerate(chain):
        if name == "parse_np":
            return (tuple(n for n, _ in chain[:idx]), chain[idx - 1][1] if idx > 0 else 0)
    return None

def site_id(site):
    return ">".join(site[0][1:] or ("parse",)) + ":" + str(site[1])

def np_span_info(events, snaps, np_spans, final_snap):
    """the outermost main-frame noun phrases: (first event, last event, role, ja, cls, site, old value of the role, where it moved)"""
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
        for ei in range(last, len(snaps) - 1):
            before, after = snaps[ei], snaps[ei + 1]
            for r in NP_ROLES:
                if after.get(r) == ja and before.get(r) != ja: role = r; break
            if role: break
        if role is None and last + 1 < len(snaps) and snaps[last + 1].get("verb") and "+" in snaps[last + 1]["verb"][0]:
            role, dropped = "obj", True                       # take a rest -> 休む: the use-verb absorbs the object (USE particle "-")
        if role is None: raise OutOfScope(f"noun phrase {ja!r} not written to a slot")
        site = np_site(events[first]["chain"])
        if site is None: raise OutOfScope("noun phrase without a parse_np frame")
        old = snaps[first].get(role); moved_to = None
        if old is not None and last + 1 < len(snaps):
            for r2 in NP_ROLES:
                if r2 != role and snaps[last + 1].get(r2) == old and snaps[first].get(r2) != old: moved_to = r2
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
            actions += [(f"{r}.{sub}", None, ("clear",)) for sub in NP_SUBS]; continue
        # a slot acquiring another slot's old string is a move (give him a book: obj -> iobj)
        src_role = next((r2 for r2 in NP_ROLES if r2 != r and before.get(r2) == v), None)
        if src_role:
            actions += [(f"{r}.{sub}", None, ("move", f"{src_role}.{sub}")) for sub in NP_SUBS]; continue
        if "time" in rs and v == rs["time"]: src = ("copy", "time")
        elif "loc" in rs and v == rs["loc"]: src = ("copy", "loc")
        elif "noun" in rs and v == rs["noun"][0]: src = ("copy", "noun")
        elif "pron" in rs and v == rs["pron"][0]: src = ("copy", "pron")
        else: src = ("const",)
        actions.append((f"{r}.noun", v, src)); actions.append((f"{r}.cls", "t" if r != "subj" else "h", ("const",)))
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
        if e["snap"] is None: raise OutOfScope("inner clause")
    snaps = [e["snap"] for e in events] + [final]
    rows, roles, prev, pend = transition_rows(toks, events, snaps, np_spans, final)
    P_end = frozenset(predicates(toks, len(toks), final, set(range(len(toks), len(toks) + WINDOW)), ALL_REGS, pend))
    rows.append((prev, P_end, frozenset(), (("DONE",), 0, ())))
    return rows, roles

def transition_rows(toks, events, snaps, np_spans, final):
    """the rows of the consumption events given (a whole parse, or the prefix of one up to the point where it stops being
    representable or is refused): -> (rows, roles used, the control state after the last event, the pending preposition)"""
    import np_module as NPM
    info = np_span_info(events, snaps, np_spans, final)
    in_span = {k: n for n, (first, last, *_) in enumerate(info) for k in range(first, last + 1)}
    rows = []; roles = set(); prev = ("START",); pend = None
    if not hasattr(hybrid_rows, "module_log"): hybrid_rows.module_log = collections.Counter()
    noreads = {"win_reads": set(), "reg_reads": set()}
    ei = 0
    while ei < len(events):
        e = events[ei]
        if ei in in_span:
            first, last, role, ja, cls, site, old, moved_to, dropped = info[in_span[ei]]
            roles.add(role); sid = site_id(site)
            # entry
            acts = [("ret", sid, ("const",))]
            if old is not None:
                if moved_to: acts += [(f"{moved_to}.{sub}", None, ("move", f"{role}.{sub}")) for sub in NP_SUBS]
                acts += [(f"{role}.{sub}", None, ("clear",)) for sub in NP_SUBS]
            e1 = events[first]; e2 = events[first + 1] if first + 1 < len(events) else noreads
            P_read = frozenset(predicates(toks, e1["i"], snaps[first], set(e1["win_reads"]) | set(e2["win_reads"]), set(e1["reg_reads"]) | set(e2["reg_reads"]), pend))
            P_all = frozenset(predicates(toks, e1["i"], snaps[first], set(range(e1["i"], e1["i"] + WINDOW)), ALL_REGS, pend))
            rows.append((prev, P_all, P_read, (("NP0", role), 0, norm_actions(acts))))
            # exit: the module's slots as it leaves them (its own run), then the post-NP effects of the rule parser
            el = events[last]; i1 = el["i"] + el["k"]
            regs = {"verb": snaps[first].get("verb"), "cop_kind": snaps[first].get("cop_kind"), "subj": snaps[first].get("subj") if role != "subj" else None}
            mod = NPM.run_module(role, toks, e1["i"], regs, pend)
            heads = {}
            if mod[0] == "RET":
                slots = mod[2]; heads = {role: slots.get(f"{role}.noun2") or slots.get(f"{role}.noun")}
                hybrid_rows.module_log["module agrees" if (NPM.np_string(slots, role) == ja and mod[1] == i1) else "module differs"] += 1
                if NPM.np_string(slots, role) != ja or mod[1] != i1: hybrid_rows.module_log[("differs", " ".join(toks[e1["i"]:i1]), ja, NPM.np_string(slots, role))] += 1
            else: hybrid_rows.module_log["module refuses: " + mod[1].split(" at ")[0][:30]] += 1
            snap_ret = dict(snaps[last]); snap_ret[role] = ja; snap_ret["ret"] = sid
            if role == "subj": snap_ret["subj_cls"] = cls
            if moved_to: snap_ret[moved_to] = old
            post = register_actions(snaps[last], snaps[last + 1], None, allow_copy=False)
            post += np_register_actions(snap_ret, snaps[last + 1], toks[i1] if i1 < len(toks) else None, {role})
            if dropped: post += [(f"{role}.{sub}", None, ("clear",)) for sub in NP_SUBS]
            post = fold_use(post, role, heads.get(role), snaps[first].get("verb"))
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
        before, after = snaps[ei], snaps[ei + 1]; tok = toks[e["i"]]
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
        if p.forks:                                         # this is the default reading; from its first fork on, the path is the choice's, not the window's
            fpos = p.forks[0][0]
            cut = min(cut, next((k for k, e in enumerate(events) if e["i"] + e["k"] > fpos), cut))
            cut = np_entry_event(events, spans, cut) if cut < len(events) else cut
            if cut is None: return []
        final_snap = events[cut]["snap"] if cut < len(events) else fs
        if not isinstance(final_snap, dict): return []
        try: rows, roles, prev, pend = prefix_rows(toks, events, spans, cut, final_snap)
        except OutOfScope: return []
        if cut < len(events) or "parse_np" in p.fail_stack: return rows
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
    try: rows, roles, prev, pend = prefix_rows(toks, events, spans, cut, events[cut]["snap"])
    except OutOfScope: return []
    e = events[cut]
    nxt_e = events[cut + 1] if cut + 1 < len(events) else {"win_reads": set(), "reg_reads": set()}
    P_all = predicates(toks, e["i"], e["snap"], set(range(e["i"], e["i"] + WINDOW)), ALL_REGS, pend)
    P_read = predicates(toks, e["i"], e["snap"], set(e["win_reads"]) | set(nxt_e["win_reads"]), set(e["reg_reads"]) | set(nxt_e["reg_reads"]), pend)
    return rows + [(prev, frozenset(P_all), frozenset(P_read), (FAIL, 0, ()))]

def module_rules(roles):
    """np_module's rules and derived predicates for the roles in use, in the compiler's rule format"""
    import np_module as NPM
    rules = []; derived = {}
    for role in sorted(roles):
        D = NPM.derived(role)
        for name, conjs in D.items(): derived[f"{name}@{role}"] = conjs
        def lit(l):
            neg = l.startswith("¬"); q = l[1:] if neg else l
            if q.startswith("d."): q = f"{q}@{role}"
            return (q, not neg)
        for state, lits, (nxt, k, acts) in NPM.rules_for(role):
            if nxt == "RET": tgt = ("RET", role)
            elif nxt.startswith("FAIL:"): tgt = FAIL
            else: tgt = (nxt, role)
            rules.append(((state, role), tuple(lit(l) for l in lits), (tgt, k, norm_actions(acts))))
    return rules, derived

def eval_rule_lit(q, truth, P, derived):
    if q in derived:
        v = any(all(eval_rule_lit(l[1:] if l.startswith("¬") else l, not l.startswith("¬"), P, derived) for l in conj) for conj in derived[q])
    else: v = q in P
    return v == truth

def run_rules(rules, derived, toks, bound_per_token=12):
    """interpret the hybrid rule set (induced outer rules + module rules) on a sentence — the check of the rules before the matrix"""
    by_prev = collections.defaultdict(list)
    for prev, lits, out in rules: by_prev[prev].append((lits, out))
    state = ("START",); i = 0; frame = {}; pend = None
    for _ in range(bound_per_token * (len(toks) + 2)):
        if state == ("DONE",): return frame
        if state == FAIL: raise ParseFail(f"RULES: FAIL at token {i}", i)
        P = predicates(toks, i, frame_to_snapshot(frame), set(range(i, i + WINDOW)), ALL_REGS | {"ret"}, pend, heads_of(frame)) | slot_predicates(frame)
        P.discard("agree.bad")                       # the rules read it as the derived predicate, evaluated from the slot and the window
        hits = [out for lits, out in by_prev.get(state, []) if all(eval_rule_lit(q, t, P, derived) for q, t in lits)]
        if len(hits) != 1: raise ParseFail(f"RULES: {len(hits)} rules fire at {state} token {i}", i)
        cp, k, actions = hits[0]
        if cp == ("CONFLICT",) or (isinstance(cp, tuple) and cp and cp[0] == "CONFLICT"): raise ParseFail("RULES: conflict", i)
        apply_actions(frame, actions, toks, i)
        for j in range(i, i + k):
            if j < len(toks) and "prep" in dict(readings_of(toks[j])): pend = toks[j]
        i += k; state = cp
        if i > len(toks): raise ParseFail("RULES: ran past the end", i)
    raise ParseFail("RULES: did not finish", i)

def build_hybrid(sentences, verbose=False):
    """rows from the corpus (outer transitions + NP hand-overs + FAIL rows), induced trees, the module's rules -> compiled cell"""
    rows = []; roles = set(); scope = collections.Counter()
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
    comp = compile_matrices(trees, extra_rules=mrules, derived=derived)
    if verbose: print("hybrid build:", dict(scope), "| data rows", len(rows), "| module rules", len(mrules), "| roles", sorted(roles))
    comp["scope"] = scope; comp["rows"] = rows
    return comp

def induce(rows):
    """per previous control point, a decision tree over predicates whose leaves are outcomes; returns {prev: tree}.
    A leaf also carries the guard the rule parser itself applied: the class predicates of the token in slot 0 that every row
    reaching the leaf read as true (an `end` transition keeps w0.class=end even when no training row ever had another token
    there), so that an induced rule never consumes a kind of token the parser was not seen to consume."""
    by_prev = collections.defaultdict(list); read_by_prev = collections.defaultdict(set)
    for prev, P_all, P_read, outcome in rows: by_prev[prev].append((P_all, P_read, outcome)); read_by_prev[prev] |= P_read
    def guard(items):
        common = set.intersection(*(set(R_) for _, R_, _ in items)) if items else set()
        return tuple(sorted(q for q in common if q.startswith("w0.class=")))
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

def tree_rules(tree, path=()):
    """flatten a tree into (literals, outcome) rules; literals are (predicate, truth), the leaf's guard appended"""
    if tree[0] == "leaf":
        seen = {q for q, t in path}
        return [(path + tuple((q, True) for q in tree[2] if q not in seen), tree[1])]
    if tree[0] == "conflict": return [(path, ("CONFLICT", tree[1], tree[2]))]
    _, q, yes, no = tree
    return tree_rules(yes, path + ((q, True),)) + tree_rules(no, path + ((q, False),))

def frame_to_snapshot(frame):
    """the slot dict as the snapshot() view (so predicates read it the same way)"""
    s = {f: bool(frame.get(f)) for f in FLAGS}
    for r in ("modal", "want", "degree", "obj_particle", "wh", "time", "purpose", "causative", "verb", "adv1", "adv2", "front", "cop_kind", "cop_adj", "cop_adv", "cop_part", "pp1_part", "pp2_part", "pp3_part", "cmp_kind", "ret"):
        s[r] = frame.get(r)
    for role in NP_ROLES:
        s[role] = np_string(frame, role)
    s["subj_cls"] = frame.get("subj.cls") if s["subj"] else None
    s["subj_num"] = (frame.get("subj.number") if frame.get("subj.number") != "none" else None) if s["subj"] else None
    return s

def np_string(frame, role):
    parts = [frame.get(f"{role}.{sub}") for sub in ("det", "poss", "num", "adj1", "adj2", "noun")]
    if frame.get(f"{role}.noun") is None: return None
    s = "".join(p for p in parts if p)
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
    for slot, v, src in actions:
        if src[0] == "copy" and src[1] not in ("ncls", "nnum") and not rs[{"num_time": "num", "adj_pair": "adj", "adj_mono": "adj", "adj_caus": "adj", "pron_poss": "pron", "adv_front": "adv"}.get(src[1], src[1])]:
            raise ParseFail(f"TABLE: nothing to copy into {slot} from {src[1]} at {tok!r}", i)       # the machine's crash: the rule fired on a token without that reading
    old = dict(frame)                                   # moves and clears read the state before this step, as the matrix does
    for slot, v, src in actions:
        if src[0] == "use":                                 # the feedback lookup: slot is the role whose head noun carries the verb
            head = old.get(f"{slot}.noun2") or old.get(f"{slot}.noun"); verb = old.get("verb")
            hit = USE_TABLE.get((head, verb[0])) if head and verb else None
            if hit is None: raise ParseFail(f"TABLE: no use pair for {head!r} and {verb and verb[0]!r} at {tok!r}", i)
            new_verb, upart, stative = hit
            frame["verb"] = new_verb
            if upart == "-":
                frame.pop("obj_particle", None)
                for sub in NP_SUBS: frame.pop(f"{slot}.{sub}", None)
            elif slot.startswith("pp"): frame[f"{slot}_part"] = (upart, upart)
            else: frame["obj_particle"] = upart
            if stative: frame["prog"] = True
            continue
        if src[0] == "clear": frame.pop(slot, None)
        elif src[0] == "const": frame[slot] = v
        elif src[0] in ("move", "move_time"):              # the gate clears the slot; the write lands only if the source slot is filled
            if old.get(src[1]) is not None: frame[slot] = old.get(src[1]) + ("時" if src[0] == "move_time" else "")
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
            elif cls == "time": frame[slot] = rs["time"][-1]
            elif cls == "loc": frame[slot] = rs["loc"][-1]
            elif cls == "wh": frame[slot] = rs["wh"][-1]
            elif cls == "modal": frame[slot] = rs["modal"][-1][0]
            elif cls == "nnum": frame[slot] = R.number_of(tok) or "none"
            elif cls == "verb":
                lem, form, ja, conj = rs["verb"][0]; frame[slot] = (lem, ja, conj)

def frame_to_clause(frame):
    c = Clause()
    for f in FLAGS: setattr(c, f, bool(frame.get(f)))
    for r in ("modal", "want", "obj_particle", "wh", "purpose", "causative", "verb"): setattr(c, r, frame.get(r))
    c.degree = frame.get("degree") or ""; c.time = frame.get("time") or ""; c.adv = frame.get("adv1") or ""; c.front = frame.get("front") or ""
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

COPY_SOURCES = ("noun", "ncls", "pron", "pron_poss", "det", "num", "num_time", "adj", "adj_pair", "adj_mono", "adj_caus", "adv", "adv_front", "time", "wh", "modal", "verb", "nnum", "loc")

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
            out.append(("adj_mono", v[0] + ("な" if v[1] == "na" and not v[0].endswith("な") else "") + "もの"))
            out.append(("adj_caus", ("make+" + tok, R.adj_causative(v[0], v[1]), "i")))
        elif c == "adv": out.append(("adv", v)); out.append(("adv_front", v + "、"))
        elif c == "loc": out.append(("loc", v))
        elif c == "time": out.append(("time", v))
        elif c == "wh": out.append(("wh", v))
        elif c == "modal": out.append(("modal", v[0]))
        elif c == "verb": out.append(("verb", (v[0], v[2], v[3])))
    if any(c in ("noun", "pron") for c, v in readings_of(tok)): out.append(("nnum", R.number_of(tok) or "none"))   # the head's grammatical number
    # the rule parser takes the LAST reading of a class (dict() semantics) for noun/adj/adv/time/wh, the FIRST for pron/det/num/verb
    last = {}; first = {}
    for s, v in out:
        last[s] = v; first.setdefault(s, v)
    pick = {s: (last[s] if s in ("noun", "ncls", "adj", "adj_pair", "adj_mono", "adj_caus", "adv", "adv_front", "loc", "time", "wh") else first[s]) for s in last}
    return pick

def compile_matrices(trees, vocab_hint=(), extra_rules=(), derived=None):
    """trees (per previous control point) + hand-written rules -> the cell: five sparse ReLU layers over a one-hot state and
    the lexical input, then the new state.  `derived` maps a predicate name to a DNF over base literals (np_module.derived)."""
    derived = dict(derived or {})
    rules = []   # (prev, literals, (cp, k, actions))
    for prev, t in trees.items():
        for lits, out in tree_rules(t):
            if out[0] == "CONFLICT": continue
            rules.append((prev, lits, out))
    rules += list(extra_rules)
    # ---- vocabularies
    slot_vocab = collections.defaultdict(set)
    for _, _, (cp, k, actions) in rules:
        for slot, v, src in actions:
            if src[0] == "const" and v is not None: slot_vocab[slot].add(v)
    # copy sources: every value the lexicon can produce for that source
    src_vocab = collections.defaultdict(set)
    for tok in list(R.LEX) + list(R.PRONOUNS) + list(R.DETS) + list(R.NUMBERS) + list(R.WH) + list(R.MODALS) + list(R.VERB_FORMS):
        for s, v in source_values(tok).items(): src_vocab[s].add(v)
    for s in src_vocab: src_vocab[s] = sorted(src_vocab[s], key=str)
    copy_pairs = set(); move_pairs = set()
    MOVE_FN = {"move": lambda v: v, "move_time": lambda v: v + "時"}
    for _, _, (cp, k, actions) in rules:
        for slot, v, src in actions:
            if src[0] == "copy": copy_pairs.add((slot, src[1])); slot_vocab[slot] |= set(src_vocab[src[1]])
            if src[0] in MOVE_FN: move_pairs.add((slot, src[1], src[0]))
    for _ in range(3):
        for slot, src_slot, fn in list(move_pairs): slot_vocab[slot] |= {MOVE_FN[fn](v) for v in slot_vocab[src_slot]}
    # the USE lookup: every value a pair of this cell can write (verb after the feedback, particle, prog)
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
    S.add(("shift",))
    def slot_units(slot): return [S[("slot", slot, v)] for v in slot_vocab.get(slot, [])]
    # ---- input index: window predicates, copy sources, prep-surface of slot 0
    X = Index()
    def is_window(q): return q.startswith("w") and q[1].isdigit() or q.startswith("hit=")
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
        """-> list of (ref, 1.0) for a base literal, or None if the predicate can never be true in this cell"""
        if is_window(q): return [(("X", X[("pred", q)]), 1.0)] if ("pred", q) in X else None
        if q.startswith("slot.") and "=" in q:
            slot, v = q[5:].split("=", 1)
            vals = [u for u in slot_vocab.get(slot, []) if str(u) == v]
            return [(("S", S[("slot", slot, u)]), 1.0) for u in vals] or None
        if q.startswith("slot."): return [(("S", u), 1.0) for u in slot_units(q[5:])] or None
        if q == "reg.verb": return [(("S", u), 1.0) for u in slot_units("verb")] or None
        if q == "reg.cop_kind": return [(("S", u), 1.0) for u in slot_units("cop_kind")] or None
        if q == "reg.subj": return [(("S", u), 1.0) for u in slot_units("subj.noun")] or None
        if q == "subj.sore": return [(("S", S[("slot", "subj.noun", "それ")]), 1.0)] if ("slot", "subj.noun", "それ") in S else None
        if q.startswith("pend_prep="): return [(("S", S[("pend", q[10:])]), 1.0)] if ("pend", q[10:]) in S else None
        if q.startswith("reg.ret="): return [(("S", S[("slot", "ret", q[8:])]), 1.0)] if ("slot", "ret", q[8:]) in S else None
        return None
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
    conj_units = {}; derived_terms = collections.defaultdict(list); dead_conj = 0
    for name, conjs in derived.items():
        for conj in conjs:
            key = tuple(sorted(conj))
            if key not in conj_units:
                edges = []; npos = 0; ok = True
                for l in conj:
                    neg = l.startswith("¬"); q = l[1:] if neg else l
                    es = base_edges(q)
                    if es is None:
                        if not neg: ok = False; break            # a positive literal the cell cannot see: the conjunction is never true
                        continue                                  # a negative literal on an unseen predicate is vacuously true
                    for ref, w in es: edges.append((ref, -w if neg else w))
                    if not neg: npos += 1
                if not ok: conj_units[key] = None; dead_conj += 1
                else:
                    u = L1.add(("conj", key))
                    for ref, w in edges: W1.append((u, ref, w))
                    b1[u] = -(npos - 1) if npos else 0.0
                    conj_units[key] = u
            if conj_units[key] is not None: derived_terms[name].append(conj_units[key])
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
    # ---- layer 2: pp.count, NOT-derived (1 - sum of the conjunctions), use.<role> (OR of pairs)
    L2 = Index(); W2 = []; b2 = {}
    u = L2.add(("pred", "pp.count=0")); b2[u] = 1.0
    for k in range(3): W2.append((u, ("L1", f[k]), -1.0))
    for n in (1, 2):
        u = L2.add(("pred", f"pp.count={n}")); W2.append((u, ("L1", f[n-1]), 1.0)); W2.append((u, ("L1", f[n]), -1.0))
    u = L2.add(("pred", "pp.count=3")); W2.append((u, ("L1", f[2]), 1.0))
    for name in derived:
        u = L2.add(("nor", name)); b2[u] = 1.0
        for t in derived_terms.get(name, []): W2.append((u, ("L1", t), -1.0))
    for role, units in use_pairs.items():
        u = L2.add(("pred", f"use.{role}"))
        for t in units: W2.append((u, ("L1", t), 1.0))
    # ---- layer 3: rules = AND(control, literals)
    L3 = Index(); W3 = []; b3 = {}
    def pred_ref(q):
        if ("pred", q) in L1: return ("L1", L1[("pred", q)])
        if ("pred", q) in L2: return ("L2", L2[("pred", q)])
        if ("pred", q) in X: return ("X", X[("pred", q)])
        return None
    unresolved = collections.Counter()
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        u = L3.add(("rule", ri)); npos = 1
        W3.append((u, ("S", S[("ctrl", prev)]), 1.0))
        for q, truth in lits:
            if q in derived:                                   # through its NOT unit: positive wants NOT = 0, negative wants NOT = 1
                nu = ("L2", L2[("nor", q)])
                if truth: W3.append((u, nu, -1.0))
                else: W3.append((u, nu, 1.0)); npos += 1
                continue
            ref = pred_ref(q)
            if ref is not None: es = [(ref, 1.0)]
            else: es = base_edges(q)
            if es is None:
                if truth: unresolved[q] += 1; W3.append((u, ("X", X[("one",)]), -5.0))      # a positive literal the cell cannot see: the rule can never fire
                continue                                                                # a negative literal on an unseen predicate is vacuously true
            for ref, w in es: W3.append((u, ref, w if truth else -w))
            if truth: npos += 1
        b3[u] = -(npos - 1)
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
    # gates per slot (any write/clear) and per (slot, source) copy / (slot, from, fn) move
    gate_slot = {}; gate_copy = {}; gate_move = {}; const_w = collections.defaultdict(list)
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        for slot, v, src in actions:
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
    # ---- layer 6: the new state = next control + shift + keep + write
    W6 = []
    for i, name in enumerate(L4.names):
        if name[0] == "next": W6.append((S[name[1]], ("L4", i), 1.0))
        if name[0] == "shift": W6.append((S[("shift",)], ("L4", i), 1.0))
    for i, name in enumerate(L5.names):
        if name[0] in ("keep", "write", "writec", "writem"):
            key = ("pend", name[2]) if name[1] == "pend" else ("slot", name[1], name[-1])
            W6.append((S[key], ("L5", i), 1.0))
        if name[0] == "writeu": W6.append((S[("slot", name[1], name[2])], ("L5", i), 1.0))
        if name[0] in ("nosrc", "nouse"): W6.append((S[("ctrl", FAIL)], ("L5", i), 1.0))
    layers = [(L1, W1, b1), (L2, W2, b2), (L3, W3, b3), (L4, W4, b4), (L5, W5, b5)]
    return {"S": S, "X": X, "layers": layers, "W6": W6, "rules": rules, "slot_vocab": slot_vocab, "src_vocab": src_vocab,
            "unresolved": unresolved, "derived": derived, "dead_conj": dead_conj}

class MatrixParser:
    """the compiled cell run as sparse matrix products with ReLU"""
    name = "matrix"
    def __init__(self, compiled):
        self.c = compiled; S, X = compiled["S"], compiled["X"]
        self.nS, self.nX = len(S), len(X)
        self.offsets = {"S": 0, "X": self.nS}; width = self.nS + self.nX
        self.mats = []
        for (L, W, b) in compiled["layers"]:
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
        for (si, ref, w) in compiled["W6"]:
            rows_.append(si); cols_.append(self.offsets[ref[0]] + ref[1]); vals_.append(w)
        self.W6 = sp.csr_matrix((vals_, (rows_, cols_)), shape=(self.nS, width))
        self.width = width
        self.shift_unit = S[("shift",)] if ("shift",) in S else None
    def nnz(self): return sum(M.nnz for M, _ in self.mats) + self.W6.nnz
    def lexical_input(self, toks, i):
        """the lexical layer's output for the window at i: class / surface / extra predicates of the four slots"""
        x = np.zeros(self.nX); X = self.c["X"]
        allw = window_features(toks, i)
        for k in range(WINDOW):
            surface, classes, extra = allw[k]
            for cq in [f"w{k}.class={c}" for c in classes] + ([f"w{k}.surf={surface}"] if surface != "*" else []):
                if ("pred", cq) in X: x[X[("pred", cq)]] = 1
            for e in extra:
                qs = [f"w{k}.vc={c}" for c in e[1]] if e[0] == "vc" else [f"w{k}.{e[0]}={e[1]}"]
                for q in qs:
                    if ("pred", q) in X: x[X[("pred", q)]] = 1
        return x
    def step(self, s, x, toks, i, verb_value):
        X = self.c["X"]
        for h in table_hits(toks, i, verb_value):
            q = "hit=" + ",".join(map(str, h))
            if ("pred", q) in X: x[X[("pred", q)]] = 1
        tok = toks[i] if i < len(toks) else None
        for sname, v in source_values(tok).items():
            if ("src", sname, v) in X: x[X[("src", sname, v)]] = 1
        if tok in R.PREPS: x[X[("prep0", tok)]] = 1
        x[X[("one",)]] = 1
        acts = np.concatenate([s, x])
        for M, bias in self.mats:
            h = np.maximum(0.0, M @ acts + bias); acts = np.concatenate([acts, h])
        return np.maximum(0.0, self.W6 @ acts)
    def read_state(self, s):
        S = self.c["S"]; frame = {}; ctrl = None; n_ctrl = 0; failed = False
        for idx in np.nonzero(s > 0.5)[0]:
            name = S.names[idx]
            if name[0] == "ctrl": ctrl = name[1]; n_ctrl += 1; failed |= name[1] == FAIL
            elif name[0] == "slot": frame[name[1]] = name[2]
        if failed: return FAIL, frame
        return (ctrl if n_ctrl == 1 else ("MULTI", n_ctrl) if n_ctrl else None), frame
    def parse(self, toks, bound_per_token=12):
        """one matrix step per parser transition; the `shift` unit tells the window when to advance (k=0 transitions do not)"""
        S = self.c["S"]; s = np.zeros(self.nS); s[S[("ctrl", ("START",))]] = 1.0; i = 0
        for step in range(bound_per_token * (len(toks) + 2)):
            ctrl, frame = self.read_state(s)
            if ctrl == ("DONE",): return frame_to_clause(frame)
            if ctrl == FAIL: raise ParseFail(f"MATRIX: FAIL at token {i} ({toks[i] if i < len(toks) else 'END'!r})", i)
            if ctrl is None: raise ParseFail(f"MATRIX: no rule fired at token {i} ({toks[i] if i < len(toks) else 'END'!r})", i)
            if ctrl[0] == "MULTI": raise ParseFail(f"MATRIX: {ctrl[1]} rules fired at token {i}", i)
            if i > len(toks): raise ParseFail("MATRIX: ran past the end", i)
            verb_value = frame.get("verb")
            x = self.lexical_input(toks, i)
            s = self.step(s, x, toks, i, verb_value)
            if self.shift_unit is None or s[self.shift_unit] > 0.5: i += 1
        raise ParseFail("MATRIX: did not reach DONE", i)
