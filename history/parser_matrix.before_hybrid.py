"""The parser layer, compiled to sparse matrices.

MatrixParser implements the same interface as the rule-based parser (parser.ParserInterface): a left-to-right
machine that reads the lexical layer's output for a 4-token window and updates a one-hot state vector with
three sparse ReLU layers per token; the frame is read out of the state at the end of the sentence.

The compile is by extraction, not by hand: the rule parser is traced on the corpus, each token consumption
becomes one row of a transition table keyed on (control point, window features, frame features), and the table
is compiled into AND units.  Register writes are factored into three kinds the matrix can express:
    copy   frame slot <- a reading of the window's current token (gated copy, one edge per vocabulary item)
    const  frame slot <- a fixed value chosen by the transition
    move   frame slot <- another frame slot (give him a book: obj -> iobj)
Sentences whose parse needs anything else (an inner clause, a complex noun phrase, a second reading) are out of
this implementation's scope and are reported as such; inside the scope it must agree with the rule parser exactly.
"""
import sys, collections, itertools
import numpy as np
import scipy.sparse as sp
import rules_ja as R
from parser_rules import Parser, Clause, ParseFail, Overflow, parse_bounded

WINDOW = 4
NP_ROLES = ("subj", "obj", "iobj", "pp1", "pp2", "pp3", "cop")
NP_SUB = ("det", "poss", "num", "adj1", "adj2", "noun", "cls")
FLAGS = ("past", "neg", "prog", "perf", "question", "imp", "please", "lets", "there", "tag", "coord", "sub", "neg_inner")
ENUMS = ("modal", "want", "degree", "obj_particle", "wh", "time", "adv1", "adv2", "cop_kind", "cop_adj", "cop_adv", "cop_part",
         "pp1_part", "pp2_part", "pp3_part", "verb", "purpose", "causative")
UNSUPPORTED_REGS = ("infinitive", "control", "embedded", "purpose_clause", "pre_clause", "manner", "so_that", "howabout",
                    "exclaim", "fixed_out", "compare", "relative", "suffix")
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
    return tuple(sorted(set(hits)))

# ----------------------------------------------------------------------------------------------- frame as slots
def np_components(toks, i0, i1, ja_out):
    """decompose the NP the rule parser built from toks[i0:i1] into sub-slot values, or raise OutOfScope"""
    comps = {}; order = []
    j = i0
    while j < i1:
        t = toks[j]; rs = dict(readings_of(t))
        if "det" in rs and "det" not in comps and "noun" not in comps and not (j > i0 and "noun" in rs):
            comps["det"] = rs["det"]; order.append(("det", j)); j += 1; continue
        if "pron" in rs and "p" in rs["pron"][1] and "noun" not in comps and j + 1 < i1:
            comps["poss"] = rs["pron"][0] if rs["pron"][1] == "p" else rs["pron"][0] + "の"; order.append(("poss", j)); j += 1; continue
        if "pron" in rs and j == i1 - 1 and "noun" not in comps and not comps:
            comps["noun"] = rs["pron"][0]; comps["cls"] = "h"; order.append(("noun", j)); j += 1; continue
        if "num" in rs and "num" not in comps and "noun" not in comps and j + 1 < i1:
            comps["num"] = rs["num"]; order.append(("num", j)); j += 1; continue
        if "adj" in rs and j + 1 < i1 and "noun" not in comps:
            ja, kind = rs["adj"]; v = ja + ("な" if kind == "na" and not ja.endswith("な") else "")
            slot = "adj1" if "adj1" not in comps else ("adj2" if "adj2" not in comps else None)
            if slot is None: raise OutOfScope("three adjectives")
            comps[slot] = v; order.append((slot, j)); j += 1; continue
        if "adv" in rs and j + 1 < i1 and "adj" in dict(readings_of(toks[j + 1])):
            slot = "adj1" if "adj1" not in comps else "adj2"
            comps[slot] = rs["adv"]; order.append((slot, j)); j += 1; continue
        if "noun" in rs and "noun" not in comps and j == i1 - 1:
            comps["noun"], comps["cls"] = rs["noun"]; order.append(("noun", j)); j += 1; continue
        if "num" in rs and j == i1 - 1 and not comps:
            comps["noun"] = rs["num"]; comps["cls"] = "t"; order.append(("noun", j)); j += 1; continue
        raise OutOfScope(f"noun phrase shape at {t!r}")
    s = "".join(comps.get(k, "") for k in ("det", "poss", "num", "adj1", "adj2", "noun"))
    if s != ja_out and not (ja_out == comps.get("noun", "") + "時" and set(comps) <= {"noun", "cls"}):
        raise OutOfScope(f"noun phrase composed as {ja_out!r}, slots give {s!r}")
    if ja_out.endswith("時") and s != ja_out: comps["noun"] = ja_out       # bare number as a clock time
    return comps, order

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
    s["adv1"] = c.adv or None; s["adv2"] = None
    s["subj"] = c.subj; s["subj_cls"] = c.subj_cls if c.subj else None
    s["obj"] = c.obj; s["iobj"] = c.iobj
    for k in range(3):
        if k < len(c.pp): s[f"pp{k+1}_part"], s[f"pp{k+1}"] = c.pp[k][0], c.pp[k][1]
        else: s[f"pp{k+1}_part"] = s[f"pp{k+1}"] = None
    if len(c.pp) > 3: raise OutOfScope("four PPs")
    kind = c.copula[0] if c.copula else None
    s["cop_kind"] = kind; s["cop_adj"] = c.copula[1] if kind == "adj" else None; s["cop_adv"] = c.copula[1] if kind == "adv" else None
    s["cop_part"] = c.copula[1] if kind == "pp" else None
    s["cop"] = c.copula[1] if kind == "np" else (c.copula[2] if kind == "pp" else None)
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
    def stack_of(frame):
        names = []; f = frame
        while f is not None:
            if f.f_code.co_name in PMETH and f.f_locals.get("self") is p: names.append(f.f_code.co_name)
            f = f.f_back
        return tuple(reversed(names))
    def tracer(frame, event, arg):
        if frame.f_code.co_filename != R.__file__ and not frame.f_code.co_filename.endswith("parser_rules.py"): return tracer
        if event == "line":
            if p.i != st["last_i"]:
                events.append({"i": st["last_i"], "k": p.i - st["last_i"], "cp": (stack_of(frame), st["line"]), "snap": st["snap"],
                               "win_reads": set(reads["win"]), "reg_reads": set(reads["reg"])})
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
    def np_wrapped(self):
        i0 = self.i
        ja, cls = orig_np(self)
        np_spans.append((i0, self.i, ja, cls, self.c is self.main))
        return ja, cls
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
    for j in range(p.i):
        if j < len(p.toks) and "prep" in dict(readings_of(p.toks[j])): p.pend_at_end = p.toks[j]
    if err is not None:
        try: fs = snapshot(p.main)
        except OutOfScope as e2: fs = e2
        p.fail_info = (err, fs, set(reads["win"]), set(reads["reg"]), p.i)
        if events: events[0]["snap"] = initial
        raise err
    if events: events[-1]["win_reads"] |= reads["win"]; events[-1]["reg_reads"] |= reads["reg"]
    if events: events[0]["snap"] = initial          # writes made before the first consumption belong to the first step
    final = snapshot(frame)
    return events, final, np_spans, p

def event_rows(toks, choices=()):
    """turn a trace into transition rows: (cp, features) -> (next cp, k, actions)"""
    events, final, np_spans, p = trace(toks, choices)
    for e in events:
        if isinstance(e["snap"], OutOfScope): raise e["snap"]
        if e["snap"] is None: raise OutOfScope("inner clause")
    snaps = [e["snap"] for e in events] + [final]
    # NP sub-slot writes, assigned to the event that consumed each component token
    np_writes = collections.defaultdict(list)     # event index -> [(slot, value, source)]
    np_role_of_span = {}
    for (i0, i1, ja, cls, main) in np_spans:
        if not main: raise OutOfScope("inner clause NP")
        comps, order = np_components(toks, i0, i1, ja)
        # which role received it: the first register that changes to this string after the span
        end_ev = max(ei for ei, e in enumerate(events) if e["i"] < i1)
        role = None
        for ei in range(end_ev, len(events)):
            before, after = snaps[ei], snaps[ei + 1]
            for r in ("subj", "obj", "iobj", "pp1", "pp2", "pp3", "cop"):
                if after.get(r) == ja and before.get(r) != ja: role = (r, ei); break
            if role: break
        if role is None: raise OutOfScope(f"noun phrase {ja!r} not written to a slot")
        r, ei = role
        np_role_of_span[(i0, i1)] = r
        for slot, j in order:
            ev_idx = [k for k, e in enumerate(events) if e["i"] <= j < e["i"] + e["k"]][0]
            src = ("copy", {"det": "det", "poss": "pron_poss", "num": "num", "adj1": "adj", "adj2": "adj", "noun": "noun"}[slot])
            if slot == "noun" and dict(readings_of(toks[j])).get("pron") and comps["noun"] == dict(readings_of(toks[j]))["pron"][0]: src = ("copy", "pron")
            if slot in ("adj1", "adj2") and "adv" in dict(readings_of(toks[j])) and "adj" not in dict(readings_of(toks[j])): src = ("copy", "adv")
            if slot == "noun" and "num" in dict(readings_of(toks[j])) and "noun" not in dict(readings_of(toks[j])): src = ("copy", "num")
            np_writes[ev_idx].append((f"{r}.{slot}", comps[slot], src))
        noun_ev = [k for k, e in enumerate(events) if e["i"] <= order[-1][1] < e["i"] + e["k"]][0]
        np_writes[noun_ev].append((f"{r}.cls", comps.get("cls", "t"), ("copy", "ncls") if comps.get("cls") != "h" or not dict(readings_of(toks[order[-1][1]])).get("pron") else ("const",)))
        if comps.get("noun", "").endswith("時") and "num" in dict(readings_of(toks[order[-1][1]])):
            np_writes[noun_ev] = [(s, v, ("copy", "num_time") if s == f"{r}.noun" else src) for s, v, src in np_writes[noun_ev]]
    rows = []
    for ei, e in enumerate(events):
        before, after = snaps[ei], snaps[ei + 1]
        actions = list(np_writes.get(ei, []))
        tok = toks[e["i"]]; rs = dict(readings_of(tok))
        for slot in before:
            if slot in ("subj", "obj", "iobj", "pp1", "pp2", "pp3", "cop", "subj_cls"): continue        # NP strings handled as sub-slots
            if before[slot] == after[slot]: continue
            v = after[slot]
            if v in (None, False, "", 0):
                actions.append((slot, None, ("clear",))); continue
            # source: a reading of the consumed token, a frame slot, or a constant
            src = ("const",)
            if slot == "verb" and "verb" in rs and v[0] == rs["verb"][0] and v[1] == rs["verb"][2] and v[2] == rs["verb"][3]: src = ("copy", "verb")
            elif slot in ("adv1",) and "adv" in rs and v == rs["adv"] and before[slot] in (None, ""): src = ("copy", "adv")
            elif slot == "time" and "time" in rs and v == rs["time"]: src = ("copy", "time")
            elif slot == "wh" and "wh" in rs and v == rs["wh"]: src = ("copy", "wh")
            elif slot == "cop_adj" and "adj" in rs and v == rs["adj"]: src = ("copy", "adj_pair")
            elif slot == "cop_adv" and "adv" in rs and v == rs["adv"]: src = ("copy", "adv")
            elif slot == "modal" and "modal" in rs and v == rs["modal"][0]: src = ("copy", "modal")
            actions.append((slot, v, src))
        # NP-string registers written without parse_np (go swimming -> obj 水泳; than today -> pp 今日): copy from the token's reading or a constant
        spans_here = {np_role_of_span[k] for k in np_role_of_span if k[1] - 1 >= e["i"] and k[0] <= e["i"] + e["k"] - 1}
        for r in ("subj", "obj", "iobj", "pp1", "pp2", "pp3", "cop"):
            if after.get(r) and after.get(r) != before.get(r) and r not in spans_here and not any(a[0].startswith(r + ".") for a in actions):
                v = after[r]
                if "time" in rs and v == rs["time"]: src = ("copy", "time")
                elif "noun" in rs and v == rs["noun"][0]: src = ("copy", "noun")
                elif "pron" in rs and v == rs["pron"][0]: src = ("copy", "pron")
                else: src = ("const",)
                actions.append((f"{r}.noun", v, src)); actions.append((f"{r}.cls", "t" if r != "subj" else "h", ("const",)))
        # moves between NP slots (give him a book: obj -> iobj) show up as a slot acquiring another slot's old string
        for r in ("iobj", "obj"):
            if after.get(r) and after[r] != before.get(r) and (r, ei) not in [(rr, e2) for (rr, e2) in [np_role_of_span.get(k, (None, None)) if isinstance(np_role_of_span.get(k), tuple) else (None, None) for k in np_role_of_span]]:
                for r2 in ("obj", "iobj", "subj"):
                    if before.get(r2) == after[r] and r2 != r:
                        for sub in NP_SUB: actions.append((f"{r}.{sub}", None, ("move", f"{r2}.{sub}")))
        nxt_e = events[ei + 1] if ei + 1 < len(events) else {"win_reads": set(), "reg_reads": set()}
        win_idx = set(e["win_reads"]) | set(nxt_e["win_reads"])          # reads that led into this consumption and reads that followed it
        reg_rd = set(e["reg_reads"]) | set(nxt_e["reg_reads"])
        allw = window_features(toks, e["i"])
        feats = tuple((j - e["i"], allw[j - e["i"]]) for j in sorted(win_idx) if 0 <= j - e["i"] < WINDOW)
        hits = table_hits(toks, e["i"], before["verb"])
        guard = []
        for rname in sorted(reg_rd):
            if rname in ("subj",): guard.append((rname, before.get("subj") is not None, before.get("subj") == "それ"))
            elif rname in ("obj", "iobj", "pp", "time", "adv", "copula", "wh", "degree", "want", "modal", "purpose", "causative", "obj_particle"):
                v = before.get({"pp": "pp1", "adv": "adv1", "copula": "cop_kind"}.get(rname, rname))
                guard.append((rname, v if rname in ("want", "modal", "purpose", "causative", "obj_particle", "degree", "wh", "copula") else v is not None, len([k for k in ("pp1", "pp2", "pp3") if before.get(k)]) if rname == "pp" else None))
            elif rname == "verb":
                vb = before["verb"]; guard.append(("verb", vb is not None, tuple(sorted(cls for cls, s in R.VC.items() if vb and vb[0] in s)), vb[2] if vb else None, " " in vb[0] if vb else None, "+" in vb[0] if vb else None))
            elif rname in FLAGS: guard.append((rname, bool(before.get(rname))))
            elif rname in ("subj_cls",): guard.append((rname, before.get("subj_cls")))
        key = (e["cp"], feats, hits, tuple(guard))
        nxt = events[ei + 1]["cp"] if ei + 1 < len(events) else ("DONE",)
        norm = tuple(sorted(((slot, None if src[0] in ("copy", "move") else v, src) for slot, v, src in actions), key=str))
        rows.append((key, (nxt, e["k"], norm)))
    return rows, final

if __name__ == "__main__":
    import run_rules as H
    pairs = H.load("test") + H.load("dev")
    table = {}; conflicts = collections.Counter(); scope = collections.Counter(); n_rows = 0
    per_sentence = []
    for en, ja in pairs:
        toks = R.merge_multiword(en if isinstance(en, list) else en.split())
        try:
            # the bounded driver's winning choices
            p0 = Parser(list(toks), []); p0.parse()
            rows, final = event_rows(toks, [])
        except ParseFail: scope["rule parser fails"] += 1; continue
        except OutOfScope as e: scope["out of scope: " + str(e).split(" at ")[0][:28]] += 1; continue
        except Exception as e: scope["extraction error " + type(e).__name__ + ": " + str(e)[:40]] += 1; continue
        scope["in scope"] += 1; per_sentence.append((toks, rows))
        for key, val in rows:
            n_rows += 1
            if key in table and table[key] != val: conflicts[key[0]] += 1
            table.setdefault(key, val)
    print("scope:", dict(scope.most_common()))
    print(f"{n_rows} consumption events -> {len(table)} distinct transition rows; conflicting keys: {sum(conflicts.values())}")
    for cp, n in conflicts.most_common(5): print("   conflict at", ">".join(cp[0][1:]), "L", cp[1], n)

# ----------------------------------------------------------------------------------------------- predicates and induction
ENUM_REGS = ("modal", "want", "cop_kind", "degree", "obj_particle", "wh", "purpose", "causative", "subj_cls", "time", "adv1")
def predicates(toks, i, before, win_idx, reg_rd, pend_prep):
    """binary predicates available to guard the transition at token i: only over what the parser read"""
    P = set()
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
        elif rname in ("obj", "iobj", "cop"):
            if before.get(rname) is not None: P.add(f"reg.{rname}")
        elif rname == "pp":
            n = len([k for k in ("pp1", "pp2", "pp3") if before.get(k)]); P.add(f"pp.count={n}")
        elif rname == "copula":
            if before.get("cop_kind"): P.add(f"reg.cop_kind={before['cop_kind']}")
        elif rname == "verb":
            vb = before["verb"]
            if vb:
                P.add("verb.set"); P.add(f"verb.conj={vb[2]}")
                if " " in vb[0]: P.add("verb.phrasal")
                if "+" in vb[0]: P.add("verb.use")
                for cls, s in R.VC.items():
                    if vb[0] in s: P.add(f"verb.vc={cls}")
        elif rname in FLAGS:
            if before.get(rname): P.add(f"flag.{rname}")
        elif rname in ENUM_REGS or rname == "adv":
            key = {"adv": "adv1"}.get(rname, rname); v = before.get(key)
            if v not in (None, "", False): P.add(f"reg.{rname}={'set' if rname in ('time', 'adv', 'adv1') else v}")
    if pend_prep: P.add(f"pend_prep={pend_prep}")
    return P

ALL_REGS = {"subj", "obj", "iobj", "cop", "pp", "copula", "verb", "adv", "time"} | set(FLAGS) | set(ENUM_REGS)
def event_rows_pred(toks, choices=()):
    """rows as (prev_cp, all predicates true here, predicates the parser actually read, outcome); outcome = (cp, k, actions)"""
    rows, final = event_rows(toks, choices)      # reuse the diffing; re-derive predicate sets from the trace
    events, _final, _spans, _p = trace(toks, choices)
    out = []; prev = ("START",); pend = None
    for ei, (e, (key, val)) in enumerate(zip(events, rows)):
        nxt_e = events[ei + 1] if ei + 1 < len(events) else {"win_reads": set(), "reg_reads": set()}
        win_idx = set(e["win_reads"]) | set(nxt_e["win_reads"]); reg_rd = set(e["reg_reads"]) | set(nxt_e["reg_reads"])
        P_read = predicates(toks, e["i"], e["snap"], win_idx, reg_rd, pend)
        P_all = predicates(toks, e["i"], e["snap"], set(range(e["i"], e["i"] + WINDOW)), ALL_REGS, pend)
        cp, k, actions = e["cp"], val[1], val[2]
        out.append((prev, frozenset(P_all), frozenset(P_read), (cp, k, actions)))
        prev = cp
        for j in range(e["i"], e["i"] + k):
            if "prep" in dict(readings_of(toks[j])): pend = toks[j]
    P_end = predicates(toks, len(toks), final, set(range(len(toks), len(toks) + WINDOW)), ALL_REGS, pend)
    out.append((prev, frozenset(P_end), frozenset(), (("DONE",), 0, ())))          # after the end token: the readout state
    return out, final

FAIL = ("FAIL",)
def fail_rows(toks):
    """for a sentence the rule parser refuses, or parses outside this implementation's scope: one FAIL row at the point
    where it stops being representable, so the compiled machine refuses there instead of guessing"""
    try:
        events, final, spans, p = trace(toks)
    except ParseFail as e:
        # the rule parser refused: the row is the transition that would have followed the last consumption
        p = _last_parser[0]
        err, fs, win_rd, reg_rd, i = p.fail_info
        if isinstance(fs, OutOfScope) or not p.events_ok: return []
        prev = p.last_cp
        P_all = predicates(toks, i, fs, set(range(i, i + WINDOW)), ALL_REGS, p.pend_at_end)
        P_read = predicates(toks, i, fs, win_rd, reg_rd, p.pend_at_end)
        return [(prev, frozenset(P_all), frozenset(P_read), (FAIL, 0, ()))]
    # parsed, but out of scope: cut at the first event whose snapshot or NP is not representable
    cut = None
    for ei, e in enumerate(events):
        if isinstance(e["snap"], OutOfScope) or e["snap"] is None: cut = ei - 1; break
    if cut is None:
        try: snapshot(final)
        except OutOfScope: cut = len(events) - 1
    for (i0, i1, ja, cls, main) in spans:
        try:
            if not main: raise OutOfScope("inner")
            np_components(toks, i0, i1, ja)
        except OutOfScope:
            ev = [k for k, e in enumerate(events) if e["i"] <= i0 < e["i"] + e["k"]]
            if ev and (cut is None or ev[0] < cut): cut = ev[0]
    if cut is None or cut < 0: return []
    e = events[cut]; prev = events[cut - 1]["cp"] if cut > 0 else ("START",)
    if isinstance(e["snap"], OutOfScope) or e["snap"] is None: return []
    pend = None
    for j in range(e["i"]):
        if "prep" in dict(readings_of(toks[j])): pend = toks[j]
    nxt_e = events[cut + 1] if cut + 1 < len(events) else {"win_reads": set(), "reg_reads": set()}
    P_all = predicates(toks, e["i"], e["snap"], set(range(e["i"], e["i"] + WINDOW)), ALL_REGS, pend)
    P_read = predicates(toks, e["i"], e["snap"], set(e["win_reads"]) | set(nxt_e["win_reads"]), set(e["reg_reads"]) | set(nxt_e["reg_reads"]), pend)
    return [(prev, frozenset(P_all), frozenset(P_read), (FAIL, 0, ()))]

_last_parser = [None]
def induce(rows):
    """per previous control point, a decision tree over predicates whose leaves are outcomes; returns {prev: tree}"""
    by_prev = collections.defaultdict(list); read_by_prev = collections.defaultdict(set)
    for prev, P_all, P_read, outcome in rows: by_prev[prev].append((P_all, outcome)); read_by_prev[prev] |= P_read
    def build(items, used, cands_all):
        outcomes = set(o for _, o in items)
        if len(outcomes) == 1: return ("leaf", next(iter(outcomes)))
        cands = cands_all - used
        best = None
        for q in cands:
            yes = [(P, o) for P, o in items if q in P]; no = [(P, o) for P, o in items if q not in P]
            if not yes or not no: continue
            score = (len(set(o for _, o in yes)) + len(set(o for _, o in no)), abs(len(yes) - len(no)))
            if best is None or score < best[0]: best = (score, q, yes, no)
        if best is None: return ("conflict", sorted(outcomes, key=str), len(items))
        _, q, yes, no = best
        return ("split", q, build(yes, used | {q}, cands_all), build(no, used | {q}, cands_all))
    trees = {}
    for prev, items in by_prev.items():
        t = build(items, frozenset(), read_by_prev[prev])
        if has_conflict(t):                                    # the read predicates do not separate: fall back to everything true here
            t = build(items, frozenset(), set().union(*(P for P, _ in items)))
        trees[prev] = t
    return trees

def has_conflict(tree):
    if tree[0] == "leaf": return False
    if tree[0] == "conflict": return True
    return has_conflict(tree[2]) or has_conflict(tree[3])

def tree_rules(tree, path=()):
    """flatten a tree into (literals, outcome) rules; literals are (predicate, truth)"""
    if tree[0] == "leaf": return [(path, tree[1])]
    if tree[0] == "conflict": return [(path, ("CONFLICT", tree[1], tree[2]))]
    _, q, yes, no = tree
    return tree_rules(yes, path + ((q, True),)) + tree_rules(no, path + ((q, False),))

def run_table(trees, toks):
    """interpret the induced machine on a sentence (a check of the extraction before the matrix exists)"""
    state = ("START",); i = 0; frame = {}; pend = None
    while True:
        if state == ("DONE",): return frame
        if state not in trees: raise ParseFail(f"TABLE: no rules at {'>'.join(state[0][1:]) if state[0] != 'START' else 'START'}", i)
        before = frame_to_snapshot(frame)
        P = predicates(toks, i, before, set(range(i, i + WINDOW)), ALL_REGS, pend)
        t = trees[state]
        while t[0] == "split":
            t = t[2] if t[1] in P else t[3]
        if t[0] == "conflict": raise ParseFail("TABLE: conflict", i)
        cp, k, actions = t[1]
        apply_actions(frame, actions, toks, i)
        for j in range(i, i + k):
            if j < len(toks) and "prep" in dict(readings_of(toks[j])): pend = toks[j]
        i += k; state = cp
        if i > len(toks): raise ParseFail("TABLE: ran past the end", i)

def frame_to_snapshot(frame):
    """the slot dict as the snapshot() view (so predicates read it the same way)"""
    s = {f: bool(frame.get(f)) for f in FLAGS}
    for r in ("modal", "want", "degree", "obj_particle", "wh", "time", "purpose", "causative", "verb", "adv1", "adv2", "cop_kind", "cop_adj", "cop_adv", "cop_part", "pp1_part", "pp2_part", "pp3_part"):
        s[r] = frame.get(r)
    for role in NP_ROLES:
        s[role] = np_string(frame, role)
    s["subj_cls"] = frame.get("subj.cls") if s["subj"] else None
    return s

def np_string(frame, role):
    parts = [frame.get(f"{role}.{sub}") for sub in ("det", "poss", "num", "adj1", "adj2", "noun")]
    if frame.get(f"{role}.noun") is None: return None
    return "".join(p for p in parts if p)

def apply_actions(frame, actions, toks, i):
    tok = toks[i] if i < len(toks) else None; rs = collections.defaultdict(list)
    for c, v in readings_of(tok): rs[c].append(v)
    for slot, v, src in actions:
        if src[0] == "copy" and src[1] not in ("ncls",) and not rs[{"num_time": "num", "adj_pair": "adj", "pron_poss": "pron"}.get(src[1], src[1])]:
            raise ParseFail(f"TABLE: nothing to copy into {slot} from {src[1]} at {tok!r}", i)       # the machine's crash: the rule fired on a token without that reading
        if src[0] == "clear": frame.pop(slot, None)
        elif src[0] == "const": frame[slot] = v
        elif src[0] == "move":
            frame[slot] = frame.get(src[1]); 
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
            elif cls == "adv": frame[slot] = rs["adv"][-1]
            elif cls == "time": frame[slot] = rs["time"][-1]
            elif cls == "wh": frame[slot] = rs["wh"][-1]
            elif cls == "modal": frame[slot] = rs["modal"][-1][0]
            elif cls == "verb":
                lem, form, ja, conj = rs["verb"][0]; frame[slot] = (lem, ja, conj)

def frame_to_clause(frame):
    c = Clause()
    for f in FLAGS: setattr(c, f, bool(frame.get(f)))
    for r in ("modal", "want", "obj_particle", "wh", "purpose", "causative", "verb"): setattr(c, r, frame.get(r))
    c.degree = frame.get("degree") or ""; c.time = frame.get("time") or ""; c.adv = frame.get("adv1") or ""
    c.subj = np_string(frame, "subj"); c.subj_cls = frame.get("subj.cls") or "h"
    c.obj = np_string(frame, "obj"); c.iobj = np_string(frame, "iobj")
    c.pp = [(frame[f"pp{k}_part"], np_string(frame, f"pp{k}")) for k in (1, 2, 3) if frame.get(f"pp{k}.noun") is not None]
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

COPY_SOURCES = ("noun", "ncls", "pron", "pron_poss", "det", "num", "num_time", "adj", "adj_pair", "adv", "time", "wh", "modal", "verb")

def source_values(tok):
    """the copy-source readings of a token, as (source, value) pairs — what the lexical layer emits for window slot 0"""
    out = []
    for c, v in readings_of(tok):
        if c == "noun": out.append(("noun", v[0])); out.append(("ncls", v[1]))
        elif c == "pron": out.append(("pron", v[0])); out.append(("pron_poss", v[0] if v[1] == "p" else v[0] + "の"))
        elif c == "det": out.append(("det", v))
        elif c == "num": out.append(("num", v)); out.append(("num_time", v + "時"))
        elif c == "adj": out.append(("adj", v[0] + ("な" if v[1] == "na" and not v[0].endswith("な") else ""))); out.append(("adj_pair", v))
        elif c == "adv": out.append(("adv", v))
        elif c == "time": out.append(("time", v))
        elif c == "wh": out.append(("wh", v))
        elif c == "modal": out.append(("modal", v[0]))
        elif c == "verb": out.append(("verb", (v[0], v[2], v[3])))
    # the rule parser takes the LAST reading of a class (dict() semantics) for noun/adj/adv/time/wh, the FIRST for pron/det/num/verb
    last = {}; first = {}
    for s, v in out:
        last[s] = v; first.setdefault(s, v)
    pick = {s: (last[s] if s in ("noun", "ncls", "adj", "adj_pair", "adv", "time", "wh") else first[s]) for s in last}
    return pick

def compile_matrices(trees, vocab_hint=()):
    """trees (per previous control point) -> the cell: four sparse layers over a one-hot state and the lexical input"""
    rules = []   # (prev, literals, (cp, k, actions))
    for prev, t in trees.items():
        for lits, out in tree_rules(t):
            if out[0] == "CONFLICT": continue
            rules.append((prev, lits, out))
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
    for _, _, (cp, k, actions) in rules:
        for slot, v, src in actions:
            if src[0] == "copy": copy_pairs.add((slot, src[1])); slot_vocab[slot] |= set(src_vocab[src[1]])
            if src[0] == "move": move_pairs.add((slot, src[1]))
    for slot, src_slot in list(move_pairs): slot_vocab[slot] |= slot_vocab[src_slot]
    for slot, src_slot in list(move_pairs): slot_vocab[slot] |= slot_vocab[src_slot]
    # ---- state index: control, slots, pend_prep
    S = Index()
    ctrl_states = {("START",), ("DONE",), FAIL} | set(trees) | set(out[0] for _, _, out in rules)
    skip_states = []
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        for j in range(1, k): skip_states.append(("SKIP", ri, j))
    for c in sorted(ctrl_states, key=str): S.add(("ctrl", c))
    for c in skip_states: S.add(("ctrl", c))
    for slot in sorted(slot_vocab):
        for v in sorted(slot_vocab[slot], key=str): S.add(("slot", slot, v))
    preps = sorted(w for w in R.PREPS)
    for w in preps: S.add(("pend", w))
    # ---- input index: window predicates, copy sources, prep-surface of slot 0
    X = Index()
    for prev, lits, out in rules:
        for q, truth in lits:
            if not (q.startswith("reg.") or q.startswith("flag.") or q.startswith("verb.") or q == "subj.sore" or q.startswith("pp.count") or q.startswith("pend_prep=")): X.add(("pred", q))
    for s in COPY_SOURCES:
        for v in src_vocab.get(s, []): X.add(("src", s, v))
    for w in preps: X.add(("prep0", w))
    X.add(("one",))
    # ---- layer 1: register predicates derived from the state (OR over slot units); plus pass-through is implicit (skip connections)
    L1 = Index(); W1 = []; b1 = {}
    def or_unit(name, src_idx):
        u = L1.add(name)
        for i in src_idx: W1.append((u, i, 1.0))
        return u
    def slot_units(slot): return [S[("slot", slot, v)] for v in slot_vocab.get(slot, [])]
    for role in NP_ROLES:
        if slot_vocab.get(f"{role}.noun"): or_unit(("pred", f"reg.{role}"), slot_units(f"{role}.noun"))
    if ("slot", "subj.noun", "それ") in S: or_unit(("pred", "subj.sore"), [S[("slot", "subj.noun", "それ")]])
    for cls in ("h", "t", "a", "p"):
        if ("slot", "subj.cls", cls) in S: or_unit(("pred", f"reg.subj_cls={cls}"), [S[("slot", "subj.cls", cls)]])
    for f in FLAGS:
        if slot_vocab.get(f): or_unit(("pred", f"flag.{f}"), slot_units(f))
    for e in ("modal", "want", "degree", "obj_particle", "wh", "purpose", "causative", "cop_kind"):
        for v in slot_vocab.get(e, []):
            or_unit(("pred", f"reg.{e}={v}"), [S[("slot", e, v)]])
    for e in ("time", "adv1"):
        if slot_vocab.get(e): or_unit(("pred", f"reg.{e}=set"), slot_units(e)); 
    if slot_vocab.get("adv1"): or_unit(("pred", "reg.adv=set"), slot_units("adv1"))
    vunits = slot_units("verb")
    if vunits:
        or_unit(("pred", "verb.set"), vunits)
        for cls, s in R.VC.items():
            idx = [S[("slot", "verb", v)] for v in slot_vocab["verb"] if v[0] in s]
            if idx: or_unit(("pred", f"verb.vc={cls}"), idx)
        for conj in set(v[2] for v in slot_vocab["verb"]):
            or_unit(("pred", f"verb.conj={conj}"), [S[("slot", "verb", v)] for v in slot_vocab["verb"] if v[2] == conj])
        or_unit(("pred", "verb.phrasal"), [S[("slot", "verb", v)] for v in slot_vocab["verb"] if " " in v[0]] or [])
        or_unit(("pred", "verb.use"), [S[("slot", "verb", v)] for v in slot_vocab["verb"] if "+" in v[0]] or [])
    # pp.count=n from the filled bits
    f = [L1.add(("filled", f"pp{k}")) for k in (1, 2, 3)]
    for k in (1, 2, 3):
        for i in slot_units(f"pp{k}.noun"): W1.append((f[k-1], i, 1.0))
    for w in preps: or_unit(("pred", f"pend_prep={w}"), [S[("pend", w)]])
    # layer 1b (inside layer 2's input): pp.count units need the filled bits -> put them in layer 2 as AND/NOT; rules then sit in layer 3
    L2 = Index(); W2 = []; b2 = {}
    u = L2.add(("pred", "pp.count=0")); b2[u] = 1.0
    for k in range(3): W2.append((u, ("L1", f[k]), -1.0))
    for n in (1, 2):
        u = L2.add(("pred", f"pp.count={n}")); W2.append((u, ("L1", f[n-1]), 1.0)); W2.append((u, ("L1", f[n]), -1.0))
    u = L2.add(("pred", "pp.count=3")); W2.append((u, ("L1", f[2]), 1.0))
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
            ref = pred_ref(q)
            if ref is None:
                if truth: unresolved[q] += 1; W3.append((u, ("X", X[("one",)]), -5.0))      # a positive literal the cell cannot see: the rule can never fire
                continue                                                                # a negative literal on an unseen predicate is vacuously true
            if truth: W3.append((u, ref, 1.0)); npos += 1
            else: W3.append((u, ref, -1.0))
        b3[u] = -(npos - 1)
    # skip-state rules: SKIP(ri, j) -> SKIP(ri, j-1) or the target
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        for j in range(1, k):
            u = L3.add(("skiprule", ri, j)); W3.append((u, ("S", S[("ctrl", ("SKIP", ri, j))]), 1.0)); b3[u] = 0.0
    # ---- layer 4: next control, gates, keep/write
    L4 = Index(); W4 = []; b4 = {}
    def rule_target(ri, k, cp): return ("ctrl", ("SKIP", ri, k - 1)) if k > 1 else ("ctrl", cp)
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        tgt = rule_target(ri, k, cp) if k >= 1 else ("ctrl", cp)
        u = L4.add(("next", tgt)); W4.append((u, ("L3", L3[("rule", ri)]), 1.0))
        for j in range(1, k):
            nxt = ("ctrl", ("SKIP", ri, j - 1)) if j > 1 else ("ctrl", cp)
            u2 = L4.add(("next", nxt)); W4.append((u2, ("L3", L3[("skiprule", ri, j)]), 1.0))
    # gates per slot (any write/clear) and per (slot, source) copy / (slot, from) move
    gate_slot = {}; gate_copy = {}; gate_move = {}; const_w = collections.defaultdict(list)
    for ri, (prev, lits, (cp, k, actions)) in enumerate(rules):
        for slot, v, src in actions:
            g = gate_slot.setdefault(slot, L4.add(("gate", slot))); W4.append((g, ("L3", L3[("rule", ri)]), 1.0))
            if src[0] == "copy":
                gc = gate_copy.setdefault((slot, src[1]), L4.add(("gcopy", slot, src[1]))); W4.append((gc, ("L3", L3[("rule", ri)]), 1.0))
            elif src[0] == "move":
                gm = gate_move.setdefault((slot, src[1]), L4.add(("gmove", slot, src[1]))); W4.append((gm, ("L3", L3[("rule", ri)]), 1.0))
            elif src[0] == "const":
                const_w[(slot, v)].append(ri)
    # pend_prep gate: any token with a prep reading in slot 0 rewrites it
    gp = L4.add(("gate", "pend"))
    for w in preps: W4.append((gp, ("X", X[("prep0", w)]), 1.0))
    # ---- layer 5: keep / write per slot value
    L5 = Index(); W5 = []; b5 = {}
    for slot in sorted(slot_vocab):
        g = gate_slot.get(slot)
        for v in slot_vocab[slot]:
            si = S[("slot", slot, v)]
            kp = L5.add(("keep", slot, v)); W5.append((kp, ("S", si), 1.0))
            if g is not None: W5.append((kp, ("L4", g), -1.0))
            for (sl, s) in copy_pairs:
                if sl == slot and v in src_vocab.get(s, []) and ("src", s, v) in X:
                    wu = L5.add(("write", slot, s, v)); W5.append((wu, ("L4", gate_copy[(sl, s)]), 1.0)); W5.append((wu, ("X", X[("src", s, v)]), 1.0)); b5[wu] = -1.0
            for (sl, from_slot) in move_pairs:
                if sl == slot and ("slot", from_slot, v) in S:
                    wu = L5.add(("writem", slot, from_slot, v)); W5.append((wu, ("L4", gate_move[(sl, from_slot)]), 1.0)); W5.append((wu, ("S", S[("slot", from_slot, v)]), 1.0)); b5[wu] = -1.0
            if (slot, v) in const_w:
                wu = L5.add(("writec", slot, v))                       # a constant write is its own unit: OR of the rules that write it
                for ri in const_w[(slot, v)]: W5.append((wu, ("L3", L3[("rule", ri)]), 1.0))
    for w in preps:
        kp = L5.add(("keep", "pend", w)); W5.append((kp, ("S", S[("pend", w)]), 1.0)); W5.append((kp, ("L4", gp), -1.0))
        wu = L5.add(("write", "pend", w)); W5.append((wu, ("X", X[("prep0", w)]), 1.0))
    # ---- layer 6: the new state = next control + keep + write
    W6 = []
    for i, name in enumerate(L4.names):
        if name[0] == "next": W6.append((S[name[1]], ("L4", i), 1.0))
    for i, name in enumerate(L5.names):
        if name[0] in ("keep", "write", "writec", "writem"):
            key = ("pend", name[2]) if name[1] == "pend" else ("slot", name[1], name[-1])
            W6.append((S[key], ("L5", i), 1.0))
    layers = [(L1, W1, b1), (L2, W2, b2), (L3, W3, b3), (L4, W4, b4), (L5, W5, b5)]
    return {"S": S, "X": X, "layers": layers, "W6": W6, "rules": rules, "slot_vocab": slot_vocab, "src_vocab": src_vocab, "unresolved": unresolved}

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
            M = sp.csr_matrix((vals_, (rows_, cols_)), shape=(len(L), width))
            bias = np.zeros(len(L))
            for u, v in b.items(): bias[u] = v
            self.mats.append((M, bias)); self.offsets["L" + str(len(self.mats))] = width; width += len(L)
        rows_, cols_, vals_ = [], [], []
        for (si, ref, w) in compiled["W6"]:
            rows_.append(si); cols_.append(self.offsets[ref[0]] + ref[1]); vals_.append(w)
        self.W6 = sp.csr_matrix((vals_, (rows_, cols_)), shape=(self.nS, width))
        self.width = width
    def nnz(self): return sum(M.nnz for M, _ in self.mats) + self.W6.nnz
    def lexical_input(self, toks, i):
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
        verb_now = None
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
        S = self.c["S"]; frame = {}; ctrl = None
        for idx in np.nonzero(s > 0.5)[0]:
            name = S.names[idx]
            if name[0] == "ctrl": ctrl = name[1]
            elif name[0] == "slot": frame[name[1]] = name[2]
        return ctrl, frame
    def parse(self, toks):
        S = self.c["S"]; s = np.zeros(self.nS); s[S[("ctrl", ("START",))]] = 1.0; i = 0
        for step in range(len(toks) + 2):
            ctrl, frame = self.read_state(s)
            if ctrl == ("DONE",): return frame_to_clause(frame)
            if ctrl == FAIL: raise ParseFail(f"MATRIX: FAIL at token {i} ({toks[i] if i < len(toks) else 'END'!r})", i)
            if ctrl is None: raise ParseFail(f"MATRIX: no rule fired at token {i} ({toks[i] if i < len(toks) else 'END'!r})", i)
            verb_value = frame.get("verb")
            x = self.lexical_input(toks, i)
            s = self.step(s, x, toks, i, verb_value)
            if not (isinstance(ctrl, tuple) and ctrl and ctrl[0] == "SKIP") or True:
                i += 1 if i < len(toks) else 0
        raise ParseFail("MATRIX: did not reach DONE", i)
