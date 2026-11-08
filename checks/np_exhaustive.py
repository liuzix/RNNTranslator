"""Exhaustive check of the hand-translated NP module against parser_rules.Parser.parse_np."""
import os, sys, itertools, collections, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rules_ja as R, np_module as M
from parser_rules import Parser, ParseFail, Clause
# one representative token per equivalence class parse_np can distinguish
tests = set(M.PERSONAL) | {"a lot", "a", "lot", "lots", "of", "no", "one", "'s", "course", "and", "fact", "who", "which", ",", "."}
def sig(t):
    rs = R.classify(t); cls = tuple(sorted(set(c for c, _ in rs)))
    ex = tuple(sorted({("ncls", v[1]) for c, v in rs if c == "noun"} | {("prole", v[1]) for c, v in rs if c == "pron"} | {("akind", v[1]) for c, v in rs if c == "adj"}))
    return (t if t in tests else "*", cls, ex)
reps = {}
for t in list(R.LEX) + list(R.PRONOUNS) + list(R.DETS) + list(R.NUMBERS) + list(R.WH) + list(R.VERB_FORMS) + ["'s", ".", ",", "and", "that", "fact", "course", "of", "lot", "a lot", "lots"]:
    reps.setdefault(sig(t), t)
alphabet = sorted(set(reps.values()) - {"."}); print(len(alphabet), "representative tokens")
CTX = {"subj": dict(phase="PRE", subj=None, verb=None, cop=None, role=None), "obj": dict(phase="VP0", subj="彼", verb=("eat", "食べる", "i"), cop=None, role=None),
       "cop": dict(phase="VP0", subj="それ", verb=None, cop=None, role=None), "ppt": dict(phase="VP0", subj="彼", verb=("eat", "食べる", "i"), cop=None, role="pp_time")}
made = [0]; depth = [0, 0]
orig_init = Clause.__init__
def counting_init(self): made[0] += 1; orig_init(self)
Clause.__init__ = counting_init
orig_np = Parser.parse_np
def np_counting(self):
    depth[0] += 1; depth[1] = max(depth[1], depth[0])
    try: return orig_np(self)
    finally: depth[0] -= 1
Parser.parse_np = np_counting
def reference(seq, ctx):
    toks = list(seq) + ["."]
    p = Parser(toks, []); p.c.subj = ctx["subj"]; p.c.verb = ctx["verb"]; p.c.copula = ctx["cop"]; p.phase = ctx["phase"]; p.np_role = ctx["role"]
    made[0] = 0; depth[0] = depth[1] = 0
    try: ja, cls = p.parse_np()
    except ParseFail as e: return ("FAIL", str(e).split(" at ")[0])
    except (IndexError, TypeError, KeyError, AttributeError) as e: return ("CRASH", type(e).__name__)
    if made[0] > 0: return ("OOS", "inner clause", ja)
    if depth[1] > 1: return ("OOS", "complex NP", ja)
    return ("OK", p.i, ja, cls, bool(p.c.neg))
def module(seq, ctx, role, rules, D):
    toks = list(seq) + ["."]
    regs = {"verb": ctx["verb"], "cop_kind": ctx["cop"][0] if ctx["cop"] else None, "subj": ctx["subj"]}
    pend = "at" if ctx["role"] == "pp_time" else None
    out = M.run_module(role, toks, 0, regs, pend, rules, D)
    if out[0] == "RET":
        _, i, slots, flags = out; return ("OK", i, M.np_string(slots, role), slots.get(f"{role}.cls"), bool(flags.get("neg")))
    return ("FAIL", out[1])
OOS_REASONS = {"relative clause", "of-phrase", "coordination", "possessive noun phrase", "the fact that", "three nouns", "three adjectives"}
def compare(ref, mod):
    if ref[0] == "OK":
        return mod[0] == "OK" and mod[1:] == ref[1:], "flat NP"
    if ref[0] == "FAIL": return mod[0] == "FAIL", "refusal"
    if ref[0] == "OOS":
        if mod[0] == "OK" and ref[1] == "complex NP" and mod[2] == ref[2]: return True, "possessive the module holds"
        return mod[0] == "FAIL" and mod[1] in OOS_REASONS, "out of scope"
    return True, "crash in reference"
t0 = time.time(); total = 0; bad = collections.Counter(); examples = {}
for ctxname, ctx in CTX.items():
    role = "subj" if ctxname == "subj" else ("cop" if ctxname == "cop" else "obj")
    rules, D = M.rules_for(role), M.derived(role)
    for L in (1, 2, 3):
        for seq in itertools.product(alphabet, repeat=L):
            total += 1
            ref = reference(seq, ctx)
            if ref[0] == "OK" and ref[1] < L: continue            # the NP ended before the sequence did: a shorter case already covers it
            mod = module(seq, ctx, role, rules, D)
            ok, kind = compare(ref, mod)
            if not ok:
                key = (ctxname, kind, str(ref[:2]), str(mod[:2])); bad[key] += 1; examples.setdefault(key, seq)
print(f"{total} cases in {time.time()-t0:.0f}s; mismatching kinds: {len(bad)}, mismatching cases: {sum(bad.values())}")
for key, n in bad.most_common(25): print(f"  {n:5d}  ctx {key[0]:4s} {key[1]:12s} ref {key[2][:60]:60s} module {key[3][:50]:50s} e.g. {examples[key]}")

# length 4 and 5 over the tokens that can occur inside or right after a flat noun phrase
small = [t for t in alphabet if set(c for c, _ in R.classify(t)) & {"det", "pron", "num", "adj", "adv", "noun"} or t in ("'s", "of", "and", "that", "who", "which", "is", "came", ",", "in")]
small = sorted(set(small))[:26]
print(len(small), "tokens for the longer sequences:", small)
t0 = time.time(); total = 0; bad = collections.Counter(); examples = {}
for ctxname, ctx in CTX.items():
    role = "subj" if ctxname == "subj" else ("cop" if ctxname == "cop" else "obj")
    rules, D = M.rules_for(role), M.derived(role)
    for L in (4,):
        for seq in itertools.product(small, repeat=L):
            total += 1
            ref = reference(seq, ctx)
            if ref[0] == "OK" and ref[1] < L: continue
            mod = module(seq, ctx, role, rules, D)
            ok, kind = compare(ref, mod)
            if not ok:
                key = (ctxname, kind, str(ref[:2]), str(mod[:2])); bad[key] += 1; examples.setdefault(key, seq)
print(f"length 4: {total} cases in {time.time()-t0:.0f}s; mismatching kinds: {len(bad)}, cases: {sum(bad.values())}")
for key, n in bad.most_common(8): print(f"  {n:5d}  {key} e.g. {examples[key]}")
