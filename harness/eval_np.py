"""Exhaustive check of the hand-translated NP module (np_module) against parser_rules.Parser.parse_np.

One representative token per equivalence class parse_np can distinguish (its reading classes, noun class, pronoun role,
adjective kind, and the surface words it tests by identity); every sequence of those up to --max-len, followed by the
end of the sentence, in four call contexts (subject, object, copula complement, time PP).  The module must return the
same string, class, position and neg flag as parse_np on every flat NP, refuse where parse_np refuses, and exit with
FAIL and a reason wherever parse_np would open an inner frame or recurse (relative clause, of-phrase, coordination, …).
"""
import sys, os, json, itertools, collections, time, argparse
sys.path.insert(0, os.getcwd())
import rules_ja as R, np_module as M
from parser_rules import Parser, ParseFail, Clause

TESTS = set(M.PERSONAL) | {"a lot", "a", "lot", "lots", "of", "no", "one", "'s", "course", "and", "fact", "who", "which", ",", "."}
def sig(t):
    rs = R.classify(t); cls = tuple(sorted(set(c for c, _ in rs)))
    ex = tuple(sorted({("ncls", v[1]) for c, v in rs if c == "noun"} | {("prole", v[1]) for c, v in rs if c == "pron"} | {("akind", v[1]) for c, v in rs if c == "adj"} | {("number", R.number_of(t))}))
    return (t if t in TESTS else "*", cls, ex)

def alphabet():
    reps = {}
    for t in list(R.LEX) + list(R.PRONOUNS) + list(R.DETS) + list(R.NUMBERS) + list(R.WH) + list(R.VERB_FORMS) + ["'s", ".", ",", "and", "that", "fact", "course", "of", "lot", "a lot", "lots"]:
        reps.setdefault(sig(t), t)
    return sorted(set(reps.values()) - {"."})

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
    role = "subj" if ctx["phase"] == "PRE" else None
    try: ja, cls = p.parse_np(role=role)
    except ParseFail as e: return ("FAIL", str(e).split(" at ")[0])
    except (IndexError, TypeError, KeyError, AttributeError) as e: return ("CRASH", type(e).__name__)
    if made[0] > 0: return ("OOS", "inner clause", ja)
    if depth[1] > 1: return ("OOS", "complex NP", ja)
    return ("OK", p.i, ja, cls, bool(p.c.neg), p.np_number if role == "subj" else None)

def module(seq, ctx, role, rules, D):
    toks = list(seq) + ["."]
    regs = {"verb": ctx["verb"], "cop_kind": ctx["cop"][0] if ctx["cop"] else None, "subj": ctx["subj"]}
    pend = "at" if ctx["role"] == "pp_time" else None
    out = M.run_module(role, toks, 0, regs, pend, rules, D)
    if out[0] == "RET":
        _, i, slots, flags = out
        number = slots.get(f"{role}.number") if role == "subj" else None
        cls = slots.get(f"{role}.cls")
        return ("OK", i, M.np_string(slots, role), cls, bool(flags.get("neg")), None if number == "none" else number)
    return ("FAIL", out[1])

OOS_REASONS = {"relative clause", "of-phrase", "coordination", "possessive noun phrase", "the fact that", "three nouns", "three adjectives"}
def compare(ref, mod):
    if ref[0] == "OK": return mod[0] == "OK" and mod[1:] == ref[1:], "flat NP"
    if ref[0] == "FAIL": return mod[0] == "FAIL", "refusal"
    if ref[0] == "OOS":
        if mod[0] == "OK" and ref[1] == "complex NP" and mod[2] == ref[2]: return True, "possessive the module holds"
        return mod[0] == "FAIL" and mod[1] in OOS_REASONS, "out of scope"
    return True, "crash in reference"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-len", type=int, default=2); ap.add_argument("--out", default="result.json")
    a = ap.parse_args()
    alpha = alphabet(); t0 = time.time(); total = 0; bad = collections.Counter(); examples = {}
    for ctxname, ctx in CTX.items():
        role = "subj" if ctxname == "subj" else ("cop" if ctxname == "cop" else "obj")
        rules = {role: M.rules_for(role), M.of_role(role): M.rules_for(M.of_role(role))}; D = {role: M.derived(role), M.of_role(role): M.derived(M.of_role(role))}
        for L in range(1, a.max_len + 1):
            for seq in itertools.product(alpha, repeat=L):
                total += 1
                ref = reference(seq, ctx)
                if ref[0] == "OK" and ref[1] < L: continue            # the NP ended before the sequence did: a shorter case covers it
                mod = module(seq, ctx, role, rules, D)
                ok, kind = compare(ref, mod)
                if not ok:
                    key = (ctxname, kind, str(ref[:2]), str(mod[:2])); bad[key] += 1; examples.setdefault(key, " ".join(seq))
    res = {"alphabet": len(alpha), "max_len": a.max_len, "cases": total, "mismatches": sum(bad.values()), "mismatch_kinds": len(bad), "seconds": round(time.time() - t0),
           "examples": [{"context": k[0], "kind": k[1], "reference": k[2], "module": k[3], "count": n, "example": examples[k]} for k, n in bad.most_common(25)]}
    res["summary"] = f"np_module vs parse_np: {total} cases (len<={a.max_len}, {len(alpha)} tokens), {res['mismatches']} mismatches"
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(res["summary"])
    for e in res["examples"][:10]: print("  ", e)

if __name__ == "__main__":
    main()
