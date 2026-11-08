import os, sys, collections, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parser_matrix as PM, rules_ja as R, run_rules as H
from parser_rules import Parser, ParseFail
pairs = H.load("dev") + H.load("test")
sents = [R.merge_multiword(en if isinstance(en, list) else en.split()) for en, ja in pairs]
t0 = time.time(); comp = PM.build_hybrid(sents, verbose=True); t1 = time.time()
print(f"built in {t1-t0:.0f}s; module log:", {k: v for k, v in PM.hybrid_rows.module_log.items() if not isinstance(k, tuple)})
for k, v in PM.hybrid_rows.module_log.items():
    if isinstance(k, tuple): print("   ", v, k)
print("rules:", len(comp["rules"]), "unresolved:", dict(comp["unresolved"].most_common(6)), "dead conj:", comp["dead_conj"])
mp = PM.MatrixParser(comp)
print(f"state {mp.nS} input {mp.nX} layers {[len(L) for L, _, _ in comp['layers']]} nnz {mp.nnz()}")
# reference: the rule parser with default choices (what was extracted)
cnt = collections.Counter(); ex = collections.defaultdict(list)
for toks in sents:
    try: want = R.generate(Parser(list(toks), []).parse())
    except ParseFail: want = None
    try: got_r = R.generate(PM.frame_to_clause(PM.run_rules(comp["rules"], comp["derived"], toks)))
    except ParseFail as e: got_r = ("FAIL", str(e).split(" at ")[0][:40])
    except Exception as e: got_r = ("EXC", type(e).__name__ + str(e)[:40])
    try: got_m = R.generate(mp.parse(list(toks)))
    except ParseFail as e: got_m = ("FAIL", str(e).split(" at ")[0][:40])
    except Exception as e: got_m = ("EXC", type(e).__name__ + str(e)[:40])
    if (got_r if not isinstance(got_r, tuple) else "refuse") != (got_m if not isinstance(got_m, tuple) else "refuse"): cnt["interpreter != matrix"] += 1; ex["interpreter != matrix"].append((toks, got_r, got_m))
    if want is None:
        key = "rules refuse / matrix refuses" if isinstance(got_m, tuple) else "rules refuse / matrix parses"
    else:
        key = "both parse, identical" if got_m == want else ("rules parse / matrix refuses" if isinstance(got_m, tuple) else "both parse, different")
    cnt[key] += 1; ex[key].append((toks, want, got_m))
print(dict(cnt.most_common()))
for key in ("interpreter != matrix", "both parse, different", "rules parse / matrix refuses"):
    print("==", key)
    sub = collections.Counter(str(g)[:60] if isinstance(g, tuple) else "frame" for _, _, g in ex[key]); print("   ", dict(sub.most_common(6)))
    for toks, want, got in ex[key][:6]: print("    ", " ".join(toks), "|", want, "|", got)
import pickle; pickle.dump(comp, open("parser_matrix.pkl", "wb"))
