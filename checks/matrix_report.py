import os, sys, collections, pickle, numpy as np; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parser_matrix as PM, rules_ja as R, run_rules as H
from parser_rules import Parser, ParseFail
def rows_for(split):
    out = {}
    for en, ja in H.load(split):
        toks = R.merge_multiword(en if isinstance(en, list) else en.split())
        try: Parser(list(toks), []).parse(); rows, final = PM.event_rows_pred(toks, [])
        except (ParseFail, PM.OutOfScope): continue
        except Exception: continue
        out[tuple(toks)] = rows
    return out
dev, test = rows_for("dev"), rows_for("test")
trees = PM.induce([r for rs in list(dev.values()) + list(test.values()) for r in rs])
comp = PM.compile_matrices(trees); mp = PM.MatrixParser(comp)
S, X = comp["S"], comp["X"]
print("STATE", mp.nS, "units:", collections.Counter(n[0] for n in S.names))
slots = collections.Counter(n[1] for n in S.names if n[0] == "slot"); print("   slot blocks:", len(slots), "largest:", slots.most_common(6))
print("INPUT", mp.nX, "units:", collections.Counter(n[0] for n in X.names))
names = ["L1 derived register predicates (OR)", "L2 pp count (AND/NOT)", "L3 rules (AND of control + literals)", "L4 next-control and gates (OR)", "L5 keep / write (AND, AND-NOT)", "W6 new state (sum)"]
tot = 0
for (M, b), nm in zip(mp.mats + [(mp.W6, None)], names):
    vals = collections.Counter(np.round(M.data).astype(int)); tot += M.nnz
    print(f"   {nm:42s} shape {M.shape}  nnz {M.nnz:6d}  weights {dict(sorted(vals.items()))}" + (f"  biases {dict(sorted(collections.Counter(np.round(b).astype(int)).items()))}" if b is not None else ""))
print("total nnz", tot, "| density", f"{tot / sum(M.shape[0]*M.shape[1] for M, _ in mp.mats + [(mp.W6, None)]):.2e}")
rules = comp["rules"]; lits = [len(l) for _, l, _ in rules]
print(f"rules {len(rules)}: literals per rule mean {np.mean(lits):.1f} max {max(lits)}; negative literals {sum(1 for _, l, _ in rules for q, t in l if not t)} of {sum(lits)}")
L5 = comp["layers"][4][0]; print("L5 units:", collections.Counter(n[0] for n in L5.names))
# generalization: compile on dev, run on test
trees_dev = PM.induce([r for rs in dev.values() for r in rs]); mp_dev = PM.MatrixParser(PM.compile_matrices(trees_dev))
same = diff = 0; kinds = collections.Counter()
for toks in test:
    want = R.generate(Parser(list(toks), []).parse())
    try: got = R.generate(mp_dev.parse(list(toks))); same += got == want; diff += got != want; kinds["wrong frame" if got != want else "ok"] += 1
    except Exception as e: diff += 1; kinds[type(e).__name__ + ": " + str(e).split(" at ")[0][:30]] += 1
print(f"compiled on dev ({len(dev)} sentences, {mp_dev.nnz()} nnz), run on test ({len(test)} in-scope): {same} identical, {diff} not — {dict(kinds.most_common(4))}")
pickle.dump({"trees": trees, "rules": rules}, open("parser_matrix.pkl", "wb"))
