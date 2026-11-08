import os, sys, collections, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parser as P, parser_matrix as PM, rules_ja as R, run_rules as H
from parser_rules import ParseFail
char_bleu = H.char_bleu
pairs = {s: H.load(s) for s in ("dev", "test")}
sents = {s: [R.merge_multiword(en if isinstance(en, list) else en.split()) for en, ja in pairs[s]] for s in pairs}
rp = P.RuleParser()
def crosstab(mp, split, label):
    cnt = collections.Counter(); hyp_r, hyp_m, ref = [], [], []
    for toks, (en, ja) in zip(sents[split], pairs[split]):
        try: want = rp.translate(toks)
        except ParseFail: want = None
        try: got = mp.translate(toks)
        except Exception: got = None
        if want is None: key = "rules refuse / matrix refuses" if got is None else "rules refuse / matrix parses"
        else: key = "both parse, identical" if got == want else ("rules parse / matrix refuses" if got is None else "both parse, different")
        cnt[key] += 1
        if want is not None and got is not None: hyp_r.append(want); hyp_m.append(got); ref.append(ja if isinstance(ja, str) else " ".join(ja))
    b_r = char_bleu(hyp_r, ref)[0] if ref else 0
    b_m = char_bleu(hyp_m, ref)[0] if ref else 0
    print(f"{label} [{split}, {len(sents[split])} sentences]: {dict(cnt.most_common())}; on the {len(ref)} both parse: BLEU rules {b_r:.1f} matrix {b_m:.1f}")
    return cnt
t0 = time.time(); mp_all = P.build_matrix_parser(sents["dev"] + sents["test"]); t1 = time.time()
print(f"compiled on dev+test in {t1-t0:.0f}s: state {mp_all.nS}, input {mp_all.nX}, nnz {mp_all.nnz()}, rules {len(mp_all.c['rules'])} ({len(mp_all.c['rules']) - sum(1 for r in mp_all.c['rules'] if isinstance(r[0], tuple) and isinstance(r[0][0], str) and r[0][0].startswith('NP'))} outer + module)")
for s in ("dev", "test"): crosstab(mp_all, s, "hybrid (compiled on both)")
t0 = time.time(); n = 0
for toks in sents["test"][:100]:
    try: mp_all.parse(toks)
    except Exception: pass
    n += 1
print(f"matrix parse speed: {(time.time()-t0)/n*1000:.0f} ms/sentence")
mp_dev = P.build_matrix_parser(sents["dev"])
crosstab(mp_dev, "test", "hybrid (compiled on dev only)")
