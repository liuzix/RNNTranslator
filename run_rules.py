"""Run the rule transducer on the held-out Tanaka sentences and sort the residuals."""
import sys, math, random, re, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules_ja as R

D = os.environ.get("ENJA_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "small_parallel_enja")) + os.sep
def load(name, n=None):
    en = [l.split() for l in open(D + name + ".en", encoding="utf-8")]
    ja = ["".join(l.split()) for l in open(D + name + ".ja", encoding="utf-8")]
    pairs = [(e, j) for e, j in zip(en, ja) if e[-1] in (".", "?", "!") and not any(t in (".", "?", "!") for t in e[:-1])]
    return pairs[:n] if n else pairs

def char_bleu(hyps, refs, n=4):
    ps = []
    for k in range(1, n + 1):
        m = t = 0
        for h, r in zip(hyps, refs):
            hg = Counter(h[i:i + k] for i in range(len(h) - k + 1)); rg = Counter(r[i:i + k] for i in range(len(r) - k + 1))
            m += sum(min(c, rg[g]) for g, c in hg.items()); t += max(len(h) - k + 1, 0)
        ps.append(m / t if t else 0)
    hl, rl = sum(map(len, hyps)), sum(map(len, refs))
    bp = 1 if hl >= rl else (math.exp(1 - rl / hl) if hl else 0)
    return 100 * bp * math.exp(sum(math.log(p) if p else -99 for p in ps) / n), ps, hl / rl

def run(pairs, label, show=0, seed=0):
    hyps, fails, oov = [], Counter(), Counter()
    rows = []
    for e, j in pairs:
        unknown = [t for t in R.merge_multiword(e) if not R.classify(t) and t not in (",", "'s")]
        try:
            h = R.translate(e)
            rows.append((e, j, h, None))
        except R.ParseFail as ex:
            why = str(ex)
            h = ""
            if unknown:
                fails["unknown word"] += 1; oov.update(unknown)
            else:
                fails[why.split(" at ")[0]] += 1
            rows.append((e, j, h, why))
        hyps.append(h)
    refs = [j for _, j in pairs]
    ok = [(h, r) for (e, r, h, w) in rows if w is None]
    b_all, _, ratio = char_bleu(hyps, refs)
    b_ok, ps_ok, ratio_ok = char_bleu([h for h, _ in ok], [r for _, r in ok]) if ok else (0, [0] * 4, 0)
    print(f"{label}: {len(pairs)} sentences, {len(ok)} translated ({100*len(ok)/len(pairs):.0f}%)")
    print(f"  char-BLEU-4 over all {b_all:.1f} (length ratio {ratio:.2f}); over the translated ones {b_ok:.1f} "
          f"(1-gram {100*ps_ok[0]:.0f}%, 2-gram {100*ps_ok[1]:.0f}%, 4-gram {100*ps_ok[3]:.0f}%, length ratio {ratio_ok:.2f})")
    print("  failures by cause:", ", ".join(f"{k} {v}" for k, v in fails.most_common(14)))
    print("  most frequent unknown words:", " ".join(f"{w}" for w, _ in oov.most_common(40)))
    if show:
        rng = random.Random(seed)
        for e, j, h, w in rng.sample([r for r in rows if r[3] is None], min(show, len(ok))):
            print(f"    {' '.join(e)}\n      ref {j}\n      out {h}")
        print("  sample failures:")
        for e, j, h, w in rng.sample([r for r in rows if r[3] is not None], min(show, len(pairs) - len(ok))):
            print(f"    {' '.join(e)}\n      ref {j}\n      why {w}")
    return rows

if __name__ == "__main__":
    dev = load("dev")
    run(dev, "held-out", show=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
