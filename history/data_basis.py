"""
Experiment: write RNN weights that fit 1,000 English-Japanese sentence pairs, assuming
nothing from linguistics. Two networks, same ReLU runtime, both written from the data.

  Stage 1  "no matter what": the corpus as a matrix.   Exact fit, zero generalization.
  Stage 2  the data's own basis: lexicon = alignment counts, word order = where each
           English word's counterpart lands, "particles" = Japanese tokens no English
           word explains, inserted by Japanese successor statistics.  Measured held-out.

Data: Tanaka Corpus (public domain) via odashi/small_parallel_enja, pre-tokenized.
"""
import sys, math, random
from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sp

D = "data/small_parallel_enja/"
def load(name, n=None):
    en = [l.split() for l in open(D + name + ".en", encoding="utf-8")]
    ja = [l.split() for l in open(D + name + ".ja", encoding="utf-8")]
    pairs = list(zip(en, ja))
    return pairs[:n] if n else pairs

train = load("train", 1000)
dev = load("dev")
# duplicates on the English side cannot be fit by any function; keep the first
seen, uniq = set(), []
for e, j in train:
    k = " ".join(e)
    if k not in seen:
        seen.add(k); uniq.append((e, j))
one_sentence = lambda e: e[-1] in (".", "?", "!") and not any(t in (".", "?", "!") for t in e[:-1])
train = [(e, j) for e, j in uniq if one_sentence(e)]
dev = [(e, j) for e, j in dev if one_sentence(e)]
print(f"fit set {len(train)} pairs (after dropping {1000 - len(train)} duplicates / odd endings), held-out {len(dev)}")

EN = sorted({w for e, _ in train for w in e} | {w for e, _ in dev for w in e})
JA = sorted({t for _, j in train for t in j})
en_idx = {w: i for i, w in enumerate(EN)}
MAXLEN = 17                      # English tokens per sentence including the final "."
ENDS = [t for t in (".", "?", "!") if t in en_idx]
assert all(e[-1] in ENDS for e, _ in train + dev)
Q = 24                           # target-position bins for stage 2
assert all(len(e) <= MAXLEN for e, _ in train + dev)

relu = lambda v: np.maximum(v, 0.0)

# ----------------------------------------------------------------------------
# A sparse ReLU RNN: z = ReLU(A [h; x] + a), h' = ReLU(B z + b); outputs = named state units
# ----------------------------------------------------------------------------
class Net:
    def __init__(self, n_state, n_scratch, n_in):
        self.nS, self.nZ, self.nX = n_state, n_scratch, n_in
        self.Ar, self.Ac, self.Av = [], [], []
        self.Br, self.Bc, self.Bv = [], [], []
        self.a = np.zeros(n_scratch); self.b = np.zeros(n_state)
        self.h0 = np.zeros(n_state)
        self.out = {}            # state index -> (sort key, token)
    def A1(self, z, col, v=1.0):          # layer-1 weight; col indexes [state; input]
        self.Ar.append(z); self.Ac.append(col); self.Av.append(v)
    def B2(self, h, z, v=1.0):
        self.Br.append(h); self.Bc.append(z); self.Bv.append(v)
    def finish(self):
        self.A = sp.csr_matrix((self.Av, (self.Ar, self.Ac)), shape=(self.nZ, self.nS + self.nX))
        self.B = sp.csr_matrix((self.Bv, (self.Br, self.Bc)), shape=(self.nS, self.nZ))
        self.A.sum_duplicates(); self.B.sum_duplicates()
        self.out_idx = np.array(sorted(self.out), dtype=np.int64)
        self.out_key = [self.out[i] for i in self.out_idx]
        del self.Ar, self.Ac, self.Av, self.Br, self.Bc, self.Bv
        return self
    def nnz(self):
        return self.A.nnz + self.B.nnz
    def translate(self, tokens):
        h = self.h0.copy(); emitted = []
        for t in tokens:
            x = np.zeros(self.nX); x[en_idx[t]] = 1.0
            z = relu(self.A @ np.concatenate([h, x]) + self.a)
            h = relu(self.B @ z + self.b)
            on = self.out_idx[h[self.out_idx] > 0.5]
            emitted += sorted(((self.out[i][0], self.out[i][1]) for i in on))
        return [tok for _, tok in emitted]

# ----------------------------------------------------------------------------
# Stage 1: the corpus as a matrix
# ----------------------------------------------------------------------------
def build_memorizer(pairs):
    V = len(EN); P = MAXLEN
    # state: pos[P] | occ[P] (position holds a token) | stored[P*V] | out[maxJ * |JA|]
    maxJ = max(len(j) for _, j in pairs)
    S_pos, S_occ, S_stored = 0, P, 2 * P
    S_out = S_stored + P * V
    nS = S_out + maxJ * len(JA)
    ja_idx = {t: i for i, t in enumerate(JA)}
    # scratch: copies of pos, occ, stored | write[P*V] | match[n] | end
    Z_cpos, Z_cocc, Z_cst = 0, P, 2 * P
    Z_wr = Z_cst + P * V; Z_match = Z_wr + P * V
    Z_end = Z_match + len(pairs)
    nZ = Z_end + 1
    net = Net(nS, nZ, V)
    net.h0[S_pos + 0] = 1.0
    # layer 1
    for p in range(P):
        net.A1(Z_cpos + p, S_pos + p); net.A1(Z_cocc + p, S_occ + p)
        for w in range(V):
            net.A1(Z_cst + p * V + w, S_stored + p * V + w)
            net.A1(Z_wr + p * V + w, S_pos + p); net.A1(Z_wr + p * V + w, nS + w)
    net.a[Z_wr:Z_wr + P * V] = -1
    for t in ENDS:
        net.A1(Z_end, nS + en_idx[t])
    for i, (e, j) in enumerate(pairs):
        words = e[:-1]                         # the final ". ? !" is the trigger, not stored
        L = len(words)
        for p, w in enumerate(words):
            net.A1(Z_match + i, S_stored + p * V + en_idx[w])
        for p in range(L, P):                  # no tokens beyond the sentence
            net.A1(Z_match + i, S_occ + p, -1.0)
        net.a[Z_match + i] = -(L - 1)
    # layer 2
    net.B2(S_pos + 0, Z_end)
    for p in range(P - 1):
        net.B2(S_pos + p + 1, Z_cpos + p); net.B2(S_pos + p + 1, Z_end, -1.0)
    for p in range(P):
        net.B2(S_occ + p, Z_cocc + p); net.B2(S_occ + p, Z_end, -1.0)
        for w in range(V):
            net.B2(S_occ + p, Z_wr + p * V + w)
    for k in range(P * V):
        net.B2(S_stored + k, Z_cst + k); net.B2(S_stored + k, Z_wr + k); net.B2(S_stored + k, Z_end, -1.0)
    for i, (e, j) in enumerate(pairs):
        for k, t in enumerate(j):
            u = S_out + k * len(JA) + ja_idx[t]
            net.B2(u, Z_match + i)
            net.b[u] = -1; net.out[u] = (k, t)
    for u in net.out:
        net.B2(u, Z_end)
    return net.finish()

# ----------------------------------------------------------------------------
# Stage 2: the data's own basis
# ----------------------------------------------------------------------------
def ibm1(pairs, iters=8):
    """p(ja | en) by expectation-maximization over co-occurrence; NULL is an English word."""
    t = defaultdict(lambda: 1.0)
    for _ in range(iters):
        count = defaultdict(float); total = defaultdict(float)
        for e, j in pairs:
            src = ["<NULL>"] + e
            for f in j:
                z = sum(t[(f, w)] for w in src)
                for w in src:
                    c = t[(f, w)] / z
                    count[(f, w)] += c; total[w] += c
        t = defaultdict(lambda: 1e-9, {k: v / total[k[1]] for k, v in count.items()})
    return t

def analyze(pairs):
    t = ibm1(pairs)
    aligned = defaultdict(Counter)       # en word -> Counter of ja tokens it generates
    offsets = defaultdict(lambda: defaultdict(list))   # en word -> ja token -> relative offsets
    explains = defaultdict(Counter)      # ja token -> Counter of the English words it aligned to
    occ = Counter()                      # en word occurrences
    ja_count = Counter(); succ = defaultdict(Counter); final = defaultdict(Counter)
    for e, j in pairs:
        src = ["<NULL>"] + e[:-1]
        final[e[-1]][j[-1]] += 1
        for w in e: occ[w] += 1
        for k, f in enumerate(j[:-1]):
            ja_count[f] += 1
            if k + 1 < len(j) - 1: succ[f][j[k + 1]] += 1
            best = max(src, key=lambda w: t[(f, w)])
            explains[f][best] += 1
            if best == "<NULL>":
                continue
            p = max(i for i, w in enumerate(e) if w == best)
            aligned[best][f] += 1
            offsets[best][f].append(k / len(j) - p / len(e))
    # a Japanese token no English word explains: its alignments are spread over many words
    inserts = {f for f in ja_count if ja_count[f] >= 10
               and explains[f].most_common(1)[0][1] / ja_count[f] < 0.25}
    lexicon, delta = {}, {}
    for w in occ:
        if occ[w] < 2 or not aligned[w]:
            continue
        toks = [f for f, c in aligned[w].most_common(2) if c / occ[w] >= 0.3 and f not in inserts]
        if toks:
            toks.sort(key=lambda f: np.median(offsets[w][f]))      # in the order they land
            lexicon[w] = toks
            delta[w] = float(np.median(offsets[w][toks[0]]))
    successor = {}
    for f, cnt in succ.items():
        u, c = cnt.most_common(1)[0]
        if u in inserts and c / ja_count[f] >= 0.4:
            successor[f] = u
    finals = {t: cnt.most_common(1)[0][0] for t, cnt in final.items()}
    return dict(t=t, lexicon=lexicon, delta=delta, inserts=inserts, successor=successor, finals=finals)

def build_basis(st):
    V = len(EN); P = MAXLEN
    lex, delta, succ = st["lexicon"], st["delta"], st["successor"]
    toks = sorted({f for fs in lex.values() for f in fs} | set(succ.values()) | set(st["finals"].values()))
    ja_idx = {t: i for i, t in enumerate(toks)}; T = len(toks)
    SUB = 4
    slot = lambda p, w, L: min(Q - 1, max(0, int(round((Q - 1) * (p / L + delta[w])))))
    # state: pos[P] | stored[P*V] | out[Q*P*SUB*T]
    S_pos, S_stored, S_out = 0, P, P + P * V
    nS = S_out + Q * P * SUB * T
    # scratch: copies | write[P*V] | g[P*V*P] (stored AND final length) | end | end-by-token
    Z_cpos, Z_cst, Z_wr, Z_g = 0, P, P + P * V, P + 2 * P * V
    Z_end = Z_g + P * V * P
    Z_endtok = Z_end + 1
    nZ = Z_endtok + len(ENDS)
    net = Net(nS, nZ, V)
    net.h0[S_pos + 0] = 1.0
    for p in range(P):
        net.A1(Z_cpos + p, S_pos + p)
        for w in range(V):
            net.A1(Z_cst + p * V + w, S_stored + p * V + w)
            net.A1(Z_wr + p * V + w, S_pos + p); net.A1(Z_wr + p * V + w, nS + w)
    net.a[Z_wr:Z_wr + P * V] = -1
    for k, t in enumerate(ENDS):
        net.A1(Z_end, nS + en_idx[t]); net.A1(Z_endtok + k, nS + en_idx[t])
    net.B2(S_pos + 0, Z_end)
    for p in range(P - 1):
        net.B2(S_pos + p + 1, Z_cpos + p); net.B2(S_pos + p + 1, Z_end, -1.0)
    for k in range(P * V):
        net.B2(S_stored + k, Z_cst + k); net.B2(S_stored + k, Z_wr + k); net.B2(S_stored + k, Z_end, -1.0)
    out_unit = lambda q, p, sub, t: S_out + ((q * P + p) * SUB + sub) * T + ja_idx[t]
    # g[p,w,L] = stored[p,w] AND pos[L]: the word w sat at position p of a sentence of length L
    for w, fs in lex.items():
        wi = en_idx[w]
        chain = list(fs)
        while chain[-1] in succ and len(chain) < SUB:
            chain.append(succ[chain[-1]])
        for p in range(P):
            for L in range(p + 1, P):
                g = Z_g + (p * V + wi) * P + (L - 1)
                net.A1(g, S_stored + p * V + wi); net.A1(g, S_pos + L)
                net.a[g] = -1
                q = slot(p, w, L)
                for sub, tok in enumerate(chain):
                    u = out_unit(q, p, sub, tok)
                    net.B2(u, g); net.b[u] = -1
                    net.out[u] = ((q, p, sub), tok)
    for u in set(net.out):
        net.B2(u, Z_end)                   # outputs only fire at the end
    for k, t in enumerate(ENDS):
        if t in st["finals"]:
            u = out_unit(Q - 1, P - 1, SUB - 1, st["finals"][t])
            net.B2(u, Z_endtok + k); net.out[u] = ((Q, P, SUB), st["finals"][t])
    return net.finish()

# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def bleu(hyps, refs, n=2):
    precisions = []
    for k in range(1, n + 1):
        match = total = 0
        for h, r in zip(hyps, refs):
            hg = Counter(tuple(h[i:i + k]) for i in range(len(h) - k + 1))
            rg = Counter(tuple(r[i:i + k]) for i in range(len(r) - k + 1))
            match += sum(min(c, rg[g]) for g, c in hg.items()); total += max(len(h) - k + 1, 0)
        precisions.append(match / total if total else 0)
    hl = sum(len(h) for h in hyps); rl = sum(len(r) for r in refs)
    bp = 1.0 if hl >= rl else math.exp(1 - rl / hl) if hl else 0
    return bp * math.exp(sum(math.log(p) if p else -99 for p in precisions) / n), precisions, hl / rl

def report(name, net, pairs):
    hyps = [net.translate(e) for e, _ in pairs]
    refs = [j for _, j in pairs]
    exact = sum(h == r for h, r in zip(hyps, refs))
    b, prec, ratio = bleu(hyps, refs)
    print(f"  {name}: exact {exact}/{len(pairs)}, BLEU-2 {100*b:.1f} (1-gram {100*prec[0]:.0f}%, "
          f"2-gram {100*prec[1]:.0f}%), length ratio {ratio:.2f}")
    return hyps

if __name__ == "__main__":
    print("\nStage 1: the corpus as a matrix")
    mem = build_memorizer(train)
    print(f"  {mem.nS:,} state units, {mem.nZ:,} scratch units, {mem.nnz():,} nonzero weights")
    report("fit set ", mem, train)
    report("held-out", mem, dev)

    print("\nStage 2: the data's own basis")
    st = analyze(train)
    print(f"  lexicon {len(st['lexicon'])} English words with a Japanese counterpart, "
          f"{len(st['inserts'])} unexplained Japanese tokens: {' '.join(sorted(st['inserts']))}")
    print(f"  successor rules {len(st['successor'])}, final tokens {st['finals']}")
    basis = build_basis(st)
    print(f"  {basis.nS:,} state units, {basis.nZ:,} scratch units, {basis.nnz():,} nonzero weights")
    report("fit set ", basis, train)
    hyps = report("held-out", basis, dev)
    cov = [w in st["lexicon"] or w in ENDS for e, _ in dev for w in e]
    print(f"  held-out English tokens with a lexicon entry: {100*sum(cov)/len(cov):.0f}%")
    # same lexicon, source order (delta = 0): what the data-derived order is worth
    st0 = dict(st); st0["delta"] = {w: 0.0 for w in st["delta"]}
    report("held-out, source order", build_basis(st0), dev)

    print("\n  held-out samples:")
    rng = random.Random(1)
    for e, j in rng.sample(dev, 8):
        print(f"    {' '.join(e)}\n      ref {''.join(j)}\n      out {''.join(basis.translate(e))}")

    # Does the data-derived position offset know what a verb is?
    verbs = set("""is are was were be been am do does did have has had go went come came get got
      make made take took see saw know knew think thought want like look looked give gave tell told
      say said eat ate read write wrote play played run ran buy bought find found help keep leave
      left live put speak study stay use work call find start need learn meet try watch wait""".split())
    nouns = set("""father mother friend friends book books house school car money time day night morning
      story teacher doctor dog train station room class sister brother child children people man
      woman boy girl water tea party letter work life game news home english hand tomorrow
      yesterday today week year""".split())
    func = set("""the a an to of in at on for with by from and or but as my your his her our their
      this that these those it i you he she we they me him us them not""".split())
    rows = []
    for w, d in st["delta"].items():
        cls = "verb" if w in verbs else "noun" if w in nouns else "function" if w in func else None
        if cls: rows.append((cls, w, d))
    print("\n  data-derived offset of the Japanese counterpart (relative position, + = later):")
    for cls in ("verb", "noun", "function"):
        ds = sorted(d for c, _, d in rows if c == cls)
        if ds:
            print(f"    {cls:9s} n={len(ds):3d}  median {np.median(ds):+.2f}  "
                  f"quartiles {np.percentile(ds, 25):+.2f} .. {np.percentile(ds, 75):+.2f}")
    print("    verbs by offset:", " ".join(f"{w}{d:+.2f}" for c, w, d in sorted(rows, key=lambda r: r[2]) if c == "verb"))
    print("    nouns by offset:", " ".join(f"{w}{d:+.2f}" for c, w, d in sorted(rows, key=lambda r: r[2]) if c == "noun"))
