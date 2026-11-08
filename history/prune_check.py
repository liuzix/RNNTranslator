"""
Is the sparse compile irredundant?  For every nonzero weight, search for a witness: a sentence
whose translation changes when that one weight is zeroed.  Weights without a witness are
removed and the network is re-verified.  What remains is 1-minimal for the fragment's language.
"""
import os, sys, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compile_sparse as C, en_ja_rnn as M

S = C.Sparse(); n = S.net
rng = random.Random(1)
sigs = M.all_clause_sigs(); outers = [None] + [(s, p, q) for s in ("pron", "N", "AN") for p in (0, 1) for q in (0, 1)]
sents = [M.gen_sentence(o, s, rng) for o in outers for s in sigs] + [M.gen_sentence(rng.choice(outers), rng.choice(sigs), rng) for _ in range(1500)]
# every lexical entry in every role, so no edge is "dead" only because the sample missed its word
art = M.article
for nn in M.NOUNS:
    for s in (["the", nn, "ate", "the", "apple", "."], ["i", "saw", "the", nn, "."], ["i", "went", "to", "the", nn, "."],
              ["the", "cat", "is", art(nn), nn, "."], ["the", "old", nn, "ran", "."], ["i", "saw", "the", "old", nn, "."],
              ["i", "ran", "with", "the", "old", nn, "."], ["the", nn, "is", "old", "."], ["the", "cat", "is", "the", "old", nn, "."],
              ["the", nn, "is", "in", "the", "park", "."], ["the", nn, "was", "not", "at", "the", "station", "."],
              ["i", "think", "that", "the", nn, "is", "red", "."], ["i", "think", "that", "the", nn, "was", "not", "in", "the", "room", "."],
              ["i", "played", "the", nn, "."], ["i", "did", "not", "play", "the", nn, "in", "the", "park", "."],
              ["he", "said", "that", "the", "cat", "is", "the", "old", nn, "."]):
        sents.append(s)
for a in M.ADJS:
    for s in (["the", a, "cat", "ran", "."], ["the", a, "teacher", "ran", "."], ["i", "bought", "the", a, "car", "."],
              ["i", "saw", "the", a, "doctor", "."], ["i", "bought", "the", a, "car", "in", "the", a, "store", "."],
              ["the", "car", "is", a, "."], ["the", "car", "was", "not", a, "."], ["the", "teacher", "is", a, "."],
              ["the", "teacher", "was", "not", a, "."], ["he", "said", "that", "the", "car", "was", a, "."],
              ["he", "said", "that", "the", "friend", "was", a, "."], ["the", "cat", "is", "the", a, "dog", "."],
              ["i", "think", "that", "the", "cat", "is", "the", a, "teacher", "."]):
        sents.append(s)
for pid, (sf, of, _) in M.PRONOUNS.items():
    cop = "is" if pid in ("3m", "3f") else ("am" if pid == "1s" else "are"); was = "was" if pid in ("3m", "3f", "1s") else "were"
    for s in ([sf, "ran", "."], ["i", "saw", of, "."], ["i", "went", "with", of, "."], ["i", "saw", of, "with", of, "."],
              ["i", "know", "that", sf, "slept", "."], [sf, cop, "old", "."], [sf, was, "in", "the", "park", "."],
              ["i", "played", "tennis", "with", of, "."], [sf, cop, "not", "a", "teacher", "."]):
        sents.append(s)
for l in M.VERBS:
    base, s3, past, _, comp = M.VERBS[l]
    tail = ["that", "i", "slept", "."] if comp else ["."]
    for s in (["he", s3] + tail, ["they", base] + tail, ["he", past] + tail, ["he", "does", "not", base] + tail,
              ["he", "did", "not", base] + tail, ["i", "think", "that", "he", past] + (["."] if not comp else ["that", "i", "slept", "."])[:1 if not comp else 4]):
        if not (comp and s[0] == "i"): sents.append(s)
for p in M.PREPS:
    for s in (["i", "ran", p, "the", "park", "."], ["i", "ate", "the", "bread", p, "the", "park", "."],
              ["the", "cat", "is", p, "the", "park", "."], ["the", "book", "is", p, "the", "park", "."], ["i", "ran", p, "him", "."]):
        sents.append(s)
sents = [s for s in sents if s[-1] == "."]
refs = []
keep = []
for e in sents:
    try: refs.append("".join(M.ref_translate(e))); keep.append(e)
    except Exception: pass
sents = keep
print(f"{len(sents)} sentences")

def run(net, tokens, record=None):
    h = S.h0.copy(); out = []
    for t in tokens:
        x = np.zeros(S.nX); x[n["X"].idx["x:" + t]] = 1.0
        z1 = C.relu(net["F"] @ x + n["f"])
        v2 = np.concatenate([h, z1, x])
        z2 = C.relu(net["G"] @ v2 + n["g"])
        v3 = np.concatenate([h, z1, z2, x])
        h = C.relu(net["H"] @ v3 + n["h"])
        if record is not None: record.append((x > 0.5, v2 > 0.5, v3 > 0.5))
        fired = sorted((S.ok[j][0], S.ok[j][1]) for j in range(len(S.oi)) if h[S.oi[j]] > 0.5)
        out += [tok for _, tok in fired]
    return "".join(out)

assert all(run(n, e) == r for e, r in zip(sents, refs)), "base network must match the reference"
t0 = time.time()
mats = {"F": n["F"].tocoo(), "G": n["G"].tocoo(), "H": n["H"].tocoo()}
# candidates per edge: sentences on which the edge is live (source active and, for a positive
# edge, target active at the same step); negative edges need only the source active
cands = {k: [[] for _ in range(m.nnz)] for k, m in mats.items()}
for si, e in enumerate(sents):
    rec = []; run(n, e, rec)
    for k, (m, si_, di_) in {"F": (mats["F"], 0, None), "G": (mats["G"], 1, None), "H": (mats["H"], 2, None)}.items():
        live = np.zeros(m.nnz, bool)
        for st in rec:
            src = st[si_]
            dst = {"F": st[1][len(S.h0):len(S.h0) + S.nZ1], "G": st[2][len(S.h0) + S.nZ1:len(S.h0) + S.nZ1 + S.nZ2], "H": None}[k]
            if k == "H":
                dst = None
            l = src[m.col]
            if dst is not None: l &= (dst[m.row] | (m.data < 0))
            live |= l
        for j in np.nonzero(live)[0]:
            if len(cands[k][j]) < 60: cands[k][j].append(si)
print(f"activity recorded: {time.time()-t0:.0f}s")

t0 = time.time(); witness = {}; tested = 0
for k, m in mats.items():
    for j in range(m.nnz):
        found = None
        net2 = dict(n); mm = m.copy(); mm.data = mm.data.copy(); mm.data[j] = 0; net2[k] = mm.tocsr()
        cs = list(cands[k][j]); rng.shuffle(cs)
        for si in cs:
            tested += 1
            if run(net2, sents[si]) != refs[si]:
                found = si; break
        witness[(k, j)] = found
print(f"witness search: {tested} translations in {time.time()-t0:.0f}s")
nowit = [(k, j) for (k, j), w in witness.items() if w is None]
print(f"edges with a witness (certified necessary): {sum(w is not None for w in witness.values())}; without: {len(nowit)}")
# the rest may be jointly necessary (redundant pairs): remove them one at a time, keeping a removal
# only if the whole verification set still matches
t0 = time.time()
cur = {k: mats[k].copy() for k in mats}
for k in cur: cur[k].data = cur[k].data.copy()
vset = list(range(0, len(sents), 4))          # every 4th sentence, ~1,500, for the greedy pass
def ok(nets):
    return all(run(nets, sents[si]) == refs[si] for si in vset)
removed = []
for k, j in nowit:
    saved = cur[k].data[j]; cur[k].data[j] = 0
    nets = {kk: cur[kk].tocsr() for kk in cur}
    if ok(nets): removed.append((k, j))
    else: cur[k].data[j] = saved
print(f"greedy pass: {len(removed)} of {len(nowit)} removable jointly ({time.time()-t0:.0f}s)")
net3 = {}
for k in cur:
    mm = cur[k].copy(); mm.eliminate_zeros(); net3[k] = mm.tocsr()
bad = sum(run(net3, e) != r for e, r in zip(sents, refs))
print(f"final: {sum(net3[k].nnz for k in mats)} nonzeros, mismatches on all {len(sents)} sentences: {bad}")
names = {"F": n["Z1"].names, "G": n["Z2"].names, "H": n["Hs"].names}
from collections import Counter
print("removed edges by target unit (top 20):")
for (k, nm), c in Counter((k, names[k][mats[k].row[j]].split(":")[0] + ":" + names[k][mats[k].row[j]].split(":")[1] if ":" in names[k][mats[k].row[j]] else names[k][mats[k].row[j]]) for k, j in removed).most_common(20):
    print(f"   {c:4d}  {k} {nm}")
import pickle; pickle.dump(dict(net3=net3, removed=removed, nowit=nowit, witness=witness), open("pruned.pkl", "wb"))
