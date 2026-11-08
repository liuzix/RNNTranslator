"""
Weight-space edit of the English->Japanese RNN, working only from the raw dump.

Inputs: en_ja_weights.npz = A, a, B, b, h0, plus the I/O layout (input column -> token,
output unit -> (slot, token)). No scratch or state unit names are used; every internal
unit below is located by its connectivity.

The edit: "with" is currently always と (companion). Make it で (instrument) when the
PP noun is inanimate:    i wrote the letter with the pen .  ->  私はペンで手紙を書いた
"""
import numpy as np
import en_ja_rnn as m

W = m.load_weights("en_ja_weights.npz")
A, a, B, b = W["A"], W["a"], W["B"], W["b"]
in_tokens = list(W["in_tokens"]); out_units = [(int(i), int(s), t) for i, s, t in W["out_units"]]
nS = B.shape[0]
xcol = lambda tok: nS + in_tokens.index(tok)

# ---- 1. token-feature units: scratch rows that read exactly a set of input columns, bias 0
def feature_unit(tokens):
    cols = {xcol(t) for t in tokens}
    hits = [i for i in range(A.shape[0])
            if a[i] == 0 and set(np.nonzero(A[i])[0]) == cols and all(A[i, c] == 1 for c in cols)]
    assert len(hits) == 1, (tokens, hits)
    return hits[0]

animate_nouns = ["cat", "dog", "fish", "bird", "animal", "teacher", "student", "friend", "child", "man", "woman", "doctor"]
pronouns = ["i", "me", "you", "he", "him", "she", "her", "we", "us", "they", "them"]
z_anim = feature_unit(animate_nouns)
z_pron = feature_unit(pronouns)

# ---- 2. the postposition units: slot-3 outputs that emit と / で
rows = lambda tok: [i for i, s, t in out_units if s == 3 and t == tok]
to_rows, de_rows = rows("と"), rows("で")
assert len(to_rows) == 2 and len(de_rows) == 2
# The two と rows share a column set and differ in the sign of one column: that column is the
# "existence" flag (と with an action vs と with existence). The action row has it at -1.
c0, c1 = (set(np.nonzero(B[r])[0]) for r in to_rows)
assert c0 == c1
flag = [c for c in c0 if B[to_rows[0], c] != B[to_rows[1], c]]
assert len(flag) == 1
z_loc = flag[0]
to_action = to_rows[0] if B[to_rows[0], z_loc] < 0 else to_rows[1]
to_cols = c0
# The two で rows are both action rows (the flag at -1); each has one private column, its
# preposition's held register. Columns common to all rows (except the flag) are "PP has ended".
de_cols = [set(np.nonzero(B[r])[0]) for r in de_rows]
assert all(B[r, z_loc] == -1 for r in de_rows)
shared = (to_cols & de_cols[0] & de_cols[1]) - {z_loc}
assert len(shared) == 2
z_ppend = sorted(int(c) for c in shared)
z_with = (to_cols - shared - {z_loc}).pop()
privates = [(r, (cs - shared - {z_loc}).pop()) for r, cs in zip(de_rows, de_cols)]

# ---- 3. prove z_with really is the held "with": copy of a state unit written from x[with]
def register_behind(zcol):
    src = np.nonzero(A[zcol])[0]
    assert len(src) == 1 and src[0] < nS and A[zcol, src[0]] == 1 and a[zcol] == 0, "not a plain copy"
    h = src[0]                                             # the state register
    writes = [j for j in np.nonzero(B[h])[0] if B[h, j] == 1 and j != zcol]
    tokens = {in_tokens[c - nS] for j in writes for c in np.nonzero(A[j])[0] if c >= nS}
    return h, tokens
assert register_behind(z_with)[1] == {"with"}, register_behind(z_with)
by_prep = {register_behind(zc)[1].pop(): (r, zc) for r, zc in privates}
assert set(by_prep) == {"in", "at"}, by_prep
de_action, z_in = by_prep["in"]

print(f"located: PP-end events z{z_ppend}, held 'with' z[{z_with}], held 'in' z[{z_in}], "
      f"existence flag z[{z_loc}], animate-noun feature z[{z_anim}], pronoun feature z[{z_pron}]")
row = lambda M, v, r: " ".join(f"{M[r, c]:+g}*z[{c}]" for c in np.nonzero(M[r])[0]) + f" {v[r]:+g}"
print(f"before:  と = ReLU({row(B, b, to_action)})\n         で = ReLU({row(B, b, de_action)})")

# ---- 4. the edit (11 numbers)
B2, b2 = B.copy(), b.copy()
# と with an action: now also requires the PP noun to be animate (or a pronoun)
B2[to_action, z_anim] = 1; B2[to_action, z_pron] = 1; b2[to_action] = -2
# で with an action: fires for "in" regardless, or for "with" with an inanimate noun
#   ReLU(3*ppend + 2*in + with - anim - pron - 2*loc - 3)
for c in z_ppend: B2[de_action, c] = 3
B2[de_action, z_in] = 2; B2[de_action, z_with] = 1
B2[de_action, z_anim] = -1; B2[de_action, z_pron] = -1; B2[de_action, z_loc] = -2
b2[de_action] = -3
changed = np.count_nonzero(B2 != B) + np.count_nonzero(b2 != b)
print(f"after:   と = ReLU({row(B2, b2, to_action)})\n         で = ReLU({row(B2, b2, de_action)})")
print(f"{changed} numbers changed out of {A.size + B.size + a.size + b.size:,}")

W2 = dict(W); W2["B"], W2["b"] = B2, b2
np.savez("en_ja_weights_edited.npz", **W2)

# ---- 5. run the unchanged runtime on the edited weights
T = m.Translator(W2)
for s in ["i wrote the letter with the pen .", "i went with him .", "i walked with the dog .",
          "the cat is with him .", "the book is with the pen .", "i ate the bread in the park .",
          "he said that she opened the door with the old pen ."]:
    print(f"  {s}\n    -> {''.join(T.translate(s.split()))}")

# full regression against the reference with the new "with" semantics
m.WITH_INSTRUMENTAL = True
m.verify(T, n_random=5000, n_streams=200)
