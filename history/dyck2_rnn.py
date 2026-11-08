r"""
A hand-written RNN that recognizes Dyck-2: balanced strings over ( ) [ ].

Every weight below is a literal I wrote; no training, no search. Python is only
the interpreter and the verifier.

Architecture (one recurrent cell, applied per symbol):
    z_t = ReLU(A @ [h_{t-1}; x_t] + a)        # 11 scratch units
    h_t = ReLU(B @ z_t + b)                   # 8 state units (carried)
    accept  iff  w_out @ h_T + 1 > 0

Input x_t is one-hot over  (  [  )  ]  .

The stack is a single number s = h0 + h1 + h2 in [0, 2/3]:
    empty stack            s = 0
    push '('               s <- (s + 1) / 4
    push '['               s <- (s + 2) / 4
    pop                    s <- 4 s - top
    top of stack           read off the interval s lies in:
                           s in [1/4, 5/12]  -> top is '('
                           s in [1/2, 2/3]   -> top is '['
i.e. the stack is the base-4 fraction 0.d1 d2 d3 ... with digits 1 and 2.
All constants are dyadic, so float64 arithmetic is exact to depth ~24.

State units h:   0 p1   result of a '(' push (else 0)
                 1 p2   result of a '[' push (else 0)
                 2 pp   result of a pop      (else 0)
                 3 ea   \ error latch, e = ea - eb in {0, 1}
                 4 eb   /
                 5 wa   pending error: ')' popped a '['
                 6 wb   pending error: ']' popped a '('
                 7 wc   pending error: pop on empty stack

Scratch units z: 0 u1 = ReLU(64 s - 8)   \ u1 - u2 = 1 iff stack non-empty
                 1 u2 = ReLU(64 s - 9)   /
                 2 u3 = ReLU(64 s - 28)  \ u3 - u4 = 1 iff top is '['
                 3 u4 = ReLU(64 s - 29)  /
                 4 vs = s
                 5 ve = e
                 6 vw = wa + wb + wc
                 7-10   copies of the four input bits
"""
import numpy as np

# ---- Layer 1: scratch from [h0..h7, (, [, ), ]] ----------------------------
#                h0  h1  h2  h3  h4  h5  h6  h7   (   [   )   ]
A = np.array([
    [64, 64, 64,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # u1
    [64, 64, 64,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # u2
    [64, 64, 64,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # u3
    [64, 64, 64,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # u4
    [ 1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # vs
    [ 0,  0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0],   # ve
    [ 0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0],   # vw
    [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],   # x(
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],   # x[
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],   # x)
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],   # x]
], dtype=np.float64)
a = np.array([-8, -9, -28, -29, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

# ---- Layer 2: new state from [u1 u2 u3 u4 vs ve vw x( x[ x) x]] -------------
#                u1  u2  u3  u4    vs  ve  vw   x(  x[  x)  x]
B = np.array([
    [ 0,  0,  0,  0, 0.25,  0,  0,   1,  0,  0,  0],   # p1 = s/4 + 1/4      if '('
    [ 0,  0,  0,  0, 0.25,  0,  0,   0,  1,  0,  0],   # p2 = s/4 + 1/2      if '['
    [-1,  1, -1,  1,    4,  0,  0,   0,  0,  1,  1],   # pp = 4 s - top      if pop
    [ 0,  0,  0,  0,    0,  2,  1,   0,  0,  0,  0],   # ea = 2 e + pending
    [ 0,  0,  0,  0,    0,  2,  1,   0,  0,  0,  0],   # eb = 2 e + pending - 1
    [ 0,  0,  1, -1,    0,  0,  0,   0,  0,  1,  0],   # wa = ')' and top '['
    [ 1, -1, -1,  1,    0,  0,  0,   0,  0,  0,  1],   # wb = ']' and top '('
    [-1,  1,  0,  0,    0,  0,  0,   0,  0,  1,  1],   # wc = pop and empty
], dtype=np.float64)
b = np.array([-0.75, -0.5, -1, 0, -1, -1, -1, 0], dtype=np.float64)

# ---- Readout: accept iff stack empty, no latched error, no pending error ----
w_out = np.array([-8, -8, -8, -4, 4, -4, -4, -4], dtype=np.float64)
b_out = 1.0

ALPHABET = "([)]"
relu = lambda v: np.maximum(v, 0.0)


def run(batch, trace=False):
    """batch: int array (N, T) of symbol indices into ALPHABET. Returns accept mask."""
    N, T = batch.shape
    h = np.zeros((N, 8))
    rows = []
    for t in range(T):
        x = np.eye(4)[batch[:, t]]
        z = relu(np.concatenate([h, x], axis=1) @ A.T + a)
        h = relu(z @ B.T + b)
        if trace:
            s = h[0, 0] + h[0, 1] + h[0, 2]
            rows.append((ALPHABET[batch[0, t]], s, h[0, 3] - h[0, 4], h[0, 5:8].sum()))
    logit = h @ w_out + b_out
    if trace:
        return logit > 0, rows
    return logit > 0


def reference(string):
    """Independent checker: an actual list stack."""
    st = []
    for ch in string:
        if ch in "([":
            st.append(ch)
        else:
            if not st or st.pop() != {")": "(", "]": "["}[ch]:
                return False
    return not st


def encode(strings):
    return np.array([[ALPHABET.index(c) for c in s] for s in strings], dtype=np.int64)


if __name__ == "__main__":
    import itertools, random
    from math import comb

    # 1. Exhaustive: every string of length 1..8 over the 4 symbols (87,380 strings)
    total = 0
    for L in range(1, 9):
        strings = ["".join(p) for p in itertools.product(ALPHABET, repeat=L)]
        got = run(encode(strings))
        want = np.array([reference(s) for s in strings])
        assert (got == want).all(), f"mismatch at length {L}"
        total += len(strings)
        if L % 2 == 0:
            n = L // 2
            catalan = comb(2 * n, n) // (n + 1)
            assert got.sum() == catalan * 2**n, (L, got.sum())
    print(f"exhaustive: {total} strings of length <= 8 all agree; "
          f"accepted counts match Catalan(n)*2^n")

    # 2. Random valid words up to depth 22, and single-symbol corruptions of them
    def random_dyck2(n_pairs, max_depth):
        out, st = [], []
        pairs_left = n_pairs
        while pairs_left or st:
            if pairs_left and (not st or len(st) < max_depth and random.random() < 0.5):
                c = random.choice("([")
                out.append(c); st.append(c); pairs_left -= 1
            else:
                out.append({"(": ")", "[": "]"}[st.pop()])
        return "".join(out)

    random.seed(0)
    n_ok = n_bad = 0
    for _ in range(300):
        w = random_dyck2(random.randint(50, 600), max_depth=22)
        assert run(encode([w]))[0] == reference(w) == True
        n_ok += 1
        w2 = list(w); i = random.randrange(len(w2))
        w2[i] = random.choice(ALPHABET.replace(w2[i], "")); w2 = "".join(w2)
        assert run(encode([w2]))[0] == reference(w2)
        n_bad += 1
    print(f"random: {n_ok} valid words (len 100-1200, depth <= 22) accepted; "
          f"{n_bad} one-symbol corruptions judged like the reference")

    # 3. Depth stress: exact arithmetic holds to depth 24
    for d in (10, 20, 24):
        w = "([" * (d // 2) + "])" * (d // 2)
        assert run(encode([w]))[0]
        assert not run(encode([w[:-1] + ("]" if w[-1] == ")" else ")")]))[0]
    print("depth: nested to 24 accepted; last bracket swapped rejected")

    # 4. A trace
    ok, rows = run(encode(["([(])"]), trace=True)
    print("\ntrace of '([(])'  (rejected)" if not ok else "\ntrace")
    print(f"{'in':>3} {'stack s':>10} {'top':>4} {'err':>4}")
    for ch, s, e, w in rows:
        top = "-" if s == 0 else ("(" if s < 0.45 else "[")
        print(f"{ch:>3} {s:>10.6f} {top:>4} {int(e + w):>4}")
