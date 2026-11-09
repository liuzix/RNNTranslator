"""The compiled cell as a PyTorch module: the same five sparse layers and the readout, held as fixed buffers (not
parameters), stepped with torch.sparse.mm and ReLU.  Nothing is learned; the module exists so the circuit runs under a
real tensor framework, batched, exportable, and comparable step for step with the scipy MatrixParser.

The arithmetic is small integers — weights ±1 (and the −5 that silences a rule), biases integer thresholds, activations
0/1 and small counts — so single precision is exact and the two runners agree bit for bit at every step: pinned by the
`torch` action (harness/eval_torch.py), which steps both runners on every dev and test sentence.

    uv run python -c "import cell_torch; cell_torch.demo()"       # needs torch importable in the environment
"""
import numpy as np

def _torch():
    import torch
    return torch

def csr_to_torch(M, dtype=None):
    """a scipy CSR matrix -> a torch sparse CSR tensor (float32 by default)"""
    torch = _torch(); dtype = dtype or torch.float32
    M = M.tocsr(); M.sort_indices()
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int64)), torch.from_numpy(M.indices.astype(np.int64)),
                                   torch.from_numpy(M.data.astype(np.float32)).to(dtype), size=M.shape)

class TorchCell:
    """one step of the cell: acts = [s; x] -> s' , as sparse matrix products.  Built from a MatrixParser's matrices so
    the two runners share one compile.  (A torch.nn.Module in all but inheritance, so that importing this file does not
    import torch; `module()` returns the nn.Module with the matrices registered as buffers.)"""
    def __init__(self, mp):
        torch = _torch()
        self.mp = mp
        self.layers = [(csr_to_torch(M), torch.from_numpy(bias.astype(np.float32))) for M, bias in mp.mats]
        self.W6 = csr_to_torch(mp.W6)
        self.nS, self.nX, self.width = mp.nS, mp.nX, mp.width

    def forward(self, acts):
        """acts: (width_in, B) column batch of [s; x] -> (nS, B) next states"""
        torch = _torch()
        for M, bias in self.layers:
            h = torch.relu(torch.sparse.mm(M, acts) + bias[:, None]); acts = torch.cat([acts, h], 0)
        return torch.relu(torch.sparse.mm(self.W6, acts))

    def module(self):
        """the same cell as a torch.nn.Module with the matrices as buffers (for export, device moves, or later training)"""
        torch = _torch()
        class Cell(torch.nn.Module):
            def __init__(m, tc):
                super().__init__()
                for k, (M, b) in enumerate(tc.layers):
                    m.register_buffer(f"W{k + 1}", M); m.register_buffer(f"b{k + 1}", b)
                m.register_buffer("W6", tc.W6); m.n = len(tc.layers)
            def forward(m, acts):
                for k in range(m.n):
                    h = torch.relu(torch.sparse.mm(getattr(m, f"W{k + 1}"), acts) + getattr(m, f"b{k + 1}")[:, None]); acts = torch.cat([acts, h], 0)
                return torch.relu(torch.sparse.mm(m.W6, acts))
        return Cell(self)

class TorchMatrixParser:
    """MatrixParser with its step computed by TorchCell; same lexical input, same heads, same read_state"""
    name = "torch"
    def __init__(self, mp):
        self.mp = mp; self.cell = TorchCell(mp)
    def parse(self, toks, bound_per_token=12):
        import parser_matrix as PM
        torch = _torch(); mp = self.mp
        S = mp.c["S"]; s = np.zeros(mp.nS, dtype=np.float32); s[S[("ctrl", ("START",))]] = 1.0; i = 0
        for _ in range(bound_per_token * (len(toks) + 2)):
            ctrl, frame = mp.read_state(s)
            if ctrl == ("DONE",): return PM.frame_to_clause(frame)
            if ctrl == PM.FAIL: raise PM.ParseFail(f"TORCH: FAIL at token {i}", i)
            if ctrl is None: raise PM.ParseFail(f"TORCH: no rule fired at token {i}", i)
            if ctrl[0] == "MULTI": raise PM.ParseFail(f"TORCH: {ctrl[1]} rules fired at token {i}", i)
            if i > len(toks): raise PM.ParseFail("TORCH: ran past the end", i)
            x = self.lexical_input(toks, i, frame.get("verb"))
            acts = torch.from_numpy(np.concatenate([s, x]).astype(np.float32))[:, None]
            s = self.cell.forward(acts)[:, 0].numpy()
            if mp.shift_unit is None or s[mp.shift_unit] > 0.5: i += 1
        raise PM.ParseFail("TORCH: did not reach DONE", i)
    def lexical_input(self, toks, i, verb_value):
        """the input vector exactly as MatrixParser.step builds it, without stepping"""
        import parser_matrix as PM, rules_ja as R
        mp = self.mp; X = mp.c["X"]; x = mp.lexical_input(toks, i)
        for h in PM.table_hits(toks, i, verb_value):
            q = "hit=" + ",".join(map(str, h))
            if ("pred", q) in X: x[X[("pred", q)]] = 1
        tok = toks[i] if i < len(toks) else None
        for sname, v in PM.source_values(tok).items():
            if ("src", sname, v) in X: x[X[("src", sname, v)]] = 1
        if tok in R.PREPS: x[X[("prep0", tok)]] = 1
        x[X[("one",)]] = 1
        return x

def states_agree(mp, toks, bound_per_token=12):
    """run scipy and torch side by side on one sentence; -> (steps, first step where the states differ or None)"""
    torch = _torch(); tmp = TorchMatrixParser(mp)
    import parser_matrix as PM
    S = mp.c["S"]; s = np.zeros(mp.nS); s[S[("ctrl", ("START",))]] = 1.0; i = 0
    for step in range(bound_per_token * (len(toks) + 2)):
        ctrl, frame = mp.read_state(s)
        if ctrl in (("DONE",), PM.FAIL, None) or ctrl[0] == "MULTI" or i > len(toks): return step, None
        x = mp.lexical_input(toks, i); s_np = mp.step(s, x.copy(), toks, i, frame.get("verb"))
        acts = torch.from_numpy(np.concatenate([s, tmp.lexical_input(toks, i, frame.get("verb"))]).astype(np.float32))[:, None]
        s_t = tmp.cell.forward(acts)[:, 0].numpy()
        if not np.array_equal(s_np.astype(np.float32), s_t): return step, step
        s = s_np
        if mp.shift_unit is None or s[mp.shift_unit] > 0.5: i += 1
    return step, None

def demo():
    import parser_matrix as PM, rules_ja as R
    toks = [R.merge_multiword(s.split()) for s in ("i like the big dog .", "she took a rest .", "he gave me a book .")]
    comp = PM.build_hybrid(toks); mp = PM.MatrixParser(comp); tp = TorchMatrixParser(mp)
    for t in toks: print(" ".join(t), "->", R.generate(tp.parse(t)), "| agree:", states_agree(mp, t))
