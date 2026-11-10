"""The compiled machine under PyTorch: each cell — the encoder and the decoder — as the same five sparse layers and readout,
held as fixed buffers (not parameters) and stepped with torch.sparse.mm and ReLU.  Nothing is learned; the modules exist so
the circuit runs under a real tensor framework, batched, exportable, and comparable step for step with the scipy runner.

The arithmetic is small integers — weights ±1 (and the −5 that silences a rule), biases integer thresholds, activations
0/1 and small counts — so single precision is exact and the two runners agree bit for bit at every step of both cells:
pinned by the `torch` action (harness/eval_torch.py), which steps both runners on every dev and test sentence.

    uv run python -c "import cell_torch; cell_torch.demo()"       # needs torch importable in the environment
"""
import numpy as np
import parser_matrix as PM

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
    """one step of a cell: acts = [s; x] -> s', as sparse matrix products.  Built from a parser_matrix.Cell's matrices, so the two
    runners share one compile.  (A torch.nn.Module in all but inheritance, so that importing this file does not import torch;
    `module()` returns the nn.Module with the matrices registered as buffers.)"""
    def __init__(self, cell):
        torch = _torch()
        self.cell = cell
        self.layers = [(csr_to_torch(M), torch.from_numpy(bias.astype(np.float32))) for M, bias in cell.mats]
        self.W6 = csr_to_torch(cell.W6)
        self.nS, self.nX, self.width = cell.nS, cell.nX, cell.width

    def forward(self, acts):
        """acts: (width_in, B) column batch of [s; x] -> (nS, B) next states"""
        torch = _torch()
        for M, bias in self.layers:
            h = torch.relu(torch.sparse.mm(M, acts) + bias[:, None]); acts = torch.cat([acts, h], 0)
        return torch.relu(torch.sparse.mm(self.W6, acts))

    def step(self, s, x):
        """one state and one input as numpy vectors -> the next state as a numpy float32 vector"""
        torch = _torch()
        return self.forward(torch.from_numpy(np.concatenate([s, x]).astype(np.float32))[:, None])[:, 0].numpy()

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

class TorchMatrixParser(PM.MatrixParser):
    """MatrixParser with both cells stepped by TorchCell: the same inputs, the same heads, the same context between the cells"""
    name = "torch"
    def __init__(self, mp):
        self.__dict__.update(mp.__dict__)
        self.mp = mp; self.tenc = TorchCell(mp.enc); self.tdec = TorchCell(mp.dec)
    def step_encoder(self, s, x): return self.tenc.step(s, x)
    def step_decoder(self, s, x): return self.tdec.step(s, x)

def states_agree(mp, toks):
    """run the scipy cells on one sentence, the encoder and then the decoder, and give the torch cell the same state and input at
    every step: -> (steps compared, the first step where the two next states differ, or None)"""
    tp = mp.__dict__.get("_torch_twin")
    if tp is None: tp = mp.__dict__["_torch_twin"] = TorchMatrixParser(mp)
    count = [0]; bad = [None]
    def on_step(which, s, x, s2):
        t = (tp.tenc if which == "encoder" else tp.tdec).step(s, x)
        if bad[0] is None and not np.array_equal(s2.astype(np.float32), t): bad[0] = count[0]
        count[0] += 1
    try:
        frames, final = mp.encode(list(toks), on_step=on_step); mp.decode(frames, final, on_step=on_step)
    except PM.ParseFail: pass                           # a refusal ends the run; the steps up to it were compared
    return count[0], bad[0]

def demo():
    import rules_ja as R
    toks = [R.merge_multiword(s.split()) for s in ("i like the big dog .", "she took a rest .", "he gave me a book .", "he said that he was tired .")]
    comp = PM.build_hybrid(toks); mp = PM.MatrixParser(comp); tp = TorchMatrixParser(mp)
    for t in toks: print(" ".join(t), "->", tp.translate(t), "| agree:", states_agree(mp, t))
