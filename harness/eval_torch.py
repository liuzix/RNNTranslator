"""The cell under PyTorch: compile the hybrid cell on dev+test as the `matrix` action does, then step the scipy runner and
the torch runner (cell_torch.TorchCell) side by side on every sentence and require the same state vector at every step.

Reported: sentences, steps compared, sentences whose runs diverge (must be 0), the torch version.  A divergence would
mean the arithmetic is not exact in single precision or the matrices were not transferred faithfully — either way the
"runs unchanged as a PyTorch module" claim is false.
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.getcwd())
import rules_ja as R, run_rules as H, parser_matrix as PM, cell_torch as CT

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="result.json"); a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    pairs = {s: H.load(s) for s in ("dev", "test")}
    sents = {s: [R.merge_multiword(en) for en, ja in pairs[s]] for s in pairs}
    t0 = time.time()
    comp = PM.build_hybrid(sents["dev"] + sents["test"]); mp = PM.MatrixParser(comp)
    diverge = []; steps = 0
    for split in ("dev", "test"):
        for k, toks in enumerate(sents[split]):
            n, bad = CT.states_agree(mp, toks)
            steps += n
            if bad is not None: diverge.append({"split": split, "index": k, "step": bad})
    res = {"sentences": sum(len(v) for v in sents.values()), "steps": steps, "diverging": len(diverge), "diverging_items": diverge[:20],
           "torch": torch.__version__, "dtype": "float32", "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"torch {torch.__version__}: {res['sentences']} sentences, {steps} steps compared, {len(diverge)} diverging")

if __name__ == "__main__":
    main()
