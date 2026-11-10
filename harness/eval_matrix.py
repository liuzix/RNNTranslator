"""The matrix parser against the rule parser: compile the encoder and the decoder on dev+test, cross-tabulate both
implementations of the parser interface on every sentence, and record the size of each cell and of what passes between them.

identical               both translate and the Japanese is the same (the matrices' is what the decoder emits)
wrong_frame             both parse, different Japanese            } silent misfires — the matrix answers where it should
matrix_parses_refused   the rule parser refuses, the matrix parses } not, or answers wrongly; the baseline guards their sum
refused_parseable       the rule parser parses, the matrix refuses (out of the encoder's scope, or a FAIL row)
both_refuse             neither parses
"""
import sys, os, json, argparse, collections, time
sys.path.insert(0, os.getcwd())
import rules_ja as R, run_rules as H, parser as P, canon as C
from parser_rules import ParseFail

def canon_survey(comp):
    """the canonical diagram of every control point of the cell (canon.py), over the cell's own slot vocabulary: how many are complete
    and consistent, the nodes in all, and one diagram's text form as the README's example"""
    by_prev = collections.defaultdict(list)
    for p, l, o in comp["rules"]: by_prev[p].append((l, o))
    tab = C.Table(C.Model(comp["slot_vocab"])); out = {"control_points": len(by_prev), "complete": 0, "consistent": 0, "nodes": 0, "not_total": []}
    for prev, rs in sorted(by_prev.items(), key=str):
        d = C.diagram(rs, table=tab); out["nodes"] += C.size(d)
        out["complete"] += C.complete(d); out["consistent"] += C.consistent(d)
        if not (C.complete(d) and C.consistent(d)) and len(out["not_total"]) < 20: out["not_total"].append(str(prev))
    example = ("NP1", "cop")
    out["example"] = C.dump(C.diagram(by_prev[example], table=tab), "NP1@cop") if example in by_prev else ""
    return out

def crosstab(mp, rp, rows):
    """rows: (line, en tokens, ja) from run_rules.load_indexed; returns the counts and the matrix output per citation"""
    cnt = collections.Counter(); hyp_r, hyp_m, ref = [], [], []; examples = collections.defaultdict(list); outputs = {}
    for line, en, ja in rows:
        toks = R.merge_multiword(en)
        try: want = rp.translate(toks)
        except ParseFail: want = None
        try: got = mp.translate(toks)
        except Exception: got = None
        outputs[f"{line}: {' '.join(en)}"] = got or ""
        if want is None: key = "both_refuse" if got is None else "matrix_parses_refused"
        else: key = "identical" if got == want else ("refused_parseable" if got is None else "wrong_frame")
        cnt[key] += 1
        if key in ("wrong_frame", "matrix_parses_refused") and len(examples[key]) < 8: examples[key].append(f"{' '.join(toks)} | {want} | {got}")
        if want is not None and got is not None: hyp_r.append(want); hyp_m.append(got); ref.append(ja)
    out = {k: cnt.get(k, 0) for k in ("identical", "wrong_frame", "matrix_parses_refused", "refused_parseable", "both_refuse")}
    out["shared"] = len(ref)
    out["bleu_rules_shared"] = round(H.char_bleu(hyp_r, ref)[0], 3) if ref else 0.0
    out["bleu_matrix_shared"] = round(H.char_bleu(hyp_m, ref)[0], 3) if ref else 0.0
    out["examples"] = dict(examples); out["outputs"] = outputs
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="result.json"); a = ap.parse_args()
    rows = {s: H.load_indexed(s) for s in ("dev", "test")}
    sents = {s: [R.merge_multiword(en) for _, en, ja in rows[s]] for s in rows}
    t0 = time.time(); mp = P.build_matrix_parser(sents["dev"] + sents["test"]); build_s = time.time() - t0
    rp = P.RuleParser()
    comp = mp.c
    kinds = collections.Counter()
    for prev, lits, out in comp["rules"]:
        if prev[0] == "P": kinds["print"] += 1                                      # the decoder: the printing positions
        elif prev[0] in ("OPEN", "CLOSE"): kinds["flush"] += 1                      # the clause rules: a frame flushed to the context, the working frame cleared
        elif prev[0] in ("M", "RETM") or (prev[0] == "RET" and out[0][0] == "M"): kinds["clause"] += 1   # the hand-translated clause modules and their hand-overs
        elif isinstance(prev[0], str) and prev[0].startswith("NP"): kinds["module"] += 1
        elif prev[0] == "RET": kinds["exit"] += 1
        elif out[0][0] == "NP0": kinds["entry"] += 1
        elif out[0] == ("FAIL",): kinds["fail"] += 1
        else: kinds["outer"] += 1
    res = {"per_split": {s: crosstab(mp, rp, rows[s]) for s in ("dev", "test")}}
    res["outputs"] = {s: res["per_split"][s].pop("outputs") for s in ("dev", "test")}
    tot = collections.Counter()
    for s in res["per_split"]:
        for k in ("identical", "wrong_frame", "matrix_parses_refused", "refused_parseable", "both_refuse"): tot[k] += res["per_split"][s][k]
    res.update(tot)
    t0 = time.time(); canon = canon_survey(comp); canon["seconds"] = round(time.time() - t0, 1)
    def cell_stats(cell): return {"state_units": cell.nS, "input_units": cell.nX, "nnz": int(cell.nnz()), "layers": [len(L) for L, _, _ in cell.c["layers"]], "rules": len(cell.c["rules"])}
    res["cell"] = {"encoder": cell_stats(mp.enc), "decoder": cell_stats(mp.dec), "frame_units": len(mp.frame_units), "sentence_units": len(mp.handoff),
                   "output_units": len(mp.out_units), "nnz": int(mp.nnz()), "rules": len(comp["rules"]),
                   "rules_by_kind": dict(kinds), "build_seconds": round(build_s), "scope": dict(comp.get("scope", {})),
                   "unresolved": {q: n for q, n in sorted(comp.get("unresolved", {}).items())},     # positive literals the cell cannot see: those rules can never fire
                   "canon": canon}
    res["summary"] = (f"identical {res['identical']}, wrong frame {res['wrong_frame']}, matrix parses refused {res['matrix_parses_refused']}, "
                      f"refused parseable {res['refused_parseable']}, both refuse {res['both_refuse']}; nnz {res['cell']['nnz']}, rules {res['cell']['rules']}; "
                      f"canon: {canon['control_points']} control points, {canon['control_points'] - canon['complete']} with a gap, {canon['control_points'] - canon['consistent']} with a conflict"
                      + (f"; UNRESOLVED literals {sum(res['cell']['unresolved'].values())}: {', '.join(res['cell']['unresolved'])}" if res["cell"]["unresolved"] else ""))
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(res["summary"])

if __name__ == "__main__":
    main()
