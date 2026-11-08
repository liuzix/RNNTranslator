"""Coverage and quality of the rule transducer on the pinned corpus: dev and test, every output recorded.

coverage      sentences the transducer translates (the rest it refuses with a ParseFail)
bleu_all      char-BLEU-4 over every sentence, a refusal counting as the empty string — the headline quality number,
              monotone in coverage, and what the baseline guards
bleu_covered  char-BLEU-4 over the translated sentences only (a harder sentence newly covered can lower this)
"""
import sys, os, json, argparse, collections
sys.path.insert(0, os.getcwd())
import rules_ja as R, run_rules as H

def evaluate(split):
    rows = H.load_indexed(split)
    outputs = {}; fails = collections.Counter(); refusals = {}   # "<line>: <sentence>" -> Japanese ("" = refused); refusals -> reason
    for line, en, ja in rows:
        key = f"{line}: {' '.join(en)}"
        try: outputs[key] = R.translate(en)
        except R.ParseFail as e:
            outputs[key] = ""; reason = str(e).split(" at ")[0][:40]; fails[reason] += 1; refusals[key] = reason
    pairs = [(en, ja) for _, en, ja in rows]
    refs = [ja for _, ja in pairs]; hyps = [outputs[f"{line}: {' '.join(en)}"] for line, en, ja in rows]
    ok = [(h, r) for h, r in zip(hyps, refs) if h]
    b_all = H.char_bleu(hyps, refs)[0]
    b_ok = H.char_bleu([h for h, _ in ok], [r for _, r in ok])[0] if ok else 0.0
    return {"n": len(pairs), "covered": len(ok), "coverage": round(len(ok) / len(pairs), 4), "bleu_all": round(b_all, 3), "bleu_covered": round(b_ok, 3),
            "failures": dict(fails.most_common(12)), "outputs": outputs, "refusals": refusals}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="result.json"); a = ap.parse_args()
    res = {s: evaluate(s) for s in ("dev", "test")}
    res["summary"] = "; ".join(f"{s}: {r['covered']}/{r['n']} covered, BLEU all {r['bleu_all']:.1f} / covered {r['bleu_covered']:.1f}" for s, r in res.items())
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(res["summary"])

if __name__ == "__main__":
    main()
