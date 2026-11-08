"""The negatives: ungrammatical sentences from the literature that the translator must refuse, each paired with a
grammatical control it must accept (harness/negatives.json).  A translator that refused everything would pass the
negatives and fail the controls; one that accepted everything would do the opposite.

Asserted: every token of every sentence and control is in the lexicon (a refusal for an unknown word proves nothing);
the rule parser accepts every control and refuses every negative.  The matrix's behaviour is reported, and the baseline
guards it: a negative the matrix accepts is a silent misfire of an extracted rule (matrix_accepted must not rise), and a
control it refuses may simply be outside its scope.
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.getcwd())
import rules_ja as R, run_rules as H, parser as P
from parser_rules import ParseFail

def known(toks): return [t for t in toks if not R.classify(t) and t not in (".", "?", "!", ",", "'s")]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="result.json"); a = ap.parse_args()
    doc = json.load(open(os.path.join("harness", "negatives.json"), encoding="utf-8"))
    pairs = {s: H.load(s) for s in ("dev", "test")}
    sents = {s: [R.merge_multiword(en) for en, ja in pairs[s]] for s in pairs}
    t0 = time.time(); mp = P.build_matrix_parser(sents["dev"] + sents["test"]); rp = P.RuleParser()
    rows = []; problems = []
    for e in doc["sentences"]:
        row = {"sentence": e["sentence"], "control": e["control"], "phenomenon": e["phenomenon"]}
        for key in ("sentence", "control"):
            toks = R.merge_multiword(e[key].split())
            unk = known(toks)
            if unk: problems.append(f"{key} {e[key]!r}: unknown tokens {unk}")
            for name, impl in (("rules", rp), ("matrix", mp)):
                try: row[f"{key}.{name}"] = impl.translate(toks)
                except ParseFail as x: row[f"{key}.{name}"] = None; row[f"{key}.{name}.reason"] = str(x).split(" at ")[0][:60]
                except Exception as x: row[f"{key}.{name}"] = None; row[f"{key}.{name}.reason"] = type(x).__name__
        if row["control.rules"] is None: problems.append(f"control not accepted by the rule parser: {e['control']!r} ({row.get('control.rules.reason')})")
        if row["sentence.rules"] is not None: problems.append(f"negative accepted by the rule parser: {e['sentence']!r} -> {row['sentence.rules']}")
        rows.append(row)
    n = len(rows)
    res = {"count": n, "accepted": sum(1 for r in rows if r["sentence.rules"] is not None),
           "matrix_accepted": sum(1 for r in rows if r["sentence.matrix"] is not None),
           "controls_refused_by_rules": sum(1 for r in rows if r["control.rules"] is None),
           "controls_refused_by_matrix": sum(1 for r in rows if r["control.matrix"] is None),
           "problems": problems, "rows": rows, "seconds": round(time.time() - t0)}
    res["summary"] = (f"negatives: rules refuse {n - res['accepted']}/{n} and accept {n - res['controls_refused_by_rules']}/{n} controls; "
                      f"matrix refuses {n - res['matrix_accepted']}/{n} and accepts {n - res['controls_refused_by_matrix']}/{n} controls") + (f"; {len(problems)} problem(s)" if problems else "")
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(res["summary"])
    for p in problems: print("  problem:", p)
    sys.exit(1 if problems else 0)

if __name__ == "__main__":
    main()
