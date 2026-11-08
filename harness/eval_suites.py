"""The acceptance suites: one file per residual class under harness/suites/, each a list of sentences with the exact
Japanese the transducer must produce.  A case is RIGHT when the output equals `expect`, REFUSED when the transducer
refuses, WRONG otherwise; a class is closed when every case is RIGHT.  The expectations are human verdicts (the suite
author's), so a case that cites a corpus line also adjudicates that output as right for the sentence item.
"""
import sys, os, json, glob, argparse, collections
sys.path.insert(0, os.getcwd())
import rules_ja as R
from parser_rules import ParseFail

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="result.json"); a = ap.parse_args()
    rows = []; per_class = {}
    for path in sorted(glob.glob(os.path.join("harness", "suites", "*.json"))):
        suite = json.load(open(path, encoding="utf-8")); cls = suite["class"]
        counts = collections.Counter()
        for case in suite["cases"]:
            toks = case["sentence"].split()
            try: out = R.translate(toks); reason = None
            except ParseFail as e: out = ""; reason = str(e).split(" at ")[0][:40]
            level = "RIGHT" if out == case["expect"] else ("REFUSED" if not out else "WRONG")
            counts[level] += 1
            rows.append({"class": cls, "sentence": case["sentence"], "expect": case["expect"], "output": out, "level": level, "reason": reason, "citation": case.get("citation")})
        per_class[cls] = {"title": suite["title"], "cases": len(suite["cases"]), "right": counts["RIGHT"], "refused": counts["REFUSED"], "wrong": counts["WRONG"], "closed": counts["RIGHT"] == len(suite["cases"])}
    res = {"classes": per_class, "rows": rows, "cases": len(rows), "right": sum(1 for r in rows if r["level"] == "RIGHT"),
           "closed": sum(1 for c in per_class.values() if c["closed"]), "open": sum(1 for c in per_class.values() if not c["closed"])}
    res["summary"] = f"suites: {res['right']}/{res['cases']} cases right, {res['closed']} of {len(per_class)} classes closed" + ("" if not per_class else " (open: " + ", ".join(sorted(c for c, v in per_class.items() if not v["closed"])) + ")")
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(res["summary"])
    for r in rows:
        if r["level"] != "RIGHT": print(f"  {r['class']} {r['level']:7s} {r['sentence']:50s} -> {r['output'] or r['reason']}")

if __name__ == "__main__":
    main()
