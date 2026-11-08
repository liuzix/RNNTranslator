import os, sys, collections; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parser_matrix as PM, rules_ja as R, run_rules as H
from parser_rules import Parser, ParseFail
def rows_for(split):
    out = {}
    for en, ja in H.load(split):
        toks = R.merge_multiword(en if isinstance(en, list) else en.split())
        try: Parser(list(toks), []).parse(); rows, final = PM.event_rows_pred(toks, [])
        except (ParseFail, PM.OutOfScope): continue
        except Exception as e: continue
        out[tuple(toks)] = rows
    return out
dev, test = rows_for("dev"), rows_for("test")
def check(trees, sents, label, show=3):
    ok = bad = 0
    for toks in sents:
        want = R.generate(Parser(list(toks), []).parse())
        try:
            got = R.generate(PM.frame_to_clause(PM.run_table(trees, list(toks))))
            if got == want: ok += 1
            else:
                bad += 1
                if bad <= show: print("   MISMATCH", " ".join(toks), "|", want, "|", got)
        except ParseFail as e:
            bad += 1
            if bad <= show: print("   FAIL    ", " ".join(toks), "|", e)
    print(f"{label}: {ok} same, {bad} different")
allrows = [r for rs in list(dev.values()) + list(test.values()) for r in rs]
trees = PM.induce(allrows); rules = [(p, l, o) for p, t in trees.items() for l, o in PM.tree_rules(t)]
print(f"all: {len(allrows)} events -> {len(trees)} states, {len(rules)} rules, mean {sum(len(l) for _, l, _ in rules)/len(rules):.1f} literals")
check(trees, list(dev) + list(test), "table built on dev+test, run on dev+test")
trees_dev = PM.induce([r for rs in dev.values() for r in rs]); rules_dev = [(p, l, o) for p, t in trees_dev.items() for l, o in PM.tree_rules(t)]
print(f"dev only: {len(trees_dev)} states, {len(rules_dev)} rules")
check(trees_dev, list(test), "table built on dev, run on test (generalization)", show=0)
