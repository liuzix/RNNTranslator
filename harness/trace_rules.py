"""Step the hybrid rule set through one sentence and print, at every step, the state, the pending preposition, how many
rules fire and which one, with its literals and actions.  The first tool to reach for when the matrix parser and the rule
parser disagree on a sentence: the rule-set interpreter is what the matrices compute, one rule per step.

    uv run python harness/trace_rules.py "our teacher comes to school by car ."          # compiles on dev+test first (~20 s)
    uv run python harness/trace_rules.py --pkl .cache/compiled.pkl "..."               # reuse a saved compile
    uv run python harness/trace_rules.py --save .cache/compiled.pkl "..."              # compile once, keep it
"""
import os, sys, collections, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parser_matrix as PM, rules_ja as R, run_rules as H, parser as P

def show_state(st):
    if isinstance(st[0], tuple): return ">".join(st[0][1:]) + ":" + str(st[1])
    return str(st)

def trace(comp, toks, quiet_lits=False):
    rules, derived = comp["rules"], comp["derived"]
    by_prev = collections.defaultdict(list)
    for prev, lits, out in rules: by_prev[prev].append((lits, out))
    state = ("START",); i = 0; frame = {}; pend = None
    for _ in range(12 * (len(toks) + 2)):
        if state == ("DONE",): print("DONE", R.generate(PM.frame_to_clause(frame))); return
        if state == PM.FAIL: print("FAIL"); return
        P_ = PM.predicates(toks, i, PM.frame_to_snapshot(frame), set(range(i, i + PM.WINDOW)), PM.ALL_REGS | {"ret"}, pend, PM.heads_of(frame)) | PM.slot_predicates(frame)
        hits = [(lits, out) for lits, out in by_prev.get(state, []) if all(PM.eval_rule_lit(q, t, P_, derived) for q, t in lits)]
        print(f"i={i} {toks[i] if i < len(toks) else 'END'!r:12} {show_state(state):50} pend={pend} hits={len(hits)}")
        if len(hits) != 1:
            print("   predicates true here (registers and slots):", sorted(q for q in P_ if not q.startswith("w"))); return
        lits, (cp, k, actions) = hits[0]
        if not quiet_lits: print("      lits", lits)
        print("      ->", show_state(cp), "k", k, [a for a in actions if a[0] != "ret"])
        PM.apply_actions(frame, actions, toks, i)
        for j in range(i, i + k):
            if j < len(toks) and "prep" in dict(PM.readings_of(toks[j])): pend = toks[j]
        i += k; state = cp
    print("did not finish")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sentence"); ap.add_argument("--pkl"); ap.add_argument("--save"); ap.add_argument("-q", action="store_true", help="omit the literals")
    a = ap.parse_args()
    if a.pkl: comp = pickle.load(open(a.pkl, "rb"))
    else:
        sents = [R.merge_multiword(en) for s in ("dev", "test") for en, ja in H.load(s)]
        comp = PM.build_hybrid(sents)
        if a.save: pickle.dump(comp, open(a.save, "wb"))
    trace(comp, R.merge_multiword(a.sentence.split()), a.q)

if __name__ == "__main__":
    main()
