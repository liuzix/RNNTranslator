import os, sys, collections, pickle; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parser_matrix as PM, rules_ja as R
from parser_rules import Parser, ParseFail
comp = pickle.load(open("parser_matrix.pkl", "rb"))
rules, derived = comp["rules"], comp["derived"]
by_prev = collections.defaultdict(list)
for prev, lits, out in rules: by_prev[prev].append((lits, out))
def show_state(st):
    if isinstance(st[0], tuple): return ">".join(st[0][1:]) + ":" + str(st[1])
    return str(st)
def run(toks):
    state = ("START",); i = 0; frame = {}; pend = None
    for _ in range(12 * (len(toks) + 2)):
        if state == ("DONE",): print("DONE", R.generate(PM.frame_to_clause(frame))); return
        if state == PM.FAIL: print("FAIL"); return
        P = PM.predicates(toks, i, PM.frame_to_snapshot(frame), set(range(i, i + PM.WINDOW)), PM.ALL_REGS | {"ret"}, pend, PM.heads_of(frame)) | PM.slot_predicates(frame)
        hits = [(lits, out) for lits, out in by_prev.get(state, []) if all(PM.eval_rule_lit(q, t, P, derived) for q, t in lits)]
        print(f"i={i} {toks[i] if i < len(toks) else 'END'!r:12} {show_state(state):50} pend={pend} hits={len(hits)}")
        if len(hits) != 1:
            print("   P:", sorted(q for q in P if not q.startswith("w"))); return
        lits, (cp, k, actions) = hits[0]
        print("      lits", lits, "\n      ->", show_state(cp), "k", k, [a for a in actions if a[0] != "ret"])
        PM.apply_actions(frame, actions, toks, i)
        for j in range(i, i + k):
            if j < len(toks) and "prep" in dict(PM.readings_of(toks[j])): pend = toks[j]
        i += k; state = cp
run(R.merge_multiword(sys.argv[1].split()))
