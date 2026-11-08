"""How much control state is the parser's own?  Two control points can share one phase symbol if the layers below
(the window's readings, the frame's register occupancy, the stack depth) never look the same at both.  The minimum
number of symbols is the chromatic number of the conflict graph; greedy colouring gives an upper bound, a clique a lower bound."""
import os, sys, collections, itertools; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rules_ja as R, run_rules as H
PMETH = set(n for n in dir(R.Parser) if callable(getattr(R.Parser, n)) and not n.startswith("__"))
REGS = ["subj", "obj", "iobj", "pp", "time", "adv", "verb", "past", "neg", "modal", "prog", "perf", "copula", "there", "question", "wh", "imp", "please", "lets", "want", "degree", "causative", "purpose", "control", "relative", "infinitive", "pre_clause", "embedded", "compare", "manner", "tag", "sub", "coord", "so_that", "from_seen"]
state = {"p": None, "last_i": 0, "line": 0}
events = []   # (control point, context pieces)
def stack_of(frame):
    names = []; f = frame
    while f is not None:
        if f.f_code.co_name in PMETH and f.f_locals.get("self") is state["p"]: names.append(f.f_code.co_name)
        f = f.f_back
    return tuple(reversed(names))
def window_classes(p, i):
    out = []
    for k in range(4):
        t = p.toks[i + k] if i + k < len(p.toks) else None
        out.append(tuple(sorted(set(c for c, _ in p.readings(t)))) if t is not None else ("END",))
    return tuple(out)
def occupancy(c):
    return tuple(bool(getattr(c, r)) for r in REGS)
def tracer(frame, event, arg):
    if frame.f_code.co_filename != R.__file__: return tracer
    if event == "line":
        p = state["p"]
        if p is not None and p.i != state["last_i"]:
            cp = (stack_of(frame), state["line"])
            c = p.c
            events.append((cp, state["win"], occupancy(c), c.depth, len(c.pp)))
            state["last_i"] = p.i
        state["line"] = frame.f_lineno
        if state["p"] is not None: state["win"] = window_classes(state["p"], state["p"].i)
    return tracer
pairs = H.load("test") + H.load("dev")
for en, ja in pairs:
    toks = R.merge_multiword(en if isinstance(en, list) else en.split())
    p = R.Parser(toks, []); state.update(p=p, last_i=0, line=0, win=window_classes(p, 0))
    sys.settrace(tracer)
    try: p.parse()
    except R.ParseFail: pass
    finally: sys.settrace(None)
cps = sorted(set(e[0] for e in events)); idx = {cp: i for i, cp in enumerate(cps)}
def chromatic(ctx_of):
    by_ctx = collections.defaultdict(set)
    for e in events: by_ctx[ctx_of(e)].add(idx[e[0]])
    n = len(cps); adj = [set() for _ in range(n)]
    for s in by_ctx.values():
        for a in s:
            adj[a] |= s
    for a in range(n): adj[a].discard(a)
    # DSATUR greedy colouring (upper bound)
    color = [-1] * n; sat = [set() for _ in range(n)]
    for _ in range(n):
        v = max((v for v in range(n) if color[v] < 0), key=lambda v: (len(sat[v]), len(adj[v])))
        c = 0
        while c in sat[v]: c += 1
        color[v] = c
        for u in adj[v]: sat[u].add(c)
    # greedy clique (lower bound)
    best = 0
    for v in sorted(range(n), key=lambda v: -len(adj[v]))[:60]:
        clique = {v}
        for u in sorted(adj[v], key=lambda u: -len(adj[u])):
            if all(u in adj[w] for w in clique): clique.add(u)
        best = max(best, len(clique))
    return max(color) + 1, best, len(by_ctx)
print(f"{len(cps)} control points visited on dev+test ({len(events)} token consumptions)\n")
for name, f in [("registers + depth + window readings (everything below)", lambda e: (e[1], e[2], e[3], e[4])),
                ("registers + depth only (no window)", lambda e: (e[2], e[3], e[4])),
                ("window readings only", lambda e: (e[1],)),
                ("nothing (the raw count)", lambda e: 0)]:
    ub, lb, nctx = chromatic(f)
    print(f"  given {name:55s}: phase symbols needed  {lb} <= chromatic <= {ub}   ({nctx} distinct contexts)")

# what the 5 symbols distinguish: colour classes under the full context, with their most-visited control points
import linecache
f = lambda e: (e[1], e[2], e[3], e[4])
by_ctx = collections.defaultdict(set)
for e in events: by_ctx[f(e)].add(idx[e[0]])
n = len(cps); adj = [set() for _ in range(n)]
for s in by_ctx.values():
    for a in s: adj[a] |= s
for a in range(n): adj[a].discard(a)
color = [-1] * n; sat = [set() for _ in range(n)]
for _ in range(n):
    v = max((v for v in range(n) if color[v] < 0), key=lambda v: (len(sat[v]), len(adj[v])))
    c = 0
    while c in sat[v]: c += 1
    color[v] = c
    for u in adj[v]: sat[u].add(c)
visits = collections.Counter(idx[e[0]] for e in events)
print()
for col in range(max(color) + 1):
    members = [v for v in range(n) if color[v] == col]
    tot = sum(visits[v] for v in members)
    print(f"symbol {col}: {len(members)} control points, {tot} visits")
    for v in sorted(members, key=lambda v: -visits[v])[:4]:
        stk, ln = cps[v]; print(f"     {visits[v]:4d}x {'>'.join(stk[1:]):50s} {linecache.getline(R.__file__, ln).strip()[:60]}")
# which conflicts force the 5th symbol: a maximum clique
best = set()
for v in sorted(range(n), key=lambda v: -len(adj[v]))[:200]:
    clique = {v}
    for u in sorted(adj[v], key=lambda u: -len(adj[u])):
        if all(u in adj[w] for w in clique): clique.add(u)
    if len(clique) > len(best): best = clique
print("\na clique of control points that must all differ (same window, same registers):")
for v in best:
    stk, ln = cps[v]; print(f"     {'>'.join(stk[1:]):50s} {linecache.getline(R.__file__, ln).strip()[:60]}")
