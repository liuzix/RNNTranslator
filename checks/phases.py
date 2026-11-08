"""Enumerate the parser's implicit phases: every control point that consumes a token, with the stack of Parser methods it is nested in."""
import os, sys, collections, linecache; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rules_ja as R, run_rules as H
PMETH = set(n for n in dir(R.Parser) if callable(getattr(R.Parser, n)) and not n.startswith("__"))
phases = collections.Counter(); sample = {}; consumed = collections.defaultdict(collections.Counter)
state = {"p": None, "last_i": 0}
def stack_of(frame):
    names = []
    f = frame
    while f is not None:
        co = f.f_code.co_name
        if co in PMETH and f.f_locals.get("self") is state["p"]: names.append(co)
        f = f.f_back
    return tuple(reversed(names))
def tracer(frame, event, arg):
    if frame.f_code.co_filename != R.__file__: return tracer
    if event == "line":
        p = state["p"]
        if p is not None and p.i != state["last_i"]:
            # the previous line consumed tokens: phase = (method stack, that line)
            key = (stack_of(frame), state["line"])
            phases[key] += 1
            tok = p.toks[state["last_i"]] if state["last_i"] < len(p.toks) else None
            consumed[key][tok] += 1
            sample.setdefault(key, state["sent"])
            state["last_i"] = p.i
        state["line"] = frame.f_lineno
    return tracer
pairs = H.load("test") + H.load("dev")
n_ok = 0
for en, ja in pairs:
    toks = R.merge_multiword(en if isinstance(en, list) else en.split())
    p = R.Parser(toks, []); state.update(p=p, last_i=0, line=0, sent=" ".join(toks))
    sys.settrace(tracer)
    try: p.parse(); n_ok += 1
    except R.ParseFail: pass
    finally: sys.settrace(None)
print(f"{n_ok} sentences parsed; {len(phases)} distinct consuming control points (method stack x line)")
by_method = collections.Counter(); depth = collections.Counter()
for (stk, ln), c in phases.items(): by_method[stk[-1] if stk else "?"] += 1; depth[len(stk)] += 1
print("control points per method:", dict(by_method.most_common()))
print("by nesting depth of Parser methods:", dict(sorted(depth.items())))
print("\nmost visited:")
for (stk, ln), c in phases.most_common(28):
    src = linecache.getline(R.__file__, ln).strip()
    toks = ", ".join(f"{t}" for t, _ in consumed[(stk, ln)].most_common(4))
    print(f"  {c:5d}x  {'>'.join(stk):55s} L{ln}: {src[:70]}\n          tokens: {toks} | e.g. {sample[(stk, ln)][:60]}")

print("\n-- the same source line as different phases (parse_np consuming a noun, L417):")
for (stk, ln), c in sorted(phases.items(), key=lambda kv: -kv[1]):
    if ln == 417 and c >= 5: print(f"  {c:4d}x  {'>'.join(stk)}")
vp = collections.Counter(); deepest = None
for (stk, ln), c in phases.items():
    k = stk.count("verb_phrase"); vp[k] += c
    if deepest is None or len(stk) > len(deepest[0]): deepest = (stk, sample[(stk, ln)])
print("\nverb_phrase nesting inside one stack (complement chains), by visits:", dict(sorted(vp.items())))
print("deepest stack:", ">".join(deepest[0]), "\n   e.g.", deepest[1])
# clause-level nesting: how many times a clause-opening method appears in the stack
cl = collections.Counter()
for (stk, ln), c in phases.items():
    cl[sum(stk.count(m) for m in ("sub_clause", "coord", "embedded_question", "embedded_after"))] += c
print("clause-opening methods on the stack, by visits:", dict(sorted(cl.items())))
