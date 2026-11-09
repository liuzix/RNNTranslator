"""The canonical form of the rules: per control point, a reduced ordered decision diagram.

A compiled rule is (control point, literals, out) — a conjunction of literals over the window and the state, and what
to do when it holds.  The rules of one control point are a function from assignments of the literals to outcomes, and
this module writes that function in the one form it has under a fixed variable order: a decision diagram whose tests
follow the order, whose identical sub-diagrams are shared, and whose tests with identical branches are removed.  Two
rule sets that define the same function give the same diagram whatever order the rules came in (pinned by
test_canon_is_canonical); the diagram's paths are the rule set in normal form — no redundant literal, every path ending
in a terminal, a GAP (no rule covers it) or a CONFLICT (two rules with different outcomes cover it).

Variables.  A literal `k=v` whose key is a one-hot family (a window token's surface form, a register's value, the
pending preposition, pp.count, a slot's value) is a test of the family variable k with one branch per value that occurs
and a `*` branch for every other value; any other literal (a token class, a flag, a slot being full, a derived
predicate) is a boolean test.  hypothesis: derived predicates are tested as opaque booleans, so a diagram may contain
paths no input can reach (two derived predicates that exclude each other both true); expanding them into their base
literals would remove those paths and is the next refinement.

The text form (dump / parse, pinned by test_canon_dsl_roundtrip):

    NP1@cop:
      w0.class=num?
        + d.one_pronoun@cop?
            + → ('NP2', 'cop') k=0 ()
            - → ('NP2', 'cop') k=1 (('cop.num', None, ('copy', 'num')),)
        - → ('NP2', 'cop') k=0 ()

A family test prints its key with a colon and one line per value (`= a »`, `= the »`, `* »`); a sub-diagram reached from two
places is printed once, labelled `#n:` where it first appears and referenced as `#n` afterwards; GAP and CONFLICT print
as those words (CONFLICT followed by its outcomes).
"""
import re, ast, random

ORDER = ("w0.", "w1.", "w2.", "w3.", "pend_prep", "hit=", "slot.", "reg.", "flag.", "verb.", "pp.count", "use.", "agree.", "subj.", "d.")
FAMILY_KEYS = re.compile(r"^(w\d\.(surf|vform|akind|prole)|pend_prep|pp\.count|reg\.[\w]+|verb\.conj|slot\.\w+\.\w+)$")

GAP, CONFLICT, TERM = "GAP", "CONFLICT", "TERM"

def var_of(q):
    """a literal -> (variable key, value or None for a boolean literal)"""
    if "=" in q:
        k, v = q.split("=", 1)
        if FAMILY_KEYS.match(k): return k + "=", v          # the family variable is named `k=`, distinct from a bare boolean `k`
    return q, None

def rank(key, order=ORDER):
    for i, p in enumerate(order):
        if key.startswith(p): return (i, key)
    return (len(order), key)

class Node:
    __slots__ = ("kind", "var", "branches", "out", "id")
    def __init__(self, kind, var=None, branches=None, out=None):
        self.kind = kind; self.var = var; self.branches = branches; self.out = out; self.id = None
    def __repr__(self): return dump(self).strip()

class Table:
    """hash-consing: structurally equal nodes are the same object, so equality is identity"""
    def __init__(self): self.nodes = {}
    def leaf(self, kind, out=None):
        key = (kind, out); n = self.nodes.get(key)
        if n is None: n = self.nodes[key] = Node(kind, out=out)
        return n
    def test(self, var, branches):
        vals = list(branches.values())
        if all(b is vals[0] for b in vals): return vals[0]                          # a test with identical branches is removed
        key = ("TEST", var, tuple(sorted(((str(k), id(b)) for k, b in branches.items()))))
        n = self.nodes.get(key)
        if n is None: n = self.nodes[key] = Node("TEST", var=var, branches=dict(branches))
        return n

def diagram(rules, order=ORDER, table=None):
    """rules: iterable of (lits, out) or (prev, lits, out) for ONE control point -> the root node of its diagram"""
    table = table or Table()
    rs = [(tuple(r[-2]), r[-1]) for r in rules]
    return _build(rs, order, table)

def _build(rs, order, table):
    keys = {var_of(q)[0] for lits, _ in rs for q, _t in lits}
    if not keys:
        outs = sorted({repr(o): o for _, o in rs}.values(), key=repr)
        if not outs: return table.leaf(GAP)
        return table.leaf(TERM, outs[0]) if len(outs) == 1 else table.leaf(CONFLICT, tuple(outs))
    k = min(keys, key=lambda x: rank(x, order))
    boolean = all(var_of(q)[1] is None for lits, _ in rs for q, _t in lits if var_of(q)[0] == k)
    if boolean:
        yes = [(tuple(l for l in lits if l[0] != k), o) for lits, o in rs if dict(lits).get(k, True)]
        no = [(tuple(l for l in lits if l[0] != k), o) for lits, o in rs if not dict(lits).get(k, False)]
        return table.test(k, {True: _build(yes, order, table), False: _build(no, order, table)})
    values = sorted({var_of(q)[1] for lits, _ in rs for q, _t in lits if var_of(q)[0] == k})
    def strip(lits): return tuple(l for l in lits if var_of(l[0])[0] != k)
    branches = {}
    for v in values:
        sel = [(strip(lits), o) for lits, o in rs if all((var_of(q)[1] == v) == t for q, t in lits if var_of(q)[0] == k)]
        branches[v] = _build(sel, order, table)
    other = [(strip(lits), o) for lits, o in rs if all(not t for q, t in lits if var_of(q)[0] == k)]
    branches["*"] = _build(other, order, table)
    return table.test(k, branches)

def equal(a, b): return a is b or dump(a) == dump(b)

def leaves(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen: return
    seen.add(id(node))
    if node.kind != "TEST": yield node; return
    for b in node.branches.values(): yield from leaves(b, seen)

def complete(node): return all(l.kind != GAP for l in leaves(node))
def consistent(node): return all(l.kind != CONFLICT for l in leaves(node))

def paths(node, prefix=()):
    """every root-to-leaf path -> (literals in diagram order, leaf)"""
    if node.kind != "TEST": yield prefix, node; return
    for v, b in node.branches.items():
        if v is True: lits = ((node.var, True),)
        elif v is False: lits = ((node.var, False),)
        elif v == "*": lits = tuple((f"{node.var}{u}", False) for u in sorted(node.branches) if u != "*")
        else: lits = ((f"{node.var}{v}", True),)
        yield from paths(b, prefix + lits)

def to_rules(prev, node):
    """the control point's rules in normal form: one per path to a TERM leaf"""
    return [(prev, lits, leaf.out) for lits, leaf in paths(node) if leaf.kind == TERM]

def size(node): return sum(1 for _ in _nodes(node))
def _nodes(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen: return
    seen.add(id(node)); yield node
    if node.kind == "TEST":
        for b in node.branches.values(): yield from _nodes(b, seen)

# ---------------------------------------------------------------- the text form
def _label_shared(root):
    counts = {}
    def walk(n):
        counts[id(n)] = counts.get(id(n), 0) + 1
        if counts[id(n)] == 1 and n.kind == "TEST":
            for b in n.branches.values(): walk(b)
    walk(root)
    labels = {}
    for n in _nodes(root):
        if counts[id(n)] > 1 and n is not root: labels[id(n)] = f"#{len(labels) + 1}"
    return labels

def dump(root, name=None):
    labels = _label_shared(root); printed = set(); out = []
    def line(depth, s): out.append("  " * depth + s)
    def node(n, depth, lead):
        tag = labels.get(id(n))
        if tag and id(n) in printed: line(depth, f"{lead}{tag}"); return
        if tag: printed.add(id(n)); lead = f"{lead}{tag}: "
        if n.kind == TERM: line(depth, f"{lead}→ {fmt_out(n.out)}")
        elif n.kind == GAP: line(depth, f"{lead}GAP")
        elif n.kind == CONFLICT: line(depth, f"{lead}CONFLICT " + " ∥ ".join(fmt_out(o) for o in n.out))
        elif set(n.branches) == {True, False}:
            line(depth, f"{lead}{n.var}?"); node(n.branches[True], depth + 2, "+ "); node(n.branches[False], depth + 2, "- ")
        else:
            line(depth, f"{lead}{n.var[:-1]}:")
            for v in sorted(n.branches, key=lambda x: (x == "*", str(x))): node(n.branches[v], depth + 2, "* » " if v == "*" else f"= {v} » ")
    if name is not None: line(0, f"{name}:"); node(root, 1, "")
    else: node(root, 0, "")
    return "\n".join(out) + "\n"

def fmt_out(out):
    nxt, k, acts = out
    return f"{nxt!r} k={k} {acts!r}"

def parse_out(s):
    m = re.match(r"^(.*) k=(\d+) (.*)$", s.strip())
    return (ast.literal_eval(m.group(1)), int(m.group(2)), ast.literal_eval(m.group(3)))

def parse(text, table=None):
    """the text form -> (name or None, root node)"""
    table = table or Table()
    lines = [(len(l) - len(l.lstrip(" ")), l.strip()) for l in text.splitlines() if l.strip()]
    name = None
    if lines and lines[0][1].endswith(":") and not lines[0][1].endswith("?:") and lines[0][0] == 0 and (len(lines) == 1 or lines[1][0] > 0) and "#" not in lines[0][1] and not _is_test_header(lines[0][1][:-1]):
        name = lines[0][1][:-1]; lines = lines[1:]
    labels = {}
    pos = [0]
    def parse_node(depth_expected):
        indent, s = lines[pos[0]]; pos[0] += 1
        m = re.match(r"^(#\d+):\s*(.*)$", s)
        tag = None
        if m: tag, s = m.group(1), m.group(2)
        elif re.match(r"^#\d+$", s): return labels[s]
        if s.startswith("→ "): n = table.leaf(TERM, parse_out(s[2:]))
        elif s == "GAP": n = table.leaf(GAP)
        elif s.startswith("CONFLICT "): n = table.leaf(CONFLICT, tuple(parse_out(p) for p in s[len("CONFLICT "):].split(" ∥ ")))
        elif s.endswith("?"):
            var = s[:-1]; branches = {}
            for _ in range(2):
                ind, t = lines[pos[0]]; lead = t[:2]; lines[pos[0]] = (ind, t[2:])
                branches[lead == "+ "] = parse_node(ind)
            n = table.test(var, branches)
        elif s.endswith(":"):
            var = s[:-1] + "="; branches = {}
            while pos[0] < len(lines) and lines[pos[0]][0] > indent:
                ind, t = lines[pos[0]]
                if t.startswith("* » "): v, rest = "*", t[4:]
                else:
                    mm = re.match(r"^= (.*?) » (.*)$", t); v, rest = mm.group(1), mm.group(2)
                lines[pos[0]] = (ind, rest); branches[v] = parse_node(ind)
            n = table.test(var, branches)
        else: raise ValueError(f"cannot parse line: {s!r}")
        if tag: labels[tag] = n
        return n
    root = parse_node(0)
    return name, root

def _is_test_header(s): return s.endswith("?") or "=" in s or s.startswith("#")

def shuffled(rules, seed=0):
    rs = list(rules); random.Random(seed).shuffle(rs); return rs
