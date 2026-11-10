"""The canonical form of the rules: per control point, a reduced ordered decision diagram over the cell's variables.

A compiled rule is (control point, literals, out): a conjunction of literals over the window and the state, and what to do
when it holds.  The rules of one control point are a function from inputs to outcomes, and this module writes that function
in the one form it has: a decision diagram whose tests follow a fixed order of the variables, whose identical sub-diagrams
are shared, and whose tests with identical branches are removed.  Two rule sets that define the same function give the
same diagram whatever order the rules came in; the diagram's paths are the rule set in normal form — no redundant literal,
every path ending in a terminal, a GAP (no rule covers it) or a CONFLICT (rules with different outcomes cover it).

Variables and values.  The diagram is built over the cell's own variables, each with a universe of values, so that no path
is a combination no input can be: the four window slots (w0..w3), whose values are the tokens of the lexicon, each as the
set of window predicates (class, surface, form, kind, table key) the lexical layer gives it, tokens with the same predicates
being one value; the pending preposition; the PP count, one variable that fills in order (pp<k>.noun is filled iff the count
reaches k, and the noun of a filled role is a variable of its own); and every other slot of the state, whose values are what
the rules can write into it (the vocabulary of the hand-written rules and the lexicon's copy sources, or a cell's own, plus the
constants the control point's rules mention) or EMPTY.  A test on a variable has one branch per class of values that agree on
every predicate the remaining rules test on it, labelled by the predicates true in that class (`*` when none is).  Every other
literal is expanded into base literals before it is tested: a derived predicate (d.*, agree.bad) by its DNF, np.<role>.time by
the role's slots, use.<role> and the hits the verb decides by their table rows over the verb slot and the window, the hits the
window alone decides by their table rows over the window.  Hence a complete diagram means the rules cover every feasible
input, and a consistent one that no feasible input is claimed by two outcomes.

The text form (dump / parse) prints a test as `<variable> ⟨p1 ∧ p2 ∧ …⟩:` followed by one line per class, `= p1 ∧ p2 »` for
the predicates true in it or `* »` for none, a leaf as `→ out`, `GAP` or `CONFLICT a ∥ b`, and a sub-diagram reached from two
places once, labelled `#n:` where it first appears and referenced as `#n` afterwards.
"""
import re, ast, random, collections
import rules_ja as R
import gen_module as GM
import parser_matrix as PM
import np_module as NPM

GAP, CONFLICT, TERM = "GAP", "CONFLICT", "TERM"
EMPTY = "∅"                                   # the value of a slot that holds nothing (a flag that is off)
OTHER = "«other»"                             # a token the lexicon does not know: no readings, its surface its own

# ---------------------------------------------------------------- the model: variables, universes, predicates
class Model:
    """the universes the diagram ranges over: the lexicon's tokens with their window predicates, every slot's values, the
    derived predicates of every module, and the expansions of the joint predicates"""
    def __init__(self, slot_vocab=None):
        self.slot_vocab = {k: set(v) for k, v in (slot_vocab if slot_vocab is not None else PM.universal_vocabulary()).items()}
        self.window_values, self.feats = self._window()
        self.derived = self._derived()
        self._expansion = {}

    @staticmethod
    def _window():
        """the window's universe: the lexicon's tokens, each as the set of window predicates true of it — grouped, since tokens with the
        same predicates are the same value to every rule.  A token's identity (tbl=) and a verb's lemma (lemma=) are predicates only
        for the tokens the tables key (the hits and their rows): for the others the cell has no unit"""
        table_toks = set(R.USE) | set(R.SOFTENER) | {t for (a, t) in R.PARTICLE_AFTER} | {t for (a, tt) in R.IDIOM_TAIL for t in tt if t != "SELF"}
        table_toks |= {t for (a, tt) in R.VERB_IDIOM for t in tt} | {t for (a, t) in R.PHRASAL} | {a for (a, b) in R.ADJ_PREP} | {b for (a, b) in R.ADJ_PREP}
        table_lemmas = {a for (a, tt) in R.VERB_IDIOM} | {a for (a, t) in R.PHRASAL}
        toks = set(R.LEX) | set(R.PRONOUNS) | set(R.DETS) | set(R.NUMBERS) | set(R.WH) | set(R.MODALS) | set(R.VERB_FORMS) | set(R.PREPS) | set(R.DURATION)
        toks |= set(PM.IDENTITY_WORDS) | set(R.CONTRACTED) | {".", "?", "!", ",", "not", "n't", "'t", "'s", "'m", "'re", "'ve", "to", "there", "that"} | set(R.HEARSAY) | table_toks
        toks |= set(R.SUBORD) | set(R.COORD)
        feats = {}
        for t in sorted(toks) + [None, OTHER]:
            surface, classes, extra = PM.window_features([t] if t is not None else [], 0)[0]
            f = {f"class={c}" for c in classes}
            if surface != "*" and surface != OTHER: f.add(f"surf={surface}")
            for e in extra:
                if e[0] == "vc": f |= {f"vc={c}" for c in e[1]}
                else: f.add(f"{e[0]}={e[1]}")
            if t in table_toks: f.add(f"tbl={t}")
            for c, v in PM.readings_of(t) if t is not None else ():
                if c == "verb" and v[0] in table_lemmas: f.add(f"lemma={v[0]}")
            feats[t] = frozenset(f)
        return sorted(set(feats.values()), key=sorted), feats

    @staticmethod
    def _derived():
        D = {}
        for role in PM.NP_ROLES:
            for r in (role, NPM.of_role(role)):
                for name, conjs in NPM.derived(r).items(): D[f"{name}@{r}"] = [list(conj) for conj in conjs]
        D.update(PM.clause_module_rules()[1]); D.update(PM.clause_rules(())[1]); D.update(GM.derived()); D["agree.bad"] = PM.AGREE_DNF
        return D

    def universe(self, var):
        if var.startswith("w") and var[1:].isdigit(): return self.window_values
        if var == "pend": return sorted(R.PREPS) + [EMPTY]
        if var == "w0=w2": return [True, False]
        if var == "pp.count": return [0, 1, 2, 3]
        if PP_NOUN.match("slot." + var): return sorted(self.slot_vocab.get(var, ()), key=str)        # a filled role's noun: never EMPTY (the count says)
        return sorted(self.slot_vocab.get(var, ()), key=str) + [EMPTY]

    def expansion(self, q):
        """the DNF (conjunctions of base literal strings) of a predicate that is not a base literal, or None"""
        if q in self._expansion: return self._expansion[q]
        dnf = None
        if q in self.derived: dnf = self.derived[q]
        elif q.startswith("hit="):
            kind = q[4:].split(",")
            verbs = sorted((v for v in self.slot_vocab.get("verb", ()) if "+" not in v[0]), key=str)
            if kind[0] == "softener": dnf = [[f"w0.tbl={t}"] for t in R.SOFTENER]
            elif kind[0] == "adjprep": dnf = [[f"w0.tbl={a}", f"w1.tbl={b}"] for (a, b) in sorted(R.ADJ_PREP)]
            elif kind[0] == "idiom": dnf = [[f"w0.lemma={lem}"] + [f"w{j + 1}.tbl={t}" for j, t in enumerate(tt)] for (lem, tt) in sorted(R.VERB_IDIOM) if len(tt) == int(kind[1])]
            elif kind[0] == "phrasal": dnf = [[f"w0.lemma={lem}", f"w1.tbl={t}"] for (lem, t) in sorted(R.PHRASAL)]
            else:
                dnf = []
                for v in verbs:
                    for hit, keys in PM.hit_rows(v[0]):
                        if hit == q: dnf.append([f"slot.verb={v}"] + [f"w{j}.{kind_}={val}" for j, (kind_, val) in keys])
        elif q.startswith("np.") and q.endswith(".time"):
            role = q[3:-5]; dnf = [[f"np.time={role}", f"¬slot.{role}.noun2", f"¬slot.{role}.mono"]]
        elif q.startswith("use."):
            role = q[4:]; dnf = []
            for (ja, lemma), _ in sorted(PM.USE_TABLE.items(), key=str):
                for v in sorted((v for v in self.slot_vocab.get("verb", ()) if v[0] == lemma), key=str):
                    dnf.append([f"slot.verb={v}", f"slot.{role}.noun2={ja}"]); dnf.append([f"slot.verb={v}", f"slot.{role}.noun={ja}", f"¬slot.{role}.compound"])
        self._expansion[q] = dnf
        return dnf

PP_NOUN = re.compile(r"^slot\.(pp[123])\.noun(=.*)?$")

def var_of(q):
    """a base literal -> (variable, predicate): the window slot and the predicate on its token, the pending preposition, a slot and a
    predicate on its value (reg.* literals are the slot literals they name); pp.count=n is a predicate of the count variable"""
    if q.startswith("w") and q[1].isdigit() and q[2] == ".": return q[:2], q[3:]
    if q.startswith("pp.count"): return "pp.count", q
    if q == "w0=w2": return q, q
    if q.startswith("pend_prep="): return "pend", q
    if q.startswith("flag."): return q[5:], q
    if q.startswith("slot."):
        return q[5:].split("=", 1)[0], q
    if q.startswith("reg."):
        e, v = (q[4:].split("=", 1) + [None])[:2]
        e = {"adv": "adv1", "subj_cls": "subj.cls"}.get(e, e)
        if e in PM.NP_ROLES: e = f"{e}.noun"
        if v is None or v == "set": return e, f"slot.{e}"
        return e, f"slot.{e}={v}"
    if q == "verb.set": return "verb", "slot.verb"
    if q == "verb.phrasal": return "verb", q
    fam = GM.family_of(q)
    if fam: return fam[0], q
    raise KeyError(q)

def holds(pred, var, value, model):
    """the predicate on one value of its variable's universe (the window's values are predicate sets; a slot's are what it holds)"""
    if var.startswith("w") and var[1:].isdigit(): return pred in value                 # a window value is the set of its predicates
    if var == "w0=w2": return value is True
    if var == "pend": return value == pred[10:]
    if var == "pp.count": return value >= int(pred[10:]) if pred.startswith("pp.count>=") else value == int(pred[9:])
    if value == EMPTY: return False
    if pred == "verb.phrasal": return " " in value[0]
    return bool(GM.base_pred(pred, {var: value}))

# ---------------------------------------------------------------- the order of the variables: the window first, then the state
SLOT_RANK = {"ret": 0, "mret": 0, "pmode": 0, "verb": 1, "cop_kind": 1, "pp.count": 1}
def rank(var):
    """a fixed total order of the variables: the window slots, then the state — the return addresses and the print mode, the verb,
    the copula kind and the PP count, the flags, the registers, the tag slots, the noun-phrase slots — alphabetically within a kind"""
    if var.startswith("w") and var[1:].isdigit(): return (0, var)
    if var == "w0=w2": return (1, var)
    if var == "pend": return (2, var)
    if var in SLOT_RANK: return (3 + SLOT_RANK[var], var)
    if var in PM.FLAGS: return (5, var)
    if var in PM.ENUMS: return (6, var)
    if var.startswith("tag."): return (7, var)
    return (8, var)

# ---------------------------------------------------------------- formulas: a rule's condition in negation normal form
# ("lit", var, pred, truth) | ("and", frozenset) | ("or", frozenset) | T | F — a predicate with an expansion becomes its DNF, nested
# expansions included, and a negated one the negation pushed down to the literals; deciding a variable simplifies the formula
T, F = ("T",), ("F",)

def formula(q, truth, model, depth=0):
    if depth > 8: raise ValueError(f"expansion of {q} nests too deep")
    m = PP_NOUN.match(q) or (re.match(r"^np\.time=(pp[123])$", q))
    if m:                                                                           # the PPs fill in order: pp<k>.noun is filled iff the count reaches k
        role = m.group(1); k = int(role[2]); count = ("lit", "pp.count", f"pp.count>={k}", True)
        f = count if PP_NOUN.match(q) and not m.group(2) else f_and(frozenset([count, ("lit", f"{role}.noun", q, True)]))
        return f if truth else negate(f)
    dnf = model.expansion(q)
    if dnf is None:
        var, pred = var_of(q); return ("lit", var, pred, truth)
    conjs = frozenset(f_and(frozenset(formula(*_lit(l), model, depth + 1) for l in conj)) for conj in dnf)
    f = f_or(conjs)
    return f if truth else negate(f)

def f_and(fs):
    out = set()
    for f in fs:
        if f == F: return F
        if f == T: continue
        if f[0] == "and": out |= f[1]
        else: out.add(f)
    return T if not out else next(iter(out)) if len(out) == 1 else ("and", frozenset(out))

def f_or(fs):
    out = set()
    for f in fs:
        if f == T: return T
        if f == F: continue
        if f[0] == "or": out |= f[1]
        else: out.add(f)
    return F if not out else next(iter(out)) if len(out) == 1 else ("or", frozenset(out))

def negate(f):
    if f == T: return F
    if f == F: return T
    if f[0] == "lit": return ("lit", f[1], f[2], not f[3])
    if f[0] == "and": return f_or(frozenset(negate(g) for g in f[1]))
    return f_and(frozenset(negate(g) for g in f[1]))

_SUBST = {}                                   # (formula, variable, class) -> the substituted formula: a rule meets the same test on many paths

def substitute(f, var, label):
    """the formula with every literal on `var` decided by the class whose true predicates are `label` (a tuple)"""
    key = (f, var, label); r = _SUBST.get(key)
    if r is None: r = _SUBST[key] = _substitute(f, var, set(label))
    return r

def _substitute(f, var, true_preds):
    if f == T or f == F: return f
    if f[0] == "lit": return f if f[1] != var else (T if (f[2] in true_preds) == f[3] else F)
    parts = frozenset(_substitute(g, var, true_preds) for g in f[1])
    return f_and(parts) if f[0] == "and" else f_or(parts)

_VARS = {}; _PREDS = {}                       # per formula: its variables; per (formula, variable): the predicates it tests on it

def variables(f):
    v = _VARS.get(f)
    if v is None:
        if f[0] == "lit": v = frozenset([f[1]])
        elif f[0] in ("and", "or"): v = frozenset().union(*(variables(g) for g in f[1]))
        else: v = frozenset()
        _VARS[f] = v
    return v

def preds_on(f, var):
    key = (f, var); p = _PREDS.get(key)
    if p is None:
        if f[0] == "lit": p = frozenset([f[2]]) if f[1] == var else frozenset()
        elif f[0] in ("and", "or"): p = frozenset().union(*(preds_on(g, var) for g in f[1]))
        else: p = frozenset()
        _PREDS[key] = p
    return p

# ---------------------------------------------------------------- the diagram
class Node:
    __slots__ = ("kind", "var", "preds", "branches", "out", "id")
    def __init__(self, kind, var=None, preds=(), branches=None, out=None):
        self.kind = kind; self.var = var; self.preds = tuple(preds); self.branches = branches; self.out = out; self.id = None
    def __repr__(self): return dump(self).strip()

def _lit(l):
    return (l[1:], False) if l.startswith("¬") else (l, True)

class Table:
    """hash-consing and the build: structurally equal nodes are the same object, so equality is identity; a sub-diagram is built
    once per set of remaining rules"""
    def __init__(self, model=None):
        self.model = model or default_model(); self.nodes = {}; self.memo = {}
    def leaf(self, kind, out=None):
        key = (kind, repr(out)); n = self.nodes.get(key)
        if n is None: n = self.nodes[key] = Node(kind, out=out)
        return n
    def test(self, var, preds, branches):
        vals = list(branches.values())
        if all(b is vals[0] for b in vals): return vals[0]                          # a test with identical branches is removed
        key = ("TEST", var, tuple(preds), tuple(sorted((lab, id(b)) for lab, b in branches.items())))
        n = self.nodes.get(key)
        if n is None: n = self.nodes[key] = Node("TEST", var=var, preds=preds, branches=dict(branches))
        return n

    def normalize(self, lits, out):
        """a rule -> (its condition as a formula, out); None when the condition is False on every input"""
        f = f_and(frozenset(formula(q, truth, self.model) for q, truth in lits))
        return None if f == F else (f, out)

    def universe(self, var, preds):
        """the variable's values: the model's, plus a value a rule names that the model does not know"""
        vals = list(self.model.universe(var))
        if not (var.startswith("w") and var[1:].isdigit()) and var not in ("pend", "w0=w2"):
            known = {str(v) for v in vals}
            for p in preds:
                if p.startswith("slot.") and "=" in p and p.split("=", 1)[1] not in known: vals.append(p.split("=", 1)[1]); known.add(p.split("=", 1)[1])
        return vals

    def build(self, rules):
        """rules: frozenset of (formula, out) -> the root node"""
        if rules in self.memo: return self.memo[rules]
        vars_ = frozenset().union(*(variables(f) for f, out in rules))
        if not vars_:
            outs = sorted({repr(o): o for f, o in rules if f == T}.values(), key=repr)
            n = self.leaf(GAP) if not outs else self.leaf(TERM, outs[0]) if len(outs) == 1 else self.leaf(CONFLICT, tuple(outs))
            self.memo[rules] = n; return n
        var = min(vars_, key=rank)
        preds = sorted(frozenset().union(*(preds_on(f, var) for f, out in rules)))
        classes = {}
        for value in self.universe(var, preds):
            classes.setdefault(tuple(p for p in preds if holds(p, var, value, self.model)), True)
        on_var = [(f, out) for f, out in rules if var in variables(f)]; rest = [(f, out) for f, out in rules if var not in variables(f)]
        branches = {}
        for label in sorted(classes):
            sub = set(rest)
            for f, out in on_var:
                g = substitute(f, var, label)
                if g != F: sub.add((g, out))
            branches[label] = self.build(frozenset(sub))
        n = self.test(var, preds, branches)
        self.memo[rules] = n; return n

_default = [None]
def default_model():
    if _default[0] is None: _default[0] = Model()
    return _default[0]

def diagram(rules, order=None, table=None):
    """rules: iterable of (lits, out) or (prev, lits, out) for ONE control point -> the root node of its diagram"""
    table = table or Table()
    if len(_SUBST) > 2_000_000: _SUBST.clear()
    rs = set()
    for r in rules:
        nr = table.normalize(tuple(r[-2]), r[-1])
        if nr is not None: rs.add(nr)
    return table.build(frozenset(rs))

def equal(a, b): return a is b or dump(a) == dump(b)

def satisfiable(f, table):
    """some input makes the formula true (its one-rule diagram has a terminal)"""
    return any(l.kind == TERM for l in leaves(table.build(frozenset({(f, ("?", 0, ()))}))))

def leaves(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen: return
    seen.add(id(node))
    if node.kind != "TEST": yield node; return
    for b in node.branches.values(): yield from leaves(b, seen)

def complete(node): return all(l.kind != GAP for l in leaves(node))
def consistent(node): return all(l.kind != CONFLICT for l in leaves(node))

def full(var, pred): return pred if not (var.startswith("w") and var[1:].isdigit()) else f"{var}.{pred}"

def paths(node, prefix=()):
    """every root-to-leaf path -> (literals in diagram order, leaf): a branch contributes the predicates true in its class and the
    negation of the others the test mentions"""
    if node.kind != "TEST": yield prefix, node; return
    for label, b in node.branches.items():
        lits = tuple((full(node.var, p), p in label) for p in node.preds)
        yield from paths(b, prefix + lits)

def path_classes(lits):
    """a path's literals -> per variable, the predicates true in the class the path took (the test's other predicates are false)"""
    classes = {}
    for q, truth in lits:
        var, pred = var_of(q) if not (q.startswith("w") and q[1].isdigit() and q[2] == ".") else (q[:2], q[3:])
        classes.setdefault(var, set())
        if truth: classes[var].add(pred)
    return classes

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
SEP = " ∧ "
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

def fmt_out(out):
    nxt, k, acts = out
    return f"{nxt!r} k={k} {acts!r}"

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
        else:
            line(depth, f"{lead}{n.var} ⟨{SEP.join(n.preds)}⟩:")
            for label in sorted(n.branches, key=lambda l: (len(l) == 0, l)):
                node(n.branches[label], depth + 2, "* » " if not label else f"= {SEP.join(label)} » ")
    if name is not None: line(0, f"{name}:"); node(root, 1, "")
    else: node(root, 0, "")
    return "\n".join(out) + "\n"

def parse_out(s):
    m = re.match(r"^(.*) k=(\d+) (.*)$", s.strip())
    return (ast.literal_eval(m.group(1)), int(m.group(2)), ast.literal_eval(m.group(3)))

def parse(text, table=None):
    """the text form -> (name or None, root node)"""
    table = table or Table()
    lines = [(len(l) - len(l.lstrip(" ")), l.strip()) for l in text.splitlines() if l.strip()]
    name = None
    if lines and lines[0][0] == 0 and lines[0][1].endswith(":") and "⟨" not in lines[0][1] and not lines[0][1].startswith(("→", "#", "GAP", "CONFLICT")) and (len(lines) == 1 or lines[1][0] > 0):
        name = lines[0][1][:-1]; lines = lines[1:]
    labels = {}; pos = [0]
    def parse_node():
        indent, s = lines[pos[0]]; pos[0] += 1
        m = re.match(r"^(#\d+):\s*(.*)$", s); tag = None
        if m: tag, s = m.group(1), m.group(2)
        elif re.match(r"^#\d+$", s): return labels[s]
        if s.startswith("→ "): n = table.leaf(TERM, parse_out(s[2:]))
        elif s == "GAP": n = table.leaf(GAP)
        elif s.startswith("CONFLICT "): n = table.leaf(CONFLICT, tuple(parse_out(p) for p in s[len("CONFLICT "):].split(" ∥ ")))
        else:
            mm = re.match(r"^(\S+) ⟨(.*)⟩:$", s)
            if not mm: raise ValueError(f"cannot parse line: {s!r}")
            var = mm.group(1); preds = tuple(mm.group(2).split(SEP)) if mm.group(2) else (); branches = {}
            while pos[0] < len(lines) and lines[pos[0]][0] > indent:
                ind, t = lines[pos[0]]
                if t.startswith("* » "): label, rest = (), t[4:]
                else:
                    m2 = re.match(r"^= (.*?) » (.*)$", t); label, rest = tuple(m2.group(1).split(SEP)), m2.group(2)
                lines[pos[0]] = (ind, rest); branches[label] = parse_node()
            n = table.test(var, preds, branches)
        if tag: labels[tag] = n
        return n
    root = parse_node()
    return name, root

def shuffled(rules, seed=0):
    rs = list(rules); random.Random(seed).shuffle(rs); return rs
