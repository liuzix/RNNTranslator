"""parser_rules.Parser.tail_pp_adv — the PP machine — translated by hand into rules with complete guards.

After the verb and its objects, the rule parser loops over the tail of the clause: prepositional phrases, time and other
adverbs, duration phrases, comparisons, particles that complete a verb, tag questions, the copula adjective of a copular
verb.  Every branch of that loop is here, in the order the parser tests them, as an ordered decision at one position
(Decision): branch k fires when its condition holds and none of the earlier conditions does, each condition a derived
predicate (a DNF over base literals, nested definitions expanded), so that at every position exactly one rule fires and
the decision costs one step.  A branch whose action depends on the token (the particle of a preposition) continues at a
table position where one rule per value fires.  What the frame cannot hold — a coordinate clause, a purpose infinitive,
the hearsay suffix, an absolute clause — exits with FAIL and says why.  The machine has two exits: RET at the end of its
clause (the end of the sentence, or inside a subordinate clause — flag.sub — a comma, `or not` or a conjunction), and
RETSUB in a main clause at a conjunction that opens a subordinate clause after it, where the clause rules take over
(parser_matrix.clause_rules); a conjunction that is also a preposition reads as the preposition and while as a coordinator,
the parser's first choices, and a second subordinate clause is refused.

States are module-local names; parser_matrix.clause_module_rules maps them to control states ("M", "pp", name), "RET" and
"RETSUB" to the return states ("RETM", "pp", "") and ("RETM", "pp", "sub"), and "FAIL:<why>" to the cell's FAIL.  A rule may target ("NP0", role) to hand a noun phrase to np_module,
writing `ret` = "pp" so that the module's own rule at ("RET", role) takes the phrase back.  Base literals are the
predicates of parser_matrix.predicates() (w{k}.class=, w{k}.surf=, w{k}.vform=, w{k}.akind=, w{k}.dur=1, w{k}.hearsay=1,
w0=w2, hit=, reg.*, flag.*, pp.count=, pend_prep=, use.<role>, np.<role>.time, slot.<slot>[=value]) and the families of
gen_module (verb.lemma=, verb.vc= …).
"""
import rules_ja as R

DUR_DETS = ("a", "an", "several", "many", "few", "some")                                   # two years / a few days / several weeks
TIME_PREP = {"until": ("まで", "まで"), "till": ("まで", "まで"), "by": ("までに", "までに"), "since": ("から", "から"), "from": ("から", "から"),
             "before": ("の前に", "の前に"), "after": ("の後で", "の後で")}                     # a preposition before a time adverb
TIME_NP = {"at": ("に", "に"), "before": ("前に", "前に"), "after": ("後に", "後に"), "around": ("ごろに", "ごろに"), "by": ("までに", "までに")}   # before a clock time
PARTICLES = ("on", "off", "up", "down", "out", "away", "back", "in")
WH_BASE = sorted({v for v in R.WH.values()} | {("何時", "t"), ("どのくらい", "adv"), ("何歳", "adv"), ("いくつ", "o"), ("いくら", "o")}, key=str)
ROLES = ("pp1", "pp2", "pp3")

def verb_values(lemma):
    """the verb register values of a lemma, from the lexicon"""
    return [(lemma, e[1], e[2]) for e in R.LEX.get(lemma, []) if e[0] == "v"]

# ------------------------------------------------------------------------------------------------ DNFs over base literals
def negate(l): return l[1:] if l.startswith("¬") else "¬" + l
def lit(l): return [[l]]
def AND(*ds):
    """the conjunction of DNFs: distributed, contradictory conjunctions dropped, absorbed ones removed"""
    out = [[]]
    for d in ds:
        out = [dedupe(a + b) for a in out for b in d]
        out = [c for c in out if c is not None]
    return absorb(out)
def OR(*ds): return absorb([c for d in ds for c in d])
def NOT(d):
    """De Morgan: not (c1 or c2 …) = (not c1) and (not c2) …, each not c a disjunction of negated literals"""
    return AND(*[[[negate(l)] for l in c] for c in d])
def dedupe(c):
    seen = []
    for l in c:
        if negate(l) in seen: return None
        if l not in seen: seen.append(l)
    return seen
def absorb(d):
    d = [tuple(c) for c in d]; keep = []
    for c in d:
        if any(set(o) < set(c) for o in d) or c in keep: continue
        keep.append(c)
    return [list(c) for c in keep]

def starts_np(k):
    """parser_rules.starts_np on window slot k: pron / det / noun / num, or an adjective (not a verb or adverb) before a noun that is
    not a verb; the adjective case looks one slot further, which the window has only for k <= 2"""
    D = [[f"w{k}.class=pron"], [f"w{k}.class=det"], [f"w{k}.class=noun"], [f"w{k}.class=num"]]
    if k <= 2: D.append([f"w{k}.class=adj", f"¬w{k}.class=verb", f"¬w{k}.class=adv", f"w{k + 1}.class=noun", f"¬w{k + 1}.class=verb"])
    return D

# ------------------------------------------------------------------------------------------------ the builder
class Decision:
    """an ordered decision at one position: branch k fires when its condition holds and no earlier one does.  Each condition is
    registered as a derived predicate d.<module>.<state>.<k>; the rule for branch k reads it positively and the earlier ones
    negatively.  `otherwise` is the branch when no condition holds."""
    def __init__(self, module, state, derived):
        self.module, self.state, self.derived, self.branches = module, state, derived, []
    def branch(self, cond, nxt, k=0, acts=()):
        """cond: a DNF (a list of conjunctions of base literals), or a list of literals (one conjunction)"""
        if cond and isinstance(cond[0], str): cond = [list(cond)]
        self.branches.append((cond, nxt, k, tuple(acts))); return self
    def otherwise(self, nxt, k=0, acts=()):
        rules = []; names = []
        for j, (cond, nxt_j, k_j, acts_j) in enumerate(self.branches):
            name = f"d.{self.module}.{self.state}.{j}"; self.derived[name] = cond; names.append(name)
            rules.append((self.state, tuple([name] + [negate(n) for n in names[:-1]]), (nxt_j, k_j, acts_j)))
        rules.append((self.state, tuple(negate(n) for n in names), (nxt, k, tuple(acts))))
        return rules

def adv_write(src, k, nxt, state, extra=(), const=None):
    """an adverb joins the adverb register: adv1 when empty, else adv2, else the frame cannot hold a third — three rules at a
    position of their own; `src` a copy source, or `const` a value"""
    write = lambda slot: (slot, const, ("const",)) if const is not None else (slot, None, ("copy", src))
    return [(state, ("¬slot.adv1",), (nxt, k, (write("adv1"),) + tuple(extra))),
            (state, ("slot.adv1", "¬slot.adv2"), (nxt, k, (write("adv2"),) + tuple(extra))),
            (state, ("slot.adv1", "slot.adv2"), ("FAIL:three adverbs", 0, ()))]

def COUNT(n):
    """pp.count=n as base literals: the PPs fill in order, so the count is the first empty role"""
    roles = ("pp1", "pp2", "pp3")
    if n == 3: return lit("slot.pp3.noun")
    return AND(*([lit(f"slot.{r}.noun") for r in roles[:n]] + [lit(f"¬slot.{roles[n]}.noun")]))

def TIMENP(role):
    """the role's noun phrase is a clock time: its head ends in 時 and nothing follows it (parser_matrix's np.<role>.time)"""
    return AND(lit(f"np.time={role}"), lit(f"¬slot.{role}.noun2"), lit(f"¬slot.{role}.mono"))

def np_slots(role):
    from parser_matrix import np_slots as _np_slots
    return _np_slots(role)

def clears(role): return [(sl, None, ("clear",)) for sl in np_slots(role)]

# ------------------------------------------------------------------------------------------------ the rules
def derived():
    """the named conditions the rules read (filled in by rules(); the base DNFs here are shared by the other modules)"""
    return {}

def build():
    """-> (rules, derived): the PP machine's rules and every condition they read"""
    D = {}; rs = []
    SNP = {k: starts_np(k) for k in (0, 1, 2, 3)}
    END1 = OR(lit("w1.class=END"), lit("w1.class=end"), lit("w1.class=prep"), lit("w1.class=time"))      # what may follow a verb particle
    COORD0 = OR(*[AND(lit("w0.class=conj"), lit(f"w0.surf={w}")) for w in R.COORD])
    SUBORD0 = OR(*[AND(lit("w0.class=conj"), lit(f"w0.surf={w}")) for w in R.SUBORD if w not in R.PREPS])   # a subordinator that is also a preposition reads as the PP, the parser's default
    AUX1 = OR(*([lit(f"w1.surf={w}") for w in R.CONTRACTED] + [lit("w1.class=aux_do"), lit("w1.class=aux_be"), lit("w1.class=modal")]))
    DURDET0 = OR(lit("w0.class=num"), *[lit(f"w0.surf={w}") for w in DUR_DETS]); DURDET1 = OR(lit("w1.class=num"), *[lit(f"w1.surf={w}") for w in DUR_DETS])
    PREP_TIME = AND(lit("w0.class=prep"), lit("w1.class=time"), NOT(SNP[1]))
    PREP_NP = AND(lit("w0.class=prep"), SNP[1])
    PAFTER = OR(*[AND(lit(f"w0.surf={part}"), lit(f"slot.verb={v}")) for (lemma, part) in R.PARTICLE_AFTER for v in verb_values(lemma)])
    WHPREP = OR(*[AND(lit("w0.class=prep"), lit(f"slot.wh={whv}"), lit("w1.class=end")) for whv in WH_BASE])
    ANIMATE_OBJ = OR(lit("slot.obj.cls=h"), lit("slot.obj.cls=a"))
    MAKE_ADJ = AND(lit("w0.class=adj"), lit("slot.verb"), lit("verb.lemma=make"), lit("slot.obj.noun"), NOT(SNP[0]), ANIMATE_OBJ, lit("¬slot.obj.noun=自分"))
    d = Decision("pp", "PP0", D)
    d.branch(lit("w0.class=END"), "RET")
    d.branch(lit("w0.class=end"), "RET")
    d.branch(AND(lit("flag.sub"), OR(lit("w0.surf=,"), AND(lit("w0.surf=or"), lit("w1.class=not")), lit("w0.class=conj"))), "RET")   # a subordinate clause ends at a comma, at `or not`, at a conjunction
    d.branch(["w0.surf=,", "w1.class=pron", "w2.hearsay=1", "w3.class=end"], "FAIL:hearsay suffix")
    d.branch(AND(COORD0, lit("¬w0.surf=so")), "FAIL:coordinate clause")
    d.branch(AND(COORD0, lit("w0.surf=so"), SNP[1]), "FAIL:coordinate clause")
    d.branch(AND(SUBORD0, lit("¬slot.clause"), lit("¬slot.sublink")), "RETSUB")              # a subordinate clause after the main clause: the clause rules open it
    d.branch(SUBORD0, "FAIL:subordinate clause inside a clause, or a second one")
    d.branch(["w0.surf=,", "w1.class=conj"], "PP0", 1)
    d.branch(["w0.class=time", "¬slot.time"], "PP0", 1, [("time", None, ("copy", "time"))])
    d.branch(["w0.class=time", "slot.time"], "FAIL:two time adverbs")
    d.branch(["w0.surf=in", "w1.class=num", "w2.dur=1", "¬w2.surf=times", "¬w2.surf=time"], "PP_DURIN")
    d.branch(PREP_TIME, "PP_PREPTIME")
    d.branch(PREP_NP, "PP_PREPNP")
    d.branch(AND(PAFTER, END1), "PP_PAFTER")
    d.branch(WHPREP, "PP_WHPREP")
    d.branch(["w0.surf=to", "w1.class=verb", "w1.vform1=base", "slot.verb"], "FAIL:purpose clause")
    d.branch(AND(lit("w0.surf=as"), lit("w1.class=adv"), lit("w2.surf=as"), SNP[3]), "PP_AS1", 1)
    d.branch(["w0.surf=than", "w1.class=time"], "PP_THAN_T", 1, [("cmp_kind", "than", ("const",))] + clears("cmp"))
    d.branch(AND(lit("w0.surf=than"), lit("¬w1.class=time"), SNP[1]), ("NP0", "cmp"), 1, [("ret", "pp", ("const",)), ("cmp_kind", "than", ("const",))] + clears("cmp"))
    d.branch(AND(DURDET0, lit("w1.dur=1")), "PP_DUR")
    d.branch(AND(lit("w0.surf=for"), DURDET1, lit("w2.dur=1")), "PP0", 1)
    d.branch(AND(lit("w0.surf=,"), AUX1, lit("w2.surf='t"), lit("w3.class=pron")), "PP0", 4, [("tag", True, ("const",))])
    d.branch(AND(lit("w0.surf=,"), AUX1, lit("w2.class=pron"), NOT(SNP[3])), "PP0", 3, [("tag", True, ("const",))])
    d.branch(["w0.class=adj", "w1.surf=and", "w0=w2", "slot.verb", "verb.vc=change"], "PP_LONGER")
    d.branch(lit("w0.class=loc"), "PP_LOC")
    d.branch(["w0.surf=too", "w1.class=adv", "¬w1.class=verb"], "PP_TOO", 1)
    d.branch(lit("w0.class=adv"), "PP_ADV")
    d.branch(["w0.class=adj", "slot.verb", "verb.vc=copular"], "PP0", 1, [("cop_kind", "adj", ("const",)), ("cop_adj", None, ("copy", "adj_pair"))])
    d.branch(AND(MAKE_ADJ, OR(lit("w0.akind=a"), lit("w0.akind=na"))), "PP0", 1, [("verb", None, ("copy", "adj_caus"))])
    d.branch(lit("w0.surf=,"), "PP0", 1)
    rs += d.otherwise("FAIL:tail")
    # the tables: one rule per value, exclusive by the one-hot of the surface and of the PP count
    rs += adv_write("dur_in", 3, "PP0", "PP_DURIN")
    for n, role in enumerate(ROLES):
        D[f"d.pp.count{n}"] = COUNT(n)
        for w in sorted(R.PREPS):
            rs.append(("PP_PREPTIME", (f"w0.surf={w}", f"d.pp.count{n}"), (f"PP_TIME_{role}", 1, ((f"{role}_part", TIME_PREP.get(w, R.PREPS[w]), ("const",)),))))
            rs.append(("PP_PREPNP", (f"w0.surf={w}", f"d.pp.count{n}"), (("NP0", role), 1, (("ret", "pp", ("const",)), (f"{role}_part", R.PREPS[w], ("const",))))))
        rs.append((f"PP_TIME_{role}", ("w0.class=time",), ("PP0", 1, ((f"{role}.noun", None, ("copy", "time")), (f"{role}.cls", "t", ("const",))))))
        rs.append((f"PP_TIME_{role}", ("¬w0.class=time",), ("FAIL:time adverb expected", 0, ())))
        rs.append(("PP_LOC", (f"d.pp.count{n}",), ("PP0", 1, ((f"{role}_part", R.PREPS["at"], ("const",)), (f"{role}.noun", None, ("copy", "loc")), (f"{role}.cls", "t", ("const",))))))
    D["d.pp.count3"] = COUNT(3)
    for st in ("PP_PREPTIME", "PP_PREPNP", "PP_LOC"): rs.append((st, ("d.pp.count3",), ("FAIL:four prepositional phrases", 0, ())))
    not_prep = tuple(f"¬w0.surf={w}" for w in sorted(R.PREPS))                   # the tables' otherwise: a value outside the table is refused
    for st in ("PP_PREPTIME", "PP_PREPNP"): rs.append((st, not_prep + ("¬d.pp.count3",), ("FAIL:preposition expected", 0, ())))
    parts = sorted({part for (lemma, part) in R.PARTICLE_AFTER})
    for (lemma, part), (ja, conj) in sorted(R.PARTICLE_AFTER.items()):
        for v in verb_values(lemma):
            rs.append(("PP_PAFTER", (f"w0.surf={part}", f"slot.verb={v}"), ("PP0", 1, (("verb", (lemma + " " + part, ja, conj), ("const",)),))))
    for part in parts:
        verbs = [v for (lemma, p) in sorted(R.PARTICLE_AFTER) if p == part for v in verb_values(lemma)]
        rs.append(("PP_PAFTER", (f"w0.surf={part}",) + tuple(f"¬slot.verb={v}" for v in verbs), ("FAIL:no particle after this verb", 0, ())))
    rs.append(("PP_PAFTER", tuple(f"¬w0.surf={p}" for p in parts), ("FAIL:particle expected", 0, ())))
    for whv in WH_BASE:
        for w in sorted(R.PREPS):
            rs.append(("PP_WHPREP", (f"w0.surf={w}", f"slot.wh={whv}"), ("PP0", 1, (("wh", (whv[0] + R.PREPS[w][1 if whv[1] == "loc" else 0], "pp"), ("const",)),))))
    for w in sorted(R.PREPS): rs.append(("PP_WHPREP", (f"w0.surf={w}",) + tuple(f"¬slot.wh={whv}" for whv in WH_BASE), ("FAIL:no question word", 0, ())))
    rs.append(("PP_WHPREP", not_prep, ("FAIL:preposition expected", 0, ())))
    rs += adv_write("adv", 1, "PP_AS2", "PP_AS1")
    rs.append(("PP_AS2", ("w0.surf=as",), (("NP0", "cmp"), 1, (("ret", "pp", ("const",)), ("cmp_kind", "as", ("const",))) + tuple(clears("cmp")))))
    rs.append(("PP_AS2", ("¬w0.surf=as",), ("FAIL:as expected", 0, ())))
    rs.append(("PP_THAN_T", ("w0.class=time",), ("PP0", 1, (("cmp.noun", None, ("copy", "time")), ("cmp.cls", "t", ("const",))))))
    rs.append(("PP_THAN_T", ("¬w0.class=time",), ("FAIL:time adverb expected", 0, ())))
    rs += adv_write("dur", 2, "PP0", "PP_DUR")
    rs += adv_write(None, 3, "PP0", "PP_LONGER", extra=[("cop_kind", "adj", ("const",)), ("cop_adj", None, ("copy", "adj_pair_plain"))], const="ますます")
    rs += adv_write("adv_too", 1, "PP0", "PP_TOO")
    rs += adv_write("adv", 1, "PP0", "PP_ADV")
    # the noun phrase of a PP comes back: the use-verb feedback, the clock-time particle, the instrument, from … to, the dative
    for role in ROLES:
        rs.append(((("RET", role)), ("reg.ret=pp",), (f"PP_U_{role}", 0, (("ret", None, ("clear",)),))))
        # the use-verb feedback: use.<role> is a unit of the cell's second layer, read as a literal of the rule itself
        rs += [(f"PP_U_{role}", (f"use.{role}", "pend_prep=to", "¬slot.want"), (f"PP_T_{role}", 0, ((role, None, ("use",)),))),
               (f"PP_U_{role}", (f"¬use.{role}",), (f"PP_T_{role}", 0, ())),
               (f"PP_U_{role}", (f"use.{role}", "¬pend_prep=to"), (f"PP_T_{role}", 0, ())),
               (f"PP_U_{role}", (f"use.{role}", "pend_prep=to", "slot.want"), (f"PP_T_{role}", 0, ()))]
        t = Decision("pp", f"PP_T_{role}", D)
        for w, part in sorted(TIME_NP.items()):
            t.branch(AND(TIMENP(role), lit(f"slot.{role}.cls=t"), lit(f"pend_prep={w}")), f"PP_W_{role}", 0, [(f"{role}_part", part, ("const",))])
        rs += t.otherwise(f"PP_W_{role}")
        w_ = Decision("pp", f"PP_W_{role}", D)
        w_.branch(AND(lit("pend_prep=with"), lit("w0.vform=pp"), OR(lit("w1.class=END"), lit("w1.class=end"), lit("w1.surf=,"))), "FAIL:absolute clause")
        w_.branch(["pend_prep=with", f"slot.{role}.cls=t", "slot.verb", "¬verb.vc=comitative"], f"PP_F_{role}", 0, [(f"{role}_part", ("で", "で"), ("const",))])
        rs += w_.otherwise(f"PP_F_{role}")
        f = Decision("pp", f"PP_F_{role}", D)
        f.branch(["pend_prep=to", "flag.from_seen"], f"PP_D_{role}", 0, [(f"{role}_part", ("まで", "まで"), ("const",)), ("from_seen", None, ("clear",))])
        f.branch(["pend_prep=from"], f"PP_D_{role}", 0, [("from_seen", True, ("const",))])
        rs += f.otherwise(f"PP_D_{role}")
        dd = Decision("pp", f"PP_D_{role}", D)
        dd.branch([f"slot.{role}_part={R.PREPS['to']}", "slot.verb", "verb.vc=dative"], "PP0", 0,
                  [(sl2, None, ("move", sl)) for sl, sl2 in zip(np_slots(role), np_slots("iobj"))] + clears(role) + [(f"{role}_part", None, ("clear",))])
        rs += dd.otherwise("PP0")
    rs.append(((("RET", "cmp")), ("reg.ret=pp",), ("PP0", 0, (("ret", None, ("clear",)),))))
    return rs, D

_built = None
def rules():
    global _built
    if _built is None: _built = build()
    return list(_built[0])
def derived():
    rules(); return dict(_built[1])
