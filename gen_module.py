"""generate() translated by hand into printing rules in the normal form of np_module: per printing position, an ordered
decision over slot predicates whose terminals emit one token.

The rules read the flat frame — the same register file the compiled cell holds: one-hot blocks for the flags, the enum
registers (modal, want, cop_kind, wh, the particles, the verb) and the sub-slots of every noun-phrase role (det, poss, num,
adj1, adj2, noun, noun2, mono, cls) — and never the English.  A rule is (position, literals, (next_position, token)):
the token is None (a silent move), ("c", string) a constant morpheme, ("s", slot[, index]) the value of a slot, ("vf",
form, alt) a form of the frame's verb from the morphology table (dic, ta, te, a = negative stem, i = masu stem, e =
potential stem, caus = causative stem, pre = what precedes ある), or ("af", "stem", strip) the stem of the copula's
adjective (より stripped when a comparison follows).  Literals are the base predicates below ("slot.x", "slot.x=v",
"flag.f", and the families verb.*, wh.*, adj.*, subj.*, obj.*, adv.*: each a test on one slot's value) or a derived
predicate d.* defined as a DNF over base literals, as in np_module.

What generate() does with inner clauses, infinitives, control, fixed sentences and the idiom copula is out of scope
(parser_matrix.UNSUPPORTED_REGS): the frame cannot hold it, so render() refuses there.  Inside the scope the rules must
print exactly what generate() prints — pinned by the `printer` action (harness/eval_printer.py) on every sentence the rule
parser parses and on a sweep over the predicates the rules test.

    uv run python -c "import gen_module, rules_ja as R, parser_rules as PR; print(gen_module.PrintRules().render(PR.parse_bounded('i like the big dog .'.split())))"
"""
import rules_ja as R
from rules_ja import Pred, make_pred, adj_pred

NP_ROLES = ("subj", "obj", "iobj", "pp1", "pp2", "pp3", "cop", "cmp")
NP_SUBS = ("det", "poss", "num", "adj1", "adj2", "noun", "noun2", "cls", "mono", "compound")
FLAGS = ("past", "neg", "prog", "perf", "question", "imp", "please", "lets", "there", "tag", "coord", "sub", "neg_inner", "noless", "perf_had", "from_seen")
UNSUPPORTED = ("infinitive", "control", "embedded", "purpose_clause", "pre_clause", "manner", "so_that", "howabout", "exclaim", "fixed_out", "relative", "suffix")
# (a `folded` frame — one whose string came from an inner clause — prints like any other: the string is opaque to the rules; only the cell cannot hold it)
PRON6 = ("私", "彼", "彼女", "私たち", "彼ら", "あなた")            # generate(): a possessive coreferent with one of these subjects is dropped
NOPART = "∅"                                                      # an explicit empty object particle (generate's ""), a value a slot can hold
EMPTY = (None, "", False)

class Refuse(Exception):
    """the rules do not print this frame: no rule or more than one at a position, a form the morphology cannot give"""

# ------------------------------------------------------------------------------------------------ the flat frame
def filled(v): return v not in EMPTY

def flatten(c):
    """a parser_rules.Clause -> the flat frame the rules read; raises Refuse for a register the frame cannot hold"""
    for r in UNSUPPORTED:
        if filled(getattr(c, r, None)) and getattr(c, r) != []: raise Refuse(f"register {r}")
    if c.copula is not None and c.copula[0] not in ("adj", "np", "pp", "adv", "wh"): raise Refuse(f"copula kind {c.copula[0]}")
    if len(c.pp) > 3: raise Refuse("four PPs")
    F = {}
    for f in FLAGS:
        if getattr(c, f, False): F[f] = True
    for k in ("modal", "want", "wh", "purpose", "causative", "verb"):
        if filled(getattr(c, k)): F[k] = getattr(c, k)
    for k in ("degree", "time", "front"):
        if filled(getattr(c, k)): F[k] = getattr(c, k)
    if filled(c.adv): F["adv1"] = c.adv
    if c.obj_particle is not None: F["obj_particle"] = c.obj_particle or NOPART
    def put_np(role, s, cls=None):
        if not filled(s): return
        for p in PRON6:                                              # a pronoun possessive is its own sub-slot, as the cell holds it
            if s.startswith(p + "の") and len(s) > len(p) + 1: F[f"{role}.poss"] = p + "の"; s = s[len(p) + 1:]; break
        F[f"{role}.noun"] = s
        if cls: F[f"{role}.cls"] = cls
    put_np("subj", c.subj, c.subj_cls if c.subj else None)
    if c.subj and c.subj_num: F["subj.number"] = c.subj_num
    put_np("obj", c.obj); put_np("iobj", c.iobj)
    for k, (part, np) in enumerate(c.pp, 1): F[f"pp{k}_part"] = part; put_np(f"pp{k}", np)
    if c.copula is not None:
        kind = c.copula[0]; F["cop_kind"] = kind
        if kind == "adj": F["cop_adj"] = c.copula[1]
        elif kind == "np": put_np("cop", c.copula[1], c.copula[2])
        elif kind == "pp": F["cop_part"] = c.copula[1]; put_np("cop", c.copula[2])
        elif kind == "adv": F["cop_adv"] = c.copula[1]
    if c.compare is not None: F["cmp_kind"] = c.compare[0]; put_np("cmp", c.compare[1])
    return F

def np_string(F, role):
    if not filled(F.get(f"{role}.noun")): return None
    s = (F.get(f"{role}.det") or "") + (F.get(f"{role}.poss") or "")
    if F.get(f"{role}of.noun"): s += np_string(F, f"{role}of") + "の"
    s += "".join(F.get(f"{role}.{k}") or "" for k in ("num", "adj1", "adj2", "noun"))
    if F.get(f"{role}.noun2"): s += "の" + F[f"{role}.noun2"]
    if F.get(f"{role}.mono"): s += F[f"{role}.mono"]
    return s

# ------------------------------------------------------------------------------------------------ predicates
def adj_kind(pair):
    """the kind of predicate generate() makes of a copula adjective: ia, na, fixed, or v1 (〜ている)"""
    return adj_pred(pair[0], pair[1]).kind

def adj_stem(pair, strip):
    ja, k = pair
    if strip and ja.startswith("より"): ja = ja[2:]
    return adj_pred(ja, k).stem

def verb_pred(verb, alt=False):
    """the Pred of a verb register value; alt: the predicate generate() substitutes for 'understand' (分かる)"""
    lemma, ja, conj = verb
    if alt: return Pred("v5", "分かる")
    if conj == "fixed": return Pred("fixed", ja)
    if conj == "ia_fixed": return Pred("ia", ja[:-1])
    return make_pred(ja, conj)

def vform(verb, name, alt=False):
    """one named form of the verb's predicate; raises Refuse when the morphology has none"""
    p = verb_pred(verb, alt)
    try:
        if name == "dic": return p.dic
        if name == "pre": return verb[1][:-2] if verb[1].endswith("ある") else ""
        if p.kind not in ("v1", "v5", "suru", "kuru"): raise Refuse(f"form {name} of a {p.kind} predicate")
        if name == "ta": return p.ta()
        if name == "te": return p.te()
        if name == "a": return p.nai_stem()
        if name == "i": return p.masu_stem()
        if name == "e": return p.potential().stem
        if name == "caus": return p.causative().stem
        if name == "o": return p.volitional(False)
    except ValueError: raise Refuse(f"form {name} of {verb}")
    raise Refuse(f"unknown form {name}")

def _lemma_of(v): return v[0].split("+")[0]
# a family predicate is a test on the value of one slot; the cell compiles it as an OR over the slot's values that pass
FAMILIES = {
    "verb.lemma=": ("verb", lambda v, arg: v[0] == arg),
    "verb.conj=":  ("verb", lambda v, arg: v[2] == arg),
    "verb.vc=":    ("verb", lambda v, arg: _lemma_of(v) in R.VC.get(arg, ())),
    "verb.use":    ("verb", lambda v, arg: "+" in v[0]),
    "verb.ga":     ("verb", lambda v, arg: v[0] in R.GA_VERBS),
    "verb.stative": ("verb", lambda v, arg: v[0] in R.STATIVE),
    "verb.weather": ("verb", lambda v, arg: v[0] in ("rain", "snow")),
    "wh.kind=":    ("wh", lambda v, arg: v[1] == arg),
    "wh.text=":    ("wh", lambda v, arg: v[0] == arg),
    "adj.kind=":   ("cop_adj", lambda v, arg: adj_kind(v) == arg),
    "subj.animate": ("subj.cls", lambda v, arg: v in ("h", "a")),
    "subj.sore":   ("subj.noun", lambda v, arg: v == "それ"),
    "subj.negpol": ("subj.noun", lambda v, arg: v.endswith(("誰も", "何も"))),
    "obj.negpol":  ("obj.noun", lambda v, arg: v.endswith(("何も", "誰も"))),
    "adv.never":   ("adv1", lambda v, arg: "決して" in v),
    "adv2.never":  ("adv2", lambda v, arg: "決して" in v),
    "want.complement": ("want", lambda v, arg: v in R.COMPLEMENTS),
}

def family_of(name):
    """-> (slot, test, argument) when `name` is a family predicate, else None.  np.time=<role>: the role's head noun is a clock
    time (ends in 時), a family whose slot is named by the argument"""
    if name.startswith("np.time="): return f"{name[8:]}.noun", (lambda v, arg: isinstance(v, str) and v.endswith("時")), name[8:]
    for prefix, (slot, test) in FAMILIES.items():
        if name == prefix or (prefix.endswith("=") and name.startswith(prefix)): return slot, test, name[len(prefix):]
    return None

def base_pred(name, F):
    if name.startswith("flag."): return bool(F.get(name[5:]))
    if name.startswith("slot."):
        q = name[5:]
        if "=" in q:
            slot, v = q.split("=", 1); val = F.get(slot)
            return filled(val) and str(val) == v
        return filled(F.get(q))
    fam = family_of(name)
    if fam:
        slot, test, arg = fam; val = F.get(slot)
        return filled(val) and bool(test(val, arg))
    raise KeyError(name)

def eval_lit(lit, F, D):
    neg = lit.startswith("¬"); name = lit[1:] if neg else lit
    v = any(all(eval_lit(l, F, D) for l in conj) for conj in D[name]) if name.startswith("d.") else base_pred(name, F)
    return (not v) if neg else v

def derived():
    """the derived predicates: name -> DNF (a list of conjunctions of base literals)"""
    D = {}
    D["d.neg"] = [["flag.neg"], ["adv.never"], ["adv2.never"], ["subj.negpol"], ["obj.negpol"]]   # generate() turns these into c.neg before it prints the predicate
    D["d.whcop"] = [["slot.wh", "slot.cop_kind=wh"]]
    D["d.weather"] = [["subj.sore", "verb.weather"]]
    D["d.polite"] = [["flag.question"]]
    NEGS = ["flag.neg", "adv.never", "adv2.never", "subj.negpol", "obj.negpol"]; NOT_NEG = ["¬" + n for n in NEGS]
    # the tense generate() prints: c.past, except that a negative perfect takes the tense of its auxiliary (perf_had), and will takes none
    past_eff = [["flag.past", "¬flag.perf"], ["flag.past", "flag.perf"] + NOT_NEG] + [["flag.perf", "flag.perf_had", n] for n in NEGS]
    D["d.past_eff"] = [conj + ["¬slot.modal=will"] for conj in past_eff]
    D["d.past_raw"] = [["flag.past", "¬slot.modal=will"]]                                   # had better: the tense before the perfect correction
    D["d.will_conj"] = [conj + ["slot.modal=will", "¬flag.question", "¬slot.want"] for conj in past_eff]        # would: 〜だろう
    D["d.verbkind"] = [[f"verb.conj={c}"] for c in ("g", "i", "s", "k")]
    D["d.pp_state"] = [["slot.cop_kind=pp"], ["verb.vc=ni_location"]]                   # the PP takes its existence particle
    D["d.stative_prog"] = [["verb.stative", "¬verb.use", "¬slot.want", "¬flag.imp", "¬flag.lets", "¬slot.modal=can", "¬flag.perf"]]
    D["d.obj_default_ga"] = [["verb.ga", "¬slot.want"], ["slot.modal=can", "¬slot.want"]]
    D["d.obj_default_ni"] = [["verb.vc=ni_obj"]]
    for role in ("obj", "cmp"):                                                           # the possessive coreferent with the subject is dropped
        D[f"d.{role}_coref"] = [[f"slot.subj.noun={p}", "¬slot.subj.det", "¬slot.subj.poss", "¬slot.subj.num", "¬slot.subj.adj1", "¬slot.subj.noun2", "¬slot.subj.mono",
                                 f"slot.{role}.poss={p}の", f"¬slot.{role}.det"] for p in PRON6]
    return D

# ------------------------------------------------------------------------------------------------ the rules
class Build:
    def __init__(self): self.rules = []; self.n = 0
    def rule(self, state, lits, nxt, tokens=()):
        """one decision: at `state`, when `lits` hold, emit the tokens one per step and arrive at `nxt`"""
        tokens = list(tokens)
        if not tokens: self.rules.append((state, tuple(lits), (nxt, None))); return
        cur, cur_lits = state, tuple(lits)
        for j, tok in enumerate(tokens):
            tgt = nxt if j == len(tokens) - 1 else ("Q", self.n, j)
            self.rules.append((cur, cur_lits, (tgt, tok))); cur, cur_lits = tgt, ()
        self.n += 1
    def chain(self, state, tests, nxt):
        """a chain of silent single-literal tests: every literal must hold to reach nxt; the first that fails leaves at its own exit.
        tests: list of (literal, exit_state_if_false)"""
        cur = state
        for k, (lit, exit_state) in enumerate(tests):
            nx = nxt if k == len(tests) - 1 else (state, "t", k)
            self.rule(cur, [lit], nx); self.rule(cur, [negate(lit)], exit_state); cur = nx

def negate(lit): return lit[1:] if lit.startswith("¬") else "¬" + lit
def C(s): return ("c", s)
def S(slot, idx=None): return ("s", slot) if idx is None else ("s", slot, idx)
def VF(name, alt=False): return ("vf", name, alt)

# the finite suffix of each predicate kind, by (past, neg): plain and polite
PLAIN = {"V1C": ["る", "た", "ない", "なかった"], "IA": ["い", "かった", "くない", "くなかった"], "NA": ["だ", "だった", "ではない", "ではなかった"],
         "ARU": ["ある", "あった", "ない", "なかった"], "FIXED": ["", "", "", ""]}
POLITE = {"V1C": ["ます", "ました", "ません", "ませんでした"], "IA": ["いです", "かったです", "くないです", "くなかったです"],
          "NA": ["です", "でした", "ではありません", "ではありませんでした"], "ARU": ["あります", "ありました", "ありません", "ありませんでした"], "FIXED": ["です"] * 4}

def godan_suffixes(kana):
    """a godan ending kana -> its plain and polite finite suffixes, as Pred does them"""
    row = R.GODAN_ROWS[kana]; te = row[4]; ta = te[:-1] + ("だ" if te.endswith("で") else "た")
    return [kana, ta, row[0] + "ない", row[0] + "なかった"], [row[1] + s for s in POLITE["V1C"]]

def build_rules():
    """every printing rule, in the order of generate()'s template"""
    B = Build()
    # ---- the sentence: front, time, then the there-template or the main template
    B.rule("P0", ["slot.front"], "P_TIME", [S("front")]); B.rule("P0", ["¬slot.front"], "P_TIME")
    B.rule("P_TIME", ["slot.time"], "P_BRANCH", [S("time")]); B.rule("P_TIME", ["¬slot.time"], "P_BRANCH")
    B.rule("P_BRANCH", ["flag.there"], "T_PP1"); B.rule("P_BRANCH", ["¬flag.there"], "M_SUBJ0")
    # ---- noun phrases: det poss num adj1 adj2 noun [の noun2] [mono]
    built = set()
    def np(role, cont, coref=None):
        """the positions that print one noun-phrase role and continue at `cont` (one copy per role and continuation)"""
        P = lambda sub: ("NP", role, cont, sub)
        if (role, cont) in built: return P("det")
        built.add((role, cont))
        subs = ["det", "poss", "num", "adj1", "adj2"]
        for a, b in zip(subs, subs[1:] + ["noun"]):
            st, nx = P(a), P(b)
            if a == "poss":
                after = nx if role.endswith("of") else P("of")                 # after the possessive comes the of-phrase, if the role may have one
                if coref:                                                      # the possessive coreferent with the subject is dropped
                    B.rule(st, [f"slot.{role}.poss", f"¬{coref}"], after, [S(f"{role}.poss")]); B.rule(st, [f"slot.{role}.poss", coref], after); B.rule(st, [f"¬slot.{role}.poss"], after)
                else:
                    B.rule(st, [f"slot.{role}.poss"], after, [S(f"{role}.poss")]); B.rule(st, [f"¬slot.{role}.poss"], after)
                if after == P("of"):
                    # the of-phrase: the module once more on the role "<role>of", then の
                    B.rule(P("of"), [f"slot.{role}of.noun"], np(role + "of", P("ofno"))); B.rule(P("of"), [f"¬slot.{role}of.noun"], nx)
                    B.rule(P("ofno"), [], nx, [C("の")])
            else:
                B.rule(st, [f"slot.{role}.{a}"], nx, [S(f"{role}.{a}")]); B.rule(st, [f"¬slot.{role}.{a}"], nx)
        B.rule(P("noun"), [], P("noun2"), [S(f"{role}.noun")])
        B.rule(P("noun2"), [f"slot.{role}.noun2"], P("mono"), [C("の"), S(f"{role}.noun2")]); B.rule(P("noun2"), [f"¬slot.{role}.noun2"], P("mono"))
        B.rule(P("mono"), [f"slot.{role}.mono"], cont, [S(f"{role}.mono")]); B.rule(P("mono"), [f"¬slot.{role}.mono"], cont)
        return P("det")
    # ---- the there-template: PPs with the existence particle, the subject with が, the predicate, か
    for k in (1, 2, 3):
        nx = f"T_PP{k + 1}" if k < 3 else "T_COUNT"
        B.rule(f"T_PP{k}", [f"slot.pp{k}.noun"], np(f"pp{k}", f"T_PP{k}_PART"), []); B.rule(f"T_PP{k}", [f"¬slot.pp{k}.noun"], nx)
        B.rule(f"T_PP{k}_PART", [], nx, [S(f"pp{k}_part", 1)])
    B.rule("T_COUNT", ["wh.kind=count"], np("subj", "T_COUNT2"))
    B.rule("T_COUNT2", [], "PRED", [C("は"), C("いくつ")])
    B.rule("T_COUNT", ["¬wh.kind=count", "subj.negpol"], np("subj", "PRED"))
    B.rule("T_COUNT", ["¬wh.kind=count", "¬subj.negpol"], np("subj", "T_GA"))
    B.rule("T_GA", [], "PRED", [C("が")])
    # ---- the main template: the subject and its particle
    B.chain("M_SUBJ0", [("¬flag.imp", "M_WH_T"), ("¬flag.lets", "M_WH_T"), ("¬slot.want=mind_request", "M_WH_T"), ("¬slot.want=request", "M_WH_T"), ("¬d.weather", "M_WH_T")], "M_SUBJ1")
    B.rule("M_SUBJ1", ["d.whcop", "wh.text=何時", "subj.sore"], "M_WH_T")                       # it is what time: the dummy subject is not printed
    B.rule("M_SUBJ1", ["¬d.whcop", "wh.kind=s"], "M_WH_T", [S("wh", 0), C("が")])               # who came: the question word is the subject
    for ctx in (["d.whcop"], ["¬d.whcop", "¬wh.kind=s"]):
        B.rule("M_SUBJ1", ctx + (["¬wh.text=何時"] if ctx == ["d.whcop"] else []) + ["slot.subj.noun", "subj.negpol"], np("subj", "M_WH_T"))
        B.rule("M_SUBJ1", ctx + (["¬wh.text=何時"] if ctx == ["d.whcop"] else []) + ["slot.subj.noun", "¬subj.negpol"], np("subj", "M_SUBJ_WA"))
        B.rule("M_SUBJ1", ctx + (["¬wh.text=何時"] if ctx == ["d.whcop"] else []) + ["¬slot.subj.noun"], "M_WH_T")
    B.rule("M_SUBJ1", ["d.whcop", "wh.text=何時", "¬subj.sore", "slot.subj.noun", "subj.negpol"], np("subj", "M_WH_T"))
    B.rule("M_SUBJ1", ["d.whcop", "wh.text=何時", "¬subj.sore", "slot.subj.noun", "¬subj.negpol"], np("subj", "M_SUBJ_WA"))
    B.rule("M_SUBJ1", ["d.whcop", "wh.text=何時", "¬subj.sore", "¬slot.subj.noun"], "M_WH_T")
    B.rule("M_SUBJ_WA", [], "M_WH_T", [C("は")])
    # ---- question words before the arguments
    B.rule("M_WH_T", ["¬d.whcop", "wh.kind=t", "wh.text=何時"], "M_WH_LOC", [S("wh", 0), C("に")])
    B.rule("M_WH_T", ["¬d.whcop", "wh.kind=t", "¬wh.text=何時"], "M_WH_LOC", [S("wh", 0)])
    B.rule("M_WH_T", ["¬d.whcop", "wh.kind=pp"], "M_WH_LOC", [S("wh", 0)])
    B.rule("M_WH_T", ["¬d.whcop", "¬wh.kind=t", "¬wh.kind=pp"], "M_WH_LOC"); B.rule("M_WH_T", ["d.whcop"], "M_WH_LOC")
    B.rule("M_WH_LOC", ["¬d.whcop", "wh.kind=loc", "slot.cop_kind"], "M_WH_ADV", [C("どこ"), C("に")])
    B.rule("M_WH_LOC", ["¬d.whcop", "wh.kind=loc", "¬slot.cop_kind", "verb.vc=ni_where"], "M_WH_ADV", [C("どこ"), C("に")])
    B.rule("M_WH_LOC", ["¬d.whcop", "wh.kind=loc", "¬slot.cop_kind", "¬verb.vc=ni_where"], "M_WH_ADV", [C("どこ"), C("で")])
    B.rule("M_WH_LOC", ["¬d.whcop", "¬wh.kind=loc"], "M_WH_ADV"); B.rule("M_WH_LOC", ["d.whcop"], "M_WH_ADV")
    B.rule("M_WH_ADV", ["wh.kind=adv", "¬slot.cop_kind", "wh.text=どう"], "M_WH_COUNT", [C("どうやって")])
    B.rule("M_WH_ADV", ["wh.kind=adv", "¬slot.cop_kind", "¬wh.text=どう"], "M_WH_COUNT", [S("wh", 0)])
    B.rule("M_WH_ADV", ["wh.kind=adv", "slot.cop_kind"], "M_WH_COUNT"); B.rule("M_WH_ADV", ["¬wh.kind=adv"], "M_WH_COUNT")
    B.rule("M_WH_COUNT", ["wh.kind=count"], "M_PP1", [C("いくつの"), S("wh", 0), C("を")]); B.rule("M_WH_COUNT", ["¬wh.kind=count"], "M_PP1")
    # ---- prepositional phrases, the copula's PP, the comparison
    for k in (1, 2, 3):
        nx = f"M_PP{k + 1}" if k < 3 else "M_COPPP"
        B.rule(f"M_PP{k}", [f"slot.pp{k}.noun"], np(f"pp{k}", f"M_PP{k}_PART")); B.rule(f"M_PP{k}", [f"¬slot.pp{k}.noun"], nx)
        B.rule(f"M_PP{k}_PART", ["d.pp_state"], nx, [S(f"pp{k}_part", 1)]); B.rule(f"M_PP{k}_PART", ["¬d.pp_state"], nx, [S(f"pp{k}_part", 0)])
    B.rule("M_COPPP", ["slot.cop_kind=pp"], np("cop", "M_COPPP_PART")); B.rule("M_COPPP", ["¬slot.cop_kind=pp"], "M_CMP")
    B.rule("M_COPPP_PART", [], "M_CMP", [S("cop_part", 1)])
    B.rule("M_CMP", ["slot.cmp.noun"], np("cmp", "M_CMP_PART", coref="d.cmp_coref")); B.rule("M_CMP", ["¬slot.cmp.noun"], "M_IOBJ")
    B.rule("M_CMP_PART", ["slot.cmp_kind=than"], "M_IOBJ", [C("より")]); B.rule("M_CMP_PART", ["slot.cmp_kind=noless"], "M_IOBJ", [C("に劣らず")])
    B.rule("M_CMP_PART", ["¬slot.cmp_kind=than", "¬slot.cmp_kind=noless", "d.neg"], "M_IOBJ", [C("ほど")])
    B.rule("M_CMP_PART", ["¬slot.cmp_kind=than", "¬slot.cmp_kind=noless", "¬d.neg"], "M_IOBJ", [C("と同じくらい")])
    # ---- indirect object, object and its particle, the object question word, the adverb
    B.rule("M_IOBJ", ["slot.iobj.noun"], np("iobj", "M_IOBJ_PART")); B.rule("M_IOBJ", ["¬slot.iobj.noun"], "M_OBJ")
    B.rule("M_IOBJ_PART", [], "M_OBJ", [C("に")])
    B.rule("M_OBJ", ["slot.obj.noun"], np("obj", "M_OBJ_PART", coref="d.obj_coref")); B.rule("M_OBJ", ["¬slot.obj.noun"], "M_WH_O")
    B.rule("M_OBJ_PART", ["obj.negpol"], "M_WH_O")
    B.rule("M_OBJ_PART", ["¬obj.negpol", "slot.obj_particle"], "M_WH_O", [S("obj_particle")])
    B.rule("M_OBJ_PART", ["¬obj.negpol", "¬slot.obj_particle", "slot.verb", "d.obj_default_ga"], "M_WH_O", [C("が")])
    B.rule("M_OBJ_PART", ["¬obj.negpol", "¬slot.obj_particle", "slot.verb", "¬d.obj_default_ga", "d.obj_default_ni"], "M_WH_O", [C("に")])
    B.rule("M_OBJ_PART", ["¬obj.negpol", "¬slot.obj_particle", "slot.verb", "¬d.obj_default_ga", "¬d.obj_default_ni"], "M_WH_O", [C("を")])
    B.rule("M_OBJ_PART", ["¬obj.negpol", "¬slot.obj_particle", "¬slot.verb"], "M_WH_O", [C("を")])
    B.rule("M_WH_O", ["wh.kind=o", "¬slot.cop_kind"], "M_ADV", [S("wh", 0), C("を")]); B.rule("M_WH_O", ["wh.kind=o", "slot.cop_kind"], "M_ADV"); B.rule("M_WH_O", ["¬wh.kind=o"], "M_ADV")
    B.rule("M_ADV", ["slot.adv1"], "M_ADV2", [S("adv1")]); B.rule("M_ADV", ["¬slot.adv1"], "M_ADV2")      # the cell holds a second adverb in adv2
    B.rule("M_ADV2", ["slot.adv2"], "M_TAIL", [S("adv2")]); B.rule("M_ADV2", ["¬slot.adv2"], "M_TAIL")
    # ---- the tail: a question-word copula (何ですか), or the copula adverb, the degree and the predicate
    whk = ["o", "s", "adv", "loc", "t", "count"]
    for k in whk: B.rule("M_TAIL", ["d.whcop", f"wh.kind={k}"], ("FIN", "NA", "whcop"), [S("wh", 0)])
    B.rule("M_TAIL", ["d.whcop"] + [f"¬wh.kind={k}" for k in whk], "M_COPADV"); B.rule("M_TAIL", ["¬d.whcop"], "M_COPADV")
    B.rule("M_COPADV", ["slot.cop_kind=adv"], "M_DEGREE", [S("cop_adv")]); B.rule("M_COPADV", ["¬slot.cop_kind=adv"], "M_DEGREE")
    B.rule("M_DEGREE", ["slot.degree"], "M_DEGREE2", [S("degree")]); B.rule("M_DEGREE", ["¬slot.degree"], "M_DEGREE2")
    B.rule("M_DEGREE2", ["slot.degree2"], "PRED", [S("degree2")]); B.rule("M_DEGREE2", ["¬slot.degree2"], "PRED")
    # ---- after the predicate: か / ね, then 。
    B.rule("Q", ["flag.there", "wh.kind=count"], "END", [C("か")])
    B.rule("Q", ["flag.there", "¬wh.kind=count", "flag.question"], "END", [C("か")]); B.rule("Q", ["flag.there", "¬wh.kind=count", "¬flag.question"], "END")
    B.rule("Q", ["¬flag.there", "flag.tag"], "END", [C("ね")]); B.rule("Q", ["¬flag.there", "¬flag.tag", "flag.question"], "END", [C("か")])
    B.rule("Q", ["¬flag.there", "¬flag.tag", "¬flag.question"], "END")
    B.rule("END", [], "DONE", [C("。")])
    build_predicate(B, np)
    return reachable(B.rules)

def reachable(rules):
    """the rules of the positions reachable from P0, in their order: a kind-and-mode combination no path needs is dropped"""
    by_pos = {}
    for r in rules: by_pos.setdefault(r[0], []).append(r)
    seen = {"P0"}; frontier = ["P0"]
    while frontier:
        for pos, lits, (nxt, tok) in by_pos.get(frontier.pop(), []):
            if nxt not in seen and nxt != "DONE" and not (isinstance(nxt, str) and nxt.startswith("FAIL:")): seen.add(nxt); frontier.append(nxt)
    return [r for r in rules if r[0] in seen]

def build_predicate(B, np):
    """predicate_form(): the kind of predicate, the transformations want / prog / modal / lets / imp / please apply to it, and
    its finite form — the stem is emitted when the kind is fixed and the suffixes follow, one morpheme per step"""
    verbkind = ["d.verbkind"]
    # PRED: which predicate
    B.rule("PRED", ["flag.there", "subj.animate"], ("FIN", "V1C", "plain"), [C("い")])      # there is: no modal reaches the predicate
    B.rule("PRED", ["flag.there", "¬subj.animate"], ("FIN", "ARU", "plain"))
    B.rule("PRED", ["¬flag.there", "slot.cop_kind=wh"], "Q")
    B.rule("PRED", ["¬flag.there", "slot.cop_kind=adj", "slot.cmp.noun"], "PRED_ADJ", [("af", "stem", True)])
    B.rule("PRED", ["¬flag.there", "slot.cop_kind=adj", "¬slot.cmp.noun"], "PRED_ADJ", [("af", "stem", False)])
    for k in ("ia", "na", "fixed", "v1"): B.rule("PRED_ADJ", [f"adj.kind={k}"], ("K", {"ia": "IA", "na": "NA", "fixed": "FIXED", "v1": "V1C"}[k]))
    B.rule("PRED", ["¬flag.there", "slot.cop_kind=np"], np("cop", ("K", "NA", "noun")))
    for kind in ("pp", "adv"):
        B.rule("PRED", ["¬flag.there", f"slot.cop_kind={kind}", "subj.animate"], ("K", "V1C"), [C("い")])
        B.rule("PRED", ["¬flag.there", f"slot.cop_kind={kind}", "¬subj.animate"], ("K", "ARU"))
    B.rule("PRED", ["¬flag.there", "¬slot.cop_kind", "¬slot.verb"], "Q")
    B.rule("PRED", ["¬flag.there", "¬slot.cop_kind", "slot.verb", "verb.conj=aru"], ("K", "ARU"), [VF("pre")])
    B.rule("PRED", ["¬flag.there", "¬slot.cop_kind", "slot.verb", "verb.conj=ia_fixed"], "FAIL:control")
    B.rule("PRED", ["¬flag.there", "¬slot.cop_kind", "slot.verb", "¬verb.conj=aru", "¬verb.conj=ia_fixed"], "V_WANT")
    # ---- a copula or there predicate: only must and the finite form apply (K, kind[, noun])
    for kind in ("V1C", "IA", "NA", "FIXED", "ARU"):
        for extra in ((), ("noun",)):
            st = ("K", kind) + extra
            if extra and kind != "NA": continue
            if kind == "V1C":
                B.rule(st, ["slot.modal=must", "d.neg"], "MUST_TAIL", [C("て"), C("はいけない")]); B.rule(st, ["slot.modal=must", "¬d.neg"], "MUST_TAIL", [C("なければならない")])
            elif kind == "NA":
                B.rule(st, ["slot.modal=must"], "Q", [C("の" if extra else "な"), C("はずだ")])
            else:
                B.rule(st, ["slot.modal=must"], "Q", [C(PLAIN[kind][0]), C("はずだ")])
            B.rule(st, ["¬slot.modal=must"], ("FIN", kind, "plain"))
    # ---- a verb predicate: the complement transformation
    v = "V_WANT"
    B.rule(v, ["slot.want=tai"], ("V2", "IA"), [VF("i"), C("た")]); B.rule(v, ["slot.want=wish"], ("V2", "IA"), [VF("i"), C("た")])
    for w, tail in (("promise", "と約束する"), ("try", "ようにする"), ("decide", "ことにした")):
        B.rule(v, [f"slot.want={w}", "flag.neg_inner"], ("V2", "FIXED"), [VF("a"), C("ない"), C(tail)])
    compl = []
    for w, (tmpl, kind) in R.COMPLEMENTS.items():
        if w in ("promise", "try", "decide"): lits = [f"slot.want={w}", "¬flag.neg_inner"]
        else: lits = [f"slot.want={w}"]
        if w in ("tai", "wish"): continue
        compl.append(w)
        for inner_neg in ((False, True) if "{inner}" in tmpl else (False,)):
            toks, kind2 = template_tokens(tmpl, kind, inner_neg)
            B.rule(v, lits + (["flag.neg_inner"] if inner_neg else ["¬flag.neg_inner"] if "{inner}" in tmpl else []), ("V2",) + (kind2 if isinstance(kind2, tuple) else (kind2,)), toks)
    others = [f"¬slot.want={w}" for w in ("tai", "wish", "better", "neednot", "mind_request", "request", "canthave")] + [f"¬slot.want={w}" for w in compl if w not in ("promise", "try", "decide")]
    others += [f"¬slot.want={w}" for w in ("promise", "try", "decide")]
    B.rule(v, ["slot.want=better", "d.neg"], ("V2", "IA_NN"), [VF("a"), C("ない"), C("ほうがい")]); B.rule(v, ["slot.want=better", "¬d.neg"], ("V2", "IA_NN"), [VF("ta"), C("ほうがい")])
    B.rule(v, ["slot.want=neednot", "d.polite"], "Q", [VF("dic"), C("必要はありません")]); B.rule(v, ["slot.want=neednot", "¬d.polite"], "Q", [VF("dic"), C("必要はない")])
    B.rule(v, ["slot.want=mind_request"], "Q", [VF("te"), C("もらえません")]); B.rule(v, ["slot.want=request"], "Q", [VF("te"), C("くださいません")])
    B.rule(v, ["slot.want=canthave"], "Q", [VF("ta"), C("はずがない")])
    B.rule(v, others + ["¬verb.conj=fixed"], ("V2", "V"))       # no want, or one that leaves the governed verb as it is
    B.rule(v, others + ["verb.conj=fixed"], ("V2", "FIXED"), [VF("dic")])
    # ---- the special verbs (like -> 好き ...), stative -> progressive, perf+neg -> progressive, the progressive
    B.rule(("V2", "V"), ["¬slot.want", "¬verb.use", "verb.lemma=like"], ("V3", "NA"), [C("好き")])
    B.rule(("V2", "V"), ["¬slot.want", "¬verb.use", "verb.lemma=want", "slot.obj.noun"], ("V3", "IA"), [C("欲し")])
    B.rule(("V2", "V"), ["¬slot.want", "¬verb.use", "verb.lemma=need"], ("V3", "NA"), [C("必要")])
    B.rule(("V2", "V"), ["¬slot.want", "¬verb.use", "verb.lemma=hate"], ("V3", "NA"), [C("嫌い")])
    B.rule(("V2", "V"), ["¬slot.want", "¬verb.use", "verb.lemma=understand"], ("V3", "VA"))
    special = ["¬verb.lemma=like", "¬verb.lemma=need", "¬verb.lemma=hate", "¬verb.lemma=understand"]
    B.rule(("V2", "V"), ["¬slot.want", "¬verb.use", "verb.lemma=want", "¬slot.obj.noun"], ("V3", "V"))
    B.rule(("V2", "V"), ["¬slot.want", "¬verb.use", "¬verb.lemma=want"] + special, ("V3", "V"))
    B.rule(("V2", "V"), ["¬slot.want", "verb.use"], ("V3", "V")); B.rule(("V2", "V"), ["slot.want"], ("V3", "V"))
    for k in ("IA", "IA_NN", "FIXED", "NA", "V1C", "V5C"):
        pass
    # the progressive applies to a verb kind without want: stative, perf+neg, or prog
    for src in ("V", "VA"):
        alt = src == "VA"
        prog = ["¬slot.want"]
        B.rule(("V3", src), prog + ["d.stative_prog"], ("V4", "V1C"), [VF("te", alt), C("い")])
        B.rule(("V3", src), prog + ["¬d.stative_prog", "flag.perf", "d.neg"], ("V4", "V1C"), [VF("te", alt), C("い")])
        B.rule(("V3", src), prog + ["¬d.stative_prog", "¬flag.perf", "flag.prog"], ("V4", "V1C"), [VF("te", alt), C("い")])
        B.rule(("V3", src), prog + ["¬d.stative_prog", "flag.perf", "¬d.neg", "flag.prog"], ("V4", "V1C"), [VF("te", alt), C("い")])
        B.rule(("V3", src), prog + ["¬d.stative_prog", "flag.perf", "¬d.neg", "¬flag.prog"], ("V4", src))
        B.rule(("V3", src), prog + ["¬d.stative_prog", "¬flag.perf", "¬flag.prog"], ("V4", src))
        B.rule(("V3", src), ["slot.want"], ("V4", src))
    for k in ("IA", "NA"): B.rule(("V3", k), [], ("V4", k))
    # the complement results arrive at V2 with their kind: no special verbs, no progressive (want is set)
    for k in ("IA", "IA_NN", "FIXED", "NA", "V1C"): B.rule(("V2", k), [], ("V4", k))
    for kana in sorted({t[-1] for (t, kd) in R.COMPLEMENTS.values() if kd == "v5"}): B.rule(("V2", "V5C", kana), [], ("V4", "V5C", kana))
    # ---- modals: can (potential), must, should, may; then lets / imp / please / will-conjecture / finite
    def kinds_after_v4():
        yield ("V",); yield ("VA",); yield ("V1C",); yield ("IA",); yield ("IA_NN",); yield ("NA",); yield ("FIXED",)
        for kana in sorted({t[-1] for (t, kd) in R.COMPLEMENTS.values() if kd == "v5"}): yield ("V5C", kana)
    for kd in kinds_after_v4():
        st = ("V4",) + kd; kind = kd[0]; alt = kind == "VA"; verbal = kind in ("V", "VA", "V1C", "V5C")
        nxt_base = ("V5",) + kd
        # can
        if kind in ("V", "VA"): B.rule(st, ["slot.modal=can"], ("V5", "V1C"), [VF("e", alt)])
        elif kind == "V1C": B.rule(st, ["slot.modal=can"], ("V5", "V1C"), [C("られ")])
        elif kind == "V5C": B.rule(st, ["slot.modal=can"], ("V5", "V1C"), [C(R.GODAN_ROWS[kd[1]][2])])
        else: B.rule(st, ["slot.modal=can"], nxt_base)
        # must
        if kind in ("V", "VA"):
            B.rule(st, ["slot.modal=must", "d.neg"], "MUST_TAIL", [VF("te", alt), C("はいけない")]); B.rule(st, ["slot.modal=must", "¬d.neg"], "MUST_TAIL", [VF("a", alt), C("なければならない")])
        elif kind == "V1C":
            B.rule(st, ["slot.modal=must", "d.neg"], "MUST_TAIL", [C("て"), C("はいけない")]); B.rule(st, ["slot.modal=must", "¬d.neg"], "MUST_TAIL", [C("なければならない")])
        elif kind == "V5C":
            row = R.GODAN_ROWS[kd[1]]
            B.rule(st, ["slot.modal=must", "d.neg"], "MUST_TAIL", [C(row[4]), C("はいけない")]); B.rule(st, ["slot.modal=must", "¬d.neg"], "MUST_TAIL", [C(row[0]), C("なければならない")])
        elif kind == "NA": B.rule(st, ["slot.modal=must"], "Q", [C("な"), C("はずだ")])
        else: B.rule(st, ["slot.modal=must"], "Q", [C(PLAIN["IA" if kind in ("IA", "IA_NN") else kind][0]), C("はずだ")])
        # should: dic + べき -> na
        dic = {"V": [VF("dic")], "VA": [VF("dic", True)], "V1C": [C("る")], "IA": [], "IA_NN": [], "NA": [], "FIXED": []}.get(kind, [C(kd[1])] if kind == "V5C" else [])
        B.rule(st, ["slot.modal=should"], ("V5", "NA"), dic + [C("べき")])
        B.rule(st, ["slot.modal=may"], ("V5", "FIXED"), dic + [C("かもしれない")])
        B.rule(st, ["¬slot.modal=can", "¬slot.modal=must", "¬slot.modal=should", "¬slot.modal=may"], nxt_base)
    B.rule("MUST_TAIL", ["d.polite"], "Q", [C("です")]); B.rule("MUST_TAIL", ["¬d.polite"], "Q")
    for kd in kinds_after_v4():
        st = ("V5",) + kd; kind = kd[0]; alt = kind == "VA"; verbal = kind in ("V", "VA", "V1C", "V5C")
        # lets
        if kind in ("V", "VA"): B.rule(st, ["flag.lets"], "Q", [VF("i", alt), C("ましょう")])
        elif kind == "V1C": B.rule(st, ["flag.lets"], "Q", [C("ましょう")])
        elif kind == "V5C": B.rule(st, ["flag.lets"], "Q", [C(R.GODAN_ROWS[kd[1]][1]), C("ましょう")])
        else: B.rule(st, ["flag.lets"], ("FIN",) + (("IA",) if kind == "IA_NN" else kd) + ("lets",))
        # imp
        te = {"V": [VF("te")], "VA": [VF("te", True)], "V1C": [C("て")], "IA": [C("くて")], "IA_NN": [C("くて")], "NA": [C("で")]}.get(kind, [C(R.GODAN_ROWS[kd[1]][4])] if kind == "V5C" else None)
        a = {"V": [VF("a")], "VA": [VF("a", True)], "V1C": []}.get(kind, [C(R.GODAN_ROWS[kd[1]][0])] if kind == "V5C" else None)
        caus = {"V": [VF("caus")], "VA": [VF("caus", True)], "V1C": [C("させ")]}.get(kind, [C(R.GODAN_ROWS[kd[1]][0] + "せ")] if kind == "V5C" else None)
        if verbal:
            B.rule(st, ["¬flag.lets", "flag.imp", "slot.causative"], "Q", caus + [C("てください")])
            B.rule(st, ["¬flag.lets", "flag.imp", "¬slot.causative", "d.neg"], "Q", a + [C("ないでください")])
            B.rule(st, ["¬flag.lets", "flag.imp", "¬slot.causative", "¬d.neg"], "Q", te + [C("ください")])
        elif te is not None:
            B.rule(st, ["¬flag.lets", "flag.imp", "¬slot.causative", "¬d.neg"], "Q", te + [C("ください")])
            B.rule(st, ["¬flag.lets", "flag.imp", "¬slot.causative", "d.neg"], "FAIL:imperative")
            B.rule(st, ["¬flag.lets", "flag.imp", "slot.causative"], "FAIL:imperative")
        else: B.rule(st, ["¬flag.lets", "flag.imp"], "FAIL:imperative")
        # please (the frame's verb is set on every path through PRED's verb branch)
        if te is not None: B.rule(st, ["¬flag.lets", "¬flag.imp", "flag.please"], "Q", te + [C("ください")])
        else: B.rule(st, ["¬flag.lets", "¬flag.imp", "flag.please"], "FAIL:please")
        # would: finite(False, neg, False) + だろう
        fk = ("IA",) if kind == "IA_NN" else kd
        B.rule(st, ["¬flag.lets", "¬flag.imp", "¬flag.please", "d.will_conj"], ("FIN",) + fk + ("conj" if kind != "IA_NN" else "conj_nn",))
        B.rule(st, ["¬flag.lets", "¬flag.imp", "¬flag.please", "¬d.will_conj"], ("FIN",) + fk + ("flags" if kind != "IA_NN" else "noneg",))
    # ---- FIN: the finite suffix of a kind under (past, neg, polite); modes: flags (frame), noneg (want=better), conj (plain
    #      present, + だろう), whcop (+ か), lets (polite, present, affirmative)
    for kd in (("V",), ("VA",), ("V1C",), ("IA",), ("NA",), ("FIXED",), ("ARU",)) + tuple(("V5C", kana) for kana in sorted({t[-1] for (t, kd) in R.COMPLEMENTS.values() if kd == "v5"})):
        kind = kd[0]; alt = kind == "VA"
        for mode in ("flags", "plain", "noneg", "conj", "conj_nn", "whcop", "lets"):
            st = ("FIN",) + kd + (mode,); nxt = "Q"
            tail = [C("か")] if mode == "whcop" else [C("だろう")] if mode.startswith("conj") else []
            if mode == "whcop": nxt = "END"
            cases = []
            for past in (False, True):
                for neg in (False, True):
                    for polite in (False, True):
                        lits = []
                        if mode in ("conj", "conj_nn"):
                            if past or polite or (mode == "conj_nn" and neg): continue
                            lits += ["d.neg"] if neg else ["¬d.neg"]
                        elif mode == "lets":
                            if past or neg or not polite: continue
                        else:
                            past_lit = {"flags": "d.past_eff", "noneg": "d.past_raw"}.get(mode, "flag.past")
                            lits += [past_lit] if past else ["¬" + past_lit]
                            if mode == "noneg":
                                if neg: continue
                            else: lits += ["d.neg"] if neg else ["¬d.neg"]
                            lits += ["d.polite"] if polite else ["¬d.polite"]
                        idx = 2 * neg + past
                        if kind in ("V", "VA"):
                            toks = [VF("i", alt), C(POLITE["V1C"][idx])] if polite else ([VF("a", alt), C(PLAIN["V1C"][idx])] if neg else [VF("ta" if past else "dic", alt)])
                        elif kind == "V5C":
                            plain, pol = godan_suffixes(kd[1]); toks = [C((pol if polite else plain)[idx])]
                        else:
                            toks = [C((POLITE if polite else PLAIN)[kind][idx])]
                        toks = [t for t in toks if t != ("c", "")]
                        B.rule(st, lits, nxt, toks + tail)

def template_tokens(tmpl, kind, inner_neg):
    """a complements.tbl template -> the tokens that spell it and the kind the result conjugates as (the stem is emitted,
    its last mora left to the finite suffix for a v1 / ia / v5 result)"""
    import re
    parts = re.split(r"(\{\w+\})", tmpl); toks = []
    for p in parts:
        if not p: continue
        if p == "{dic}": toks.append(VF("dic"))
        elif p == "{stem}": toks.append(VF("i"))
        elif p == "{te}": toks.append(VF("te"))
        elif p == "{ta}": toks.append(VF("ta"))
        elif p == "{vol}": toks.append(VF("o"))
        elif p == "{inner}": toks += [VF("a"), C("ない")] if inner_neg else [VF("dic")]
        else: toks.append(C(p))
    if kind in ("v1", "ia"):
        last = toks[-1]; assert last[0] == "c" and len(last[1]) >= 1
        toks[-1] = C(last[1][:-1]); toks = [t for t in toks if t != ("c", "")]
        return toks, "V1C" if kind == "v1" else "IA"
    if kind == "v5":
        last = toks[-1]; kana = last[1][-1]; toks[-1] = C(last[1][:-1]); toks = [t for t in toks if t != ("c", "")]
        return toks, ("V5C", kana)
    return toks, {"na": "NA", "fixed": "FIXED"}[kind]

# ------------------------------------------------------------------------------------------------ the interface
class PrintRules:
    """the printing rules: positions in template order, the rules of each, the derived predicates, and a stepper"""
    def __init__(self):
        self._rules = build_rules(); self._D = derived()
        self._by_pos = {}
        for r in self._rules: self._by_pos.setdefault(r[0], []).append(r)
        order = []
        for r in self._rules:
            if r[0] not in order: order.append(r[0])
        self._positions = order
    def positions(self): return list(self._positions)
    def rules_for(self, position): return list(self._by_pos.get(position, []))
    def rules(self): return list(self._rules)
    def derived(self): return dict(self._D)

    def token_string(self, tok, F):
        kind = tok[0]
        if kind == "c": return tok[1]
        if kind == "s":
            v = F.get(tok[1])
            if len(tok) > 2: v = v[tok[2]]
            if v is None: raise Refuse(f"nothing to print in {tok[1]}")
            return "" if v == NOPART else str(v)
        if kind == "vf":
            if not filled(F.get("verb")): raise Refuse("a verb form without a verb")
            return vform(F["verb"], tok[1], tok[2])
        if kind == "af": return adj_stem(F["cop_adj"], tok[2])
        raise Refuse(f"unknown token {tok}")

    def render(self, frame, bound=200):
        """the tokens the rules emit for a frame (a Clause or a flat frame), by stepping them; raises Refuse"""
        F = flatten(frame) if not isinstance(frame, dict) else frame
        state = "P0"; out = []
        for _ in range(bound):
            if state == "DONE": return out
            if isinstance(state, str) and state.startswith("FAIL:"): raise Refuse(state[5:])
            hits = [r for r in self._by_pos.get(state, []) if all(eval_lit(l, F, self._D) for l in r[1])]
            if len(hits) != 1: raise Refuse(f"{len(hits)} rules at {state}")
            nxt, tok = hits[0][2]
            if tok is not None: out.append(self.token_string(tok, F))
            state = nxt
        raise Refuse("loop")
