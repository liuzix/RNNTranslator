"""generate() translated by hand into printing rules in the normal form of np_module: per printing position, an ordered
decision over slot predicates whose terminals emit one token.

The rules read the flat frame — the same register file the compiled cell holds: one-hot blocks for the flags, the enum
registers (modal, want, cop_kind, wh, the particles, the verb) and the sub-slots of every noun-phrase role (det, poss, num,
adj1, adj2, noun, noun2, mono, cls) — and never the English.  A rule is (position, literals, (next_position, token)):
the token is None (a silent move), ("c", string) a constant morpheme, ("s", slot[, index]) the value of a slot, ("vf",
form, alt) a form of the frame's verb from the morphology table (dic, ta, te, a = negative stem, i = masu stem, e =
potential stem, caus = causative stem, pre = what precedes ある), or ("af", "stem", strip) the stem of the copula's
adjective (より stripped when a comparison follows), or ("h", move) a move of the attention.  Literals are the base predicates below ("slot.x", "slot.x=v",
"flag.f", and the families verb.*, wh.*, adj.*, subj.*, obj.*, adv.*: each a test on one slot's value), a predicate on the
last morpheme printed ("out.class=C"), or a derived predicate d.* defined as a DNF over base literals, as in np_module.

The rules read their own output: the last morpheme printed stays with the printer (in the compiled decoder, in its state),
and a constant or a form lookup that is a stem carries the conjugation class it has as one (a third / fourth field of the
token: V1C, IA, NA, FIXED, V5C:<kana>; an adjective's stem and the copula's noun have theirs by what they are — out_class).
The finishing stage of the predicate (FIN) takes the class from there, not from its position, so a predicate whose stem is
out ends at one position per mode, whatever its class.  The modes are the finite ones and two stem forms, te and the masu
stem (rules_ja.STEM_FORMS), which the frame chooses: a clause whose `clause` mark is te or stem — spliced into another
clause like a that-clause, by a `link` of the same kind — prints up to that form of its predicate and gives the attention
back without と, and the clause it is spliced into goes on.  Pinned against generate(form=…) by the `printer` action's sweep.

The rules print from a context of frames, one per clause, with the attention on one frame at a time; a token ("h", "down")
or ("h", "up") moves the attention to the next or the previous frame instead of emitting.  A frame says what it is to the
others in its marks: `link`, the kind of inner clause spliced into it (that: a that-clause, the next frame); `sublink`, where
its subordinate clause is (before: the frame before it, after: the frame after it); `clause`, the kind of inner clause it is
itself (embedded, or a subordinate clause that came before / after its main clause); and `conj`, a subordinate clause's
conjunction.  A that-clause (c.embedded) is two frames: at the splice, after the outer clause's arguments and adverbs, the
attention goes down, the inner frame prints from P0 (が on its subject, no politeness) and at its END emits と and goes back
up, and the outer frame continues at its tail.  A subordinate clause (when, if, because …) is a frame beside its main
clause's, in the order the two were read, and is printed first either way, after the main clause's front adverb: printing
starts on the first frame, steps over a subordinate clause that came first to its main clause, and from the main clause's
P_PRE the attention goes to the subordinate frame — up or down, as `sublink` says — which prints to its END, emits its
conjunction's ending (とき, ので, たら's ら …) and gives the attention back for the main clause to go on at its time adverb.
What its conjunction changes in a clause (rules_ja.subord_adjust: the past of if) is in the frame already; unless slices the
predicate's string in generate() and is refused here; a bare -ing clause under while ends at its masu stem (the stem mode)
and takes ながら.  The sentence's question mark is
read after a that-clause or a subordinate clause, when the main frame has left the encoder: it arrives as the register
sent.question, not in a frame.
What generate() does with infinitives, control, fixed sentences and the idiom copula is out of scope
(parser_matrix.UNSUPPORTED_REGS): the frame cannot hold it, so render() refuses there.  Inside the scope the rules must
print exactly what generate() prints — pinned by the `printer` action (harness/eval_printer.py) on every sentence the rule
parser parses and on a sweep over the predicates the rules test, embedded under a reporting verb as well.

    uv run python -c "import gen_module, rules_ja as R, parser_rules as PR; print(gen_module.PrintRules().render(PR.parse_bounded('i like the big dog .'.split())))"
"""
import copy
import rules_ja as R
from rules_ja import Pred, make_pred, adj_pred

NP_ROLES = ("subj", "obj", "iobj", "pp1", "pp2", "pp3", "cop", "cmp")
NP_SUBS = ("det", "poss", "num", "adj1", "adj2", "noun", "noun2", "cls", "mono", "compound")
FLAGS = ("past", "neg", "prog", "perf", "question", "imp", "please", "lets", "there", "tag", "coord", "sub", "neg_inner", "noless", "perf_had", "from_seen")
UNSUPPORTED = ("infinitive", "control", "embedded", "purpose_clause", "pre_clause", "manner", "so_that", "howabout", "exclaim", "fixed_out", "relative", "suffix")
# (a `folded` frame — one whose string came from an inner clause — prints like any other: the string is opaque to the rules; only the cell cannot hold it)
PRON6 = ("私", "彼", "彼女", "私たち", "彼ら", "あなた")            # generate(): a possessive coreferent with one of these subjects is dropped
STEM_FORMS = R.STEM_FORMS          # te, stem: the non-finite forms a clause can end in — the te form, the masu stem
LINK_KINDS = ("that",) + STEM_FORMS   # slot `link` of a frame: the kind of inner clause spliced into it — the next frame of the context: a that-clause, or a clause that ends at a stem form
SUBLINK_KINDS = ("before", "after")   # slot `sublink` of a main clause's frame: its subordinate clause is the frame before / after it
CLAUSE_KINDS = ("embedded", "before", "after") + STEM_FORMS   # slot `clause` of a frame: the kind of inner clause it is (the main clause has none) — a that-clause, a subordinate clause read before / after its main clause, a clause that ends at the te form / the masu stem of its predicate
NOPART = "∅"                                                      # an explicit empty object particle (generate's ""), a value a slot can hold
EMPTY = (None, "", False)

class Refuse(Exception):
    """the rules do not print this frame: no rule or more than one at a position, a form the morphology cannot give"""

# ------------------------------------------------------------------------------------------------ the flat frame
def filled(v): return v not in EMPTY

def flatten(c, skip=()):
    """a parser_rules.Clause -> the flat frame the rules read; raises Refuse for a register the frame cannot hold (those in
    `skip` are left out instead: render() handles `embedded` itself)"""
    for r in UNSUPPORTED:
        if r in skip: continue
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

# the conjugation classes a printed stem can have — what the ending after it is chosen by: an ichidan stem (V1C), an i-adjective
# stem (IA), a na-adjective or a noun (NA), a phrase that is already finite (FIXED), a godan stem by its last kana (V5C:<kana>)
STEM_CLASSES = ("V1C", "IA", "NA", "FIXED")
COP_NOUN = ("cop.noun", "cop.noun2", "cop.mono")                 # the last piece of the copula's noun phrase: a noun is conjugated as a na-adjective
def v5_kanas(): return sorted({t[-1] for (t, kd) in R.COMPLEMENTS.values() if kd == "v5"})
def stem_classes(): return list(STEM_CLASSES) + [f"V5C:{k}" for k in v5_kanas()]

def out_class(tok, v=None):
    """the conjugation class of a printed morpheme as a stem, or None when nothing conjugates after it: the class the rule that
    emits it declares (a constant, a form lookup), the kind of the copula's adjective (its stem), a noun (the last piece of the
    copula's noun phrase).  tok: the token, v: the slot value it was read from"""
    kind = tok[0]
    if kind == "c": return tok[2] if len(tok) > 2 else None
    if kind == "vf": return tok[3] if len(tok) > 3 else None
    if kind == "af": return {"ia": "IA", "na": "NA", "fixed": "FIXED", "v1": "V1C"}.get(adj_kind(v)) if filled(v) else None
    if kind == "s": return "NA" if tok[1] in COP_NOUN else None
    return None

def out_pred(name, last):
    """a predicate on the last morpheme printed — (token, value), None before the first: out.class=<C>, its class as a stem"""
    if not name.startswith("out.class="): raise KeyError(name)
    return last is not None and out_class(*last) == name[10:]

def eval_lit(lit, F, D, last=None):
    """a literal on the frame under the attention F and the last morpheme printed `last`"""
    neg = lit.startswith("¬"); name = lit[1:] if neg else lit
    if name.startswith("d."): v = any(all(eval_lit(l, F, D, last) for l in conj) for conj in D[name])
    elif name.startswith("out."): v = out_pred(name, last)
    else: v = base_pred(name, F)
    return (not v) if neg else v

def frames_of(c):
    """a Clause -> the context its sentence prints from: one flat frame per clause in the order the encoder flushes them — the
    main clause, marked with the link where its inner clause is spliced, then that clause, marked embedded — or te / stem, when
    it ends at that form of its predicate (Clause.inner_of) — and so on down; and
    a subordinate clause (the one clause Clause.subords holds, when pre_clause is exactly its string) with its conjunction's
    changes made (rules_ja.subord_adjust), marked before or after and put on that side of the main clause, whose frame gets
    the sublink.  Any other pre_clause — a coordinated clause, two subordinate clauses — is a register the frames cannot hold"""
    def chain(c, kind, skip):
        frames = []
        while c is not None:
            F = flatten(c, skip=skip); skip = ("embedded",)
            if kind: F["clause"] = kind
            if c.embedded is not None:                                # the clause spliced into this one: a that-clause, or one that ends at a stem form
                kind = c.embedded.inner_of if getattr(c.embedded, "inner_of", None) in STEM_FORMS else "embedded"
                F["link"] = "that" if kind == "embedded" else kind
            frames.append(F); c = c.embedded
        return frames
    if not filled(c.pre_clause): return chain(c, None, ("embedded",))
    subs = getattr(c, "subords", [])
    if len(subs) != 1 or R.subord_form(subs[0][2], subs[0][1]) != c.pre_clause: raise Refuse("register pre_clause")
    order, conj, inner = subs[0]
    inner = copy.deepcopy(inner); R.subord_adjust(inner, conj)
    main = chain(c, None, ("embedded", "pre_clause")); sub = chain(inner, order, ("embedded",))
    main[0]["sublink"] = order; sub[0]["conj"] = conj
    return sub + main if order == "before" else main + sub

def derived():
    """the derived predicates: name -> DNF (a list of conjunctions of base literals)"""
    D = {}
    D["d.neg"] = [["flag.neg"], ["adv.never"], ["adv2.never"], ["subj.negpol"], ["obj.negpol"]]   # generate() turns these into c.neg before it prints the predicate
    D["d.whcop"] = [["slot.wh", "slot.cop_kind=wh"]]
    D["d.weather"] = [["subj.sore", "verb.weather"]]
    # the sentence is a question by a flag of the main clause's frame (the inversion) or by the question mark read after a
    # that-clause or a subordinate clause (sent.question); an inner clause — a frame with a `clause` mark of any kind — is never
    # polite, and takes か only on a question flag of its own
    D["d.polite"] = [["flag.question", "¬slot.clause"], ["slot.sent.question", "¬slot.clause"]]
    D["d.q"] = [["flag.question"], ["slot.sent.question", "¬slot.clause"]]
    NEGS = ["flag.neg", "adv.never", "adv2.never", "subj.negpol", "obj.negpol"]; NOT_NEG = ["¬" + n for n in NEGS]
    # the tense generate() prints: c.past, except that a negative perfect takes the tense of its auxiliary (perf_had), and will takes none
    past_eff = [["flag.past", "¬flag.perf"], ["flag.past", "flag.perf"] + NOT_NEG] + [["flag.perf", "flag.perf_had", n] for n in NEGS]
    D["d.past_eff"] = [conj + ["¬slot.modal=will"] for conj in past_eff]
    D["d.past_raw"] = [["flag.past", "¬slot.modal=will"]]                                   # had better: the tense before the perfect correction
    D["d.will_conj"] = [conj + ["slot.modal=will", "¬flag.question", "¬slot.want", sent] for conj in past_eff
                        for sent in ("¬slot.sent.question", "slot.clause")]                         # would: 〜だろう, unless the clause is a question
    D["d.verbkind"] = [[f"verb.conj={c}"] for c in ("g", "i", "s", "k")]
    D["d.pp_state"] = [["slot.cop_kind=pp"], ["verb.vc=ni_location"]]                   # the PP takes its existence particle
    D["d.stative_prog"] = [["verb.stative", "¬verb.use", "¬slot.want", "¬flag.imp", "¬flag.lets", "¬slot.modal=can", "¬flag.perf"]]
    D["d.obj_default_ga"] = [["verb.ga", "¬slot.want"], ["slot.modal=can", "¬slot.want"]]
    D["d.obj_default_ni"] = [["verb.vc=ni_obj"]]
    D["d.while_ing"] = [["slot.conj=while", "¬slot.subj.noun", "slot.verb", "¬slot.cop_kind"]]       # while walking: a bare -ing clause under while
    D["d.while_nopred"] = [["slot.want"], ["slot.modal"], ["flag.perf"], ["verb.use"], ["verb.conj=fixed"], ["verb.conj=ia_fixed"]]   # … whose predicate rules_ja.clause_pred does not give: it stays finite, with 間
    # the form a clause's predicate ends in is decided by the frame: the te form or the masu stem for a clause of that kind, and the
    # masu stem (〜ながら) for a bare -ing clause under while whose predicate clause_pred gives; every other frame ends finite
    D["d.form.te"] = [["slot.clause=te"]]
    #   (rules_ja.subord_form takes the stem of the verb itself: 'understand', which the predicate prints as 分かる, is left out and refused)
    D["d.form.stem"] = [["slot.clause=stem"], ["¬slot.clause=te"] + D["d.while_ing"][0] + ["¬" + c[0] for c in D["d.while_nopred"]] + ["¬flag.question", "¬flag.tag", "¬verb.lemma=understand"]]
    D["d.stemform"] = D["d.form.te"] + D["d.form.stem"]
    for c in stem_classes(): D[f"d.cls.{c}"] = [[f"out.class={c}"]]                                # the class of the stem just printed, read from the output block
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
def C(s, cls=None): return ("c", s) if cls is None else ("c", s, cls)                  # cls: the class the morpheme has as a stem (out_class)
def S(slot, idx=None): return ("s", slot) if idx is None else ("s", slot, idx)
def VF(name, alt=False, cls=None): return ("vf", name, alt) if cls is None else ("vf", name, alt, cls)
def cls_lit(c): return f"d.cls.{c}"
def H(move): return ("h", move)                   # no morpheme: the attention moves down to the next frame or up to the previous one

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
    # ---- the clause: front, the subordinate clause, time, then the there-template or the main template; the there-template has
    #      no that-clause.  Printing starts on the first frame: a subordinate clause that was read first is stepped over to its main
    #      clause, the next frame (a that-clause inside it would lie between the two: refused)
    B.rule("P0", ["slot.clause=before", "slot.link"], "FAIL:a that-clause inside a subordinate clause that comes first")
    B.rule("P0", ["slot.clause=before", "¬slot.link"], "P0", [H("down")])
    B.rule("P0", ["¬slot.clause=before", "slot.front"], "P_PRE", [S("front")]); B.rule("P0", ["¬slot.clause=before", "¬slot.front"], "P_PRE")
    # ---- the subordinate clause prints here, whichever side of its main clause it was read on: the attention goes to its frame,
    #      which prints from its own front to its END, emits its conjunction's ending and gives the attention back (S_END)
    B.rule("P_PRE", ["slot.sublink=before"], "S_IN_B", [H("up")]); B.rule("P_PRE", ["slot.sublink=after"], "S_IN_A", [H("down")]); B.rule("P_PRE", ["¬slot.sublink"], "P_TIME")
    for st, kind in (("S_IN_B", "before"), ("S_IN_A", "after")):
        B.rule(st, [f"slot.clause={kind}", "slot.front"], "P_TIME", [S("front")]); B.rule(st, [f"slot.clause={kind}", "¬slot.front"], "P_TIME")
        B.rule(st, [f"¬slot.clause={kind}"], "FAIL:the frame on that side of the main clause is not its subordinate clause")
    B.rule("P_TIME", ["slot.time"], "P_BRANCH", [S("time")]); B.rule("P_TIME", ["¬slot.time"], "P_BRANCH")
    B.rule("P_BRANCH", ["flag.there", "slot.link"], "FAIL:there-clause with a that-clause")
    B.rule("P_BRANCH", ["flag.there", "¬slot.link"], "T_PP1"); B.rule("P_BRANCH", ["¬flag.there"], "M_SUBJ0")
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
    B.rule("M_SUBJ_WA", ["slot.clause", "¬flag.coord"], "M_WH_T", [C("が")])                     # the subject of an inner clause
    B.rule("M_SUBJ_WA", ["slot.clause", "flag.coord"], "M_WH_T", [C("は")]); B.rule("M_SUBJ_WA", ["¬slot.clause"], "M_WH_T", [C("は")])
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
    B.rule("M_ADV2", ["slot.adv2"], "M_LINK", [S("adv2")]); B.rule("M_ADV2", ["¬slot.adv2"], "M_LINK")
    # ---- the splice: the inner clause is the next frame — the attention goes down, the clause prints from P0 and comes back up to
    #      M_BACK: a that-clause from its END, after と; a clause of a stem-form kind from STEM_END, where its predicate stopped
    B.rule("M_LINK", ["slot.link"], "P0", [H("down")]); B.rule("M_LINK", ["¬slot.link"], "M_TAIL")
    B.rule("M_BACK", ["slot.link"], "M_TAIL"); B.rule("M_BACK", ["¬slot.link"], "FAIL:no clause was spliced into this frame")
    # ---- the tail: a question-word copula (何ですか; inside a that-clause 何か / 何だったか), or the copula adverb, the degree and the predicate
    whk = ["o", "s", "adv", "loc", "t", "count"]
    for k in whk:
        B.rule("M_TAIL", ["d.whcop", f"wh.kind={k}", "¬slot.clause"], ("FIN", "NA", "whcop"), [S("wh", 0)])
        B.rule("M_TAIL", ["d.whcop", f"wh.kind={k}", "slot.clause"], ("FIN", "NA", "whcop_emb"), [S("wh", 0)])
    B.rule(("FIN", "NA", "whcop_emb"), ["flag.past"], "END", [C("だった"), C("か")]); B.rule(("FIN", "NA", "whcop_emb"), ["¬flag.past"], "END", [C("か")])
    B.rule("M_TAIL", ["d.whcop"] + [f"¬wh.kind={k}" for k in whk], "M_COPADV"); B.rule("M_TAIL", ["¬d.whcop"], "M_COPADV")
    B.rule("M_COPADV", ["slot.cop_kind=adv"], "M_DEGREE", [S("cop_adv")]); B.rule("M_COPADV", ["¬slot.cop_kind=adv"], "M_DEGREE")
    B.rule("M_DEGREE", ["slot.degree"], "M_DEGREE2", [S("degree")]); B.rule("M_DEGREE", ["¬slot.degree"], "M_DEGREE2")
    B.rule("M_DEGREE2", ["slot.degree2"], "PRED", [S("degree2")]); B.rule("M_DEGREE2", ["¬slot.degree2"], "PRED")
    # ---- after the predicate: か / ね, then 。
    fin = ["¬d.stemform"]                                                          # (a predicate that reaches Q is finite)
    B.rule("Q", ["d.stemform"], "FAIL:the predicate of a clause that ends at a stem form has none here")
    B.rule("Q", fin + ["flag.there", "wh.kind=count"], "END", [C("か")])
    B.rule("Q", fin + ["flag.there", "¬wh.kind=count", "d.q"], "END", [C("か")]); B.rule("Q", fin + ["flag.there", "¬wh.kind=count", "¬d.q"], "END")
    B.rule("Q", fin + ["¬flag.there", "flag.tag"], "END", [C("ね")]); B.rule("Q", fin + ["¬flag.there", "¬flag.tag", "d.q"], "END", [C("か")])
    B.rule("Q", fin + ["¬flag.there", "¬flag.tag", "¬d.q"], "END")
    B.rule("END", ["slot.clause=embedded"], "M_BACK", [C("と"), H("up")])       # the that-clause closes: と, and back up to the clause it was spliced into
    B.rule("END", ["slot.clause=before"], "S_END_B"); B.rule("END", ["slot.clause=after"], "S_END_A")
    for kind in STEM_FORMS: B.rule("END", [f"slot.clause={kind}"], "FAIL:a clause that ends at a stem form has no finite end")
    B.rule("END", ["¬slot.clause"], "DONE", [C("。")])
    # ---- a predicate that stopped at its te form or its masu stem (FIN, modes te and stem): a clause of that kind gives the
    #      attention back to the clause it is spliced into, which goes on; a bare -ing clause under while takes ながら and gives the
    #      attention back to its main clause as any subordinate clause does
    for kind in STEM_FORMS: B.rule("STEM_END", [f"slot.clause={kind}"], "M_BACK", [H("up")])
    B.rule("STEM_END", ["slot.clause=before"], "P_TIME", [C("ながら"), H("down")]); B.rule("STEM_END", ["slot.clause=after"], "P_TIME", [C("ながら"), H("up")])
    B.rule("STEM_END", [f"¬slot.clause={kind}" for kind in STEM_FORMS + ("before", "after")], "FAIL:a stem form in a clause of no such kind")
    # ---- a subordinate clause closes: its conjunction's ending, and the attention back to the main clause — down when the clause
    #      was read before it, up when after — which goes on at its time adverb.  The ending follows the clause's finite form
    #      (if and once: ら after the past the frame holds); unless slices that form: refused.  A bare -ing clause under while ends
    #      at its masu stem and never comes here (STEM_END) — unless its predicate has none, when it is finite with 間 like any other
    for st, move in (("S_END_B", "down"), ("S_END_A", "up")):
        for conj in sorted(R.SUBORD):
            if conj == "unless": B.rule(st, ["slot.conj=unless"], "FAIL:unless takes the negative stem of its clause's predicate"); continue
            ending = [C("ら" if conj in ("if", "once") else R.SUBORD[conj]), H(move)]
            if conj == "while":
                B.rule(st, ["slot.conj=while", "d.while_ing", "¬d.while_nopred"], "FAIL:a bare -ing clause under while that did not end at its stem")
                B.rule(st, ["slot.conj=while", "d.while_ing", "d.while_nopred"], "P_TIME", ending); B.rule(st, ["slot.conj=while", "¬d.while_ing"], "P_TIME", ending)
            else: B.rule(st, [f"slot.conj={conj}"], "P_TIME", ending)
        B.rule(st, [f"¬slot.conj={conj}" for conj in sorted(R.SUBORD)], "FAIL:a subordinate clause without a conjunction the table has")
    build_predicate(B, np)
    return reachable(B.rules)

def is_terminal(pos): return pos == "DONE" or (isinstance(pos, str) and pos.startswith("FAIL:"))        # where a run of the rules ends

def reachable(rules):
    """the rules of the positions reachable from P0, in their order: a kind-and-mode combination no path needs is dropped"""
    by_pos = {}
    for r in rules: by_pos.setdefault(r[0], []).append(r)
    seen = {"P0"}; frontier = ["P0"]
    while frontier:
        for pos, lits, (nxt, tok) in by_pos.get(frontier.pop(), []):
            if nxt not in seen and not is_terminal(nxt): seen.add(nxt); frontier.append(nxt)
    return [r for r in rules if r[0] in seen]

def build_predicate(B, np):
    """predicate_form(): the kind of predicate, the transformations want / prog / modal / lets / imp / please apply to it, and
    its ending — the stem is emitted when the kind is fixed and the suffixes follow, one morpheme per step.  A stem is emitted
    with its conjugation class (out_class), and the finishing stage FIN reads the class from the last morpheme printed instead
    of taking it from its position: one position per mode for every predicate whose stem is out (track S), and one each for a
    plain verb (V, VA: its form is one lookup, chosen with the ending) and for ある (ARU), which have no stem out.  The mode is
    the finite form the flags choose, or — for a frame whose clause kind says so (d.form.te, d.form.stem) — the te form or the
    masu stem"""
    verbkind = ["d.verbkind"]
    def to_fin(st, lits, track, mode, toks=()):
        """into the finishing stage of a track (V, VA, ARU, or S: the stem is out): the finite mode, or the form the frame's kind asks for"""
        B.rule(st, lits + ["¬d.form.te", "¬d.form.stem"], ("FIN", track, mode), toks)
        B.rule(st, lits + ["d.form.te"], ("FIN", track, "te"), toks); B.rule(st, lits + ["d.form.stem"], ("FIN", track, "stem"), toks)
    # PRED: which predicate
    to_fin("PRED", ["flag.there", "subj.animate"], "S", "plain", [C("い", "V1C")])            # there is: no modal reaches the predicate
    to_fin("PRED", ["flag.there", "¬subj.animate"], "ARU", "plain")
    B.rule("PRED", ["¬flag.there", "slot.cop_kind=wh"], "Q")
    B.rule("PRED", ["¬flag.there", "slot.cop_kind=adj", "slot.cmp.noun"], "PRED_ADJ", [("af", "stem", True)])
    B.rule("PRED", ["¬flag.there", "slot.cop_kind=adj", "¬slot.cmp.noun"], "PRED_ADJ", [("af", "stem", False)])
    for k in ("ia", "na", "fixed", "v1"): B.rule("PRED_ADJ", [f"adj.kind={k}"], ("K", {"ia": "IA", "na": "NA", "fixed": "FIXED", "v1": "V1C"}[k]))
    B.rule("PRED_ADJ", [f"¬adj.kind={k}" for k in ("ia", "na", "fixed", "v1")], "FAIL:adjective kind")
    B.rule("PRED", ["¬flag.there", "slot.cop_kind=np"], np("cop", ("K", "NA", "noun")))
    for kind in ("pp", "adv"):
        B.rule("PRED", ["¬flag.there", f"slot.cop_kind={kind}", "subj.animate"], ("K", "V1C"), [C("い", "V1C")])
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
            to_fin(st, ["¬slot.modal=must"], "ARU" if kind == "ARU" else "S", "plain")
    # ---- a verb predicate: the complement transformation
    v = "V_WANT"
    B.rule(v, ["slot.want=tai"], ("V2", "IA"), [VF("i"), C("た", "IA")]); B.rule(v, ["slot.want=wish"], ("V2", "IA"), [VF("i"), C("た", "IA")])
    for w, tail in (("promise", "と約束する"), ("try", "ようにする"), ("decide", "ことにした")):
        B.rule(v, [f"slot.want={w}", "flag.neg_inner"], ("V2", "FIXED"), [VF("a"), C("ない"), C(tail, "FIXED")])
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
    B.rule(v, ["slot.want=better", "d.neg"], ("V2", "IA_NN"), [VF("a"), C("ない"), C("ほうがい", "IA")]); B.rule(v, ["slot.want=better", "¬d.neg"], ("V2", "IA_NN"), [VF("ta"), C("ほうがい", "IA")])
    B.rule(v, ["slot.want=neednot", "d.polite"], "Q", [VF("dic"), C("必要はありません")]); B.rule(v, ["slot.want=neednot", "¬d.polite"], "Q", [VF("dic"), C("必要はない")])
    B.rule(v, ["slot.want=mind_request"], "Q", [VF("te"), C("もらえません")]); B.rule(v, ["slot.want=request"], "Q", [VF("te"), C("くださいません")])
    B.rule(v, ["slot.want=canthave"], "Q", [VF("ta"), C("はずがない")])
    B.rule(v, others + ["¬verb.conj=fixed"], ("V2", "V"))       # no want, or one that leaves the governed verb as it is
    B.rule(v, others + ["verb.conj=fixed"], ("V2", "FIXED"), [VF("dic", False, "FIXED")])
    # ---- the special verbs (like -> 好き ...), stative -> progressive, perf+neg -> progressive, the progressive
    B.rule(("V2", "V"), ["¬slot.want", "¬verb.use", "verb.lemma=like"], ("V3", "NA"), [C("好き", "NA")])
    B.rule(("V2", "V"), ["¬slot.want", "¬verb.use", "verb.lemma=want", "slot.obj.noun"], ("V3", "IA"), [C("欲し", "IA")])
    B.rule(("V2", "V"), ["¬slot.want", "¬verb.use", "verb.lemma=need"], ("V3", "NA"), [C("必要", "NA")])
    B.rule(("V2", "V"), ["¬slot.want", "¬verb.use", "verb.lemma=hate"], ("V3", "NA"), [C("嫌い", "NA")])
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
        B.rule(("V3", src), prog + ["d.stative_prog"], ("V4", "V1C"), [VF("te", alt), C("い", "V1C")])
        B.rule(("V3", src), prog + ["¬d.stative_prog", "flag.perf", "d.neg"], ("V4", "V1C"), [VF("te", alt), C("い", "V1C")])
        B.rule(("V3", src), prog + ["¬d.stative_prog", "¬flag.perf", "flag.prog"], ("V4", "V1C"), [VF("te", alt), C("い", "V1C")])
        B.rule(("V3", src), prog + ["¬d.stative_prog", "flag.perf", "¬d.neg", "flag.prog"], ("V4", "V1C"), [VF("te", alt), C("い", "V1C")])
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
        if kind in ("V", "VA"): B.rule(st, ["slot.modal=can"], ("V5", "V1C"), [VF("e", alt, "V1C")])
        elif kind == "V1C": B.rule(st, ["slot.modal=can"], ("V5", "V1C"), [C("られ", "V1C")])
        elif kind == "V5C": B.rule(st, ["slot.modal=can"], ("V5", "V1C"), [C(R.GODAN_ROWS[kd[1]][2], "V1C")])
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
        B.rule(st, ["slot.modal=should"], ("V5", "NA"), dic + [C("べき", "NA")])
        B.rule(st, ["slot.modal=may"], ("V5", "FIXED"), dic + [C("かもしれない", "FIXED")])
        B.rule(st, ["¬slot.modal=can", "¬slot.modal=must", "¬slot.modal=should", "¬slot.modal=may"], nxt_base)
    B.rule("MUST_TAIL", ["d.polite"], "Q", [C("です")]); B.rule("MUST_TAIL", ["¬d.polite"], "Q")
    for kd in kinds_after_v4():
        st = ("V5",) + kd; kind = kd[0]; alt = kind == "VA"; verbal = kind in ("V", "VA", "V1C", "V5C")
        # lets
        if kind in ("V", "VA"): B.rule(st, ["flag.lets"], "Q", [VF("i", alt), C("ましょう")])
        elif kind == "V1C": B.rule(st, ["flag.lets"], "Q", [C("ましょう")])
        elif kind == "V5C": B.rule(st, ["flag.lets"], "Q", [C(R.GODAN_ROWS[kd[1]][1]), C("ましょう")])
        else: B.rule(st, ["flag.lets"], ("FIN", "S", "lets"))
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
        # would: finite(False, neg, False) + だろう; else the ending the frame asks for
        track = kind if kind in ("V", "VA") else "S"
        B.rule(st, ["¬flag.lets", "¬flag.imp", "¬flag.please", "d.will_conj"], ("FIN", track, "conj" if kind != "IA_NN" else "conj_nn"))
        to_fin(st, ["¬flag.lets", "¬flag.imp", "¬flag.please", "¬d.will_conj"], track, "flags" if kind != "IA_NN" else "noneg")
    # ---- FIN: the ending of the predicate.  Finite modes — the suffix of a class under (past, neg, polite): flags (the frame's),
    #      plain, noneg (want=better), conj (plain present + だろう), lets (polite, present, affirmative); whcop (+ か) for the
    #      question-word copula; and the stem-form modes te and stem, which stop the predicate there (STEM_END)
    def finite_rules(st, kind, mode, guard):
        alt = kind == "VA"; nxt = "END" if mode == "whcop" else "Q"
        tail = [C("か")] if mode == "whcop" else [C("だろう")] if mode.startswith("conj") else []
        for past in (False, True):
            for neg in (False, True):
                for polite in (False, True):
                    lits = list(guard)
                    if mode in ("conj", "conj_nn"):
                        if past or polite: continue
                        if mode == "conj_nn" and neg: B.rule(st, lits + ["d.neg"], "FAIL:a negated ほうがいい under would"); continue
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
                    elif kind.startswith("V5C:"):
                        plain, pol = godan_suffixes(kind[4:]); toks = [C((pol if polite else plain)[idx])]
                    else:
                        toks = [C((POLITE if polite else PLAIN)[kind][idx])]
                    toks = [t for t in toks if t != ("c", "")]
                    B.rule(st, lits, nxt, toks + tail)
    modes = ("flags", "plain", "noneg", "conj", "conj_nn", "lets")
    no_stem = ["¬" + cls_lit(c) for c in stem_classes()]
    for kind in ("V", "VA", "ARU"):
        for mode in modes: finite_rules(("FIN", kind, mode), kind, mode, [])
    finite_rules(("FIN", "NA", "whcop"), "NA", "whcop", [])
    for mode in modes:
        for c in stem_classes(): finite_rules(("FIN", "S", mode), c, mode, [cls_lit(c)])
        B.rule(("FIN", "S", mode), no_stem, "FAIL:no stem was printed for this ending")
    # the te form and the masu stem, as rules_ja.Pred gives them: a plain verb by lookup; after a stem, the ending of its class —
    # an ichidan stem is its own masu stem, an adjective, a noun and a fixed phrase have none
    for kind, alt in (("V", False), ("VA", True)):
        B.rule(("FIN", kind, "te"), [], "STEM_END", [VF("te", alt)]); B.rule(("FIN", kind, "stem"), [], "STEM_END", [VF("i", alt)])
    B.rule(("FIN", "ARU", "te"), [], "STEM_END", [C("あって")]); B.rule(("FIN", "ARU", "stem"), [], "FAIL:ある has no masu stem")
    endings = {"te": {"V1C": [C("て")], "IA": [C("くて")], "NA": [C("で")]}, "stem": {"V1C": []}}
    for k in v5_kanas(): endings["te"][f"V5C:{k}"] = [C(R.GODAN_ROWS[k][4])]; endings["stem"][f"V5C:{k}"] = [C(R.GODAN_ROWS[k][1])]
    for mode in STEM_FORMS:
        for c in stem_classes():
            if c in endings[mode]: B.rule(("FIN", "S", mode), [cls_lit(c)], "STEM_END", endings[mode][c])
            else: B.rule(("FIN", "S", mode), [cls_lit(c)], f"FAIL:no {mode} form of a predicate of this class")
        B.rule(("FIN", "S", mode), no_stem, "FAIL:no stem was printed for this ending")

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
    def classed(tok, c): return C(tok[1], c) if tok[0] == "c" else VF(tok[1], tok[2], c)       # the stem: the token after which the result conjugates
    if kind in ("v1", "ia"):
        last = toks[-1]; assert last[0] == "c" and len(last[1]) >= 2
        toks[-1] = C(last[1][:-1], "V1C" if kind == "v1" else "IA")
        return toks, "V1C" if kind == "v1" else "IA"
    if kind == "v5":
        last = toks[-1]; kana = last[1][-1]; assert last[0] == "c" and len(last[1]) >= 2
        toks[-1] = C(last[1][:-1], f"V5C:{kana}")
        return toks, ("V5C", kana)
    k2 = {"na": "NA", "fixed": "FIXED"}[kind]; toks[-1] = classed(toks[-1], k2)
    return toks, k2

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

    def render(self, frame, bound=400):
        """the tokens the rules emit for a Clause (its frames_of), a flat frame or a list of frames; raises Refuse"""
        return self.run(frame if isinstance(frame, list) else [frame] if isinstance(frame, dict) else frames_of(frame), bound)

    def run(self, frames, bound=400):
        """step the rules from P0 with the attention on the first frame until DONE: the rules read the frame under the attention and
        the last morpheme printed, a head token moves the attention; no rule, two rules, a token the frame cannot give, or the
        attention leaving the context are refusals"""
        state = "P0"; at = 0; out = []; last = None
        for _ in range(bound):
            if state == "DONE": return out
            if isinstance(state, str) and state.startswith("FAIL:"): raise Refuse(state[5:])
            if not 0 <= at < len(frames): raise Refuse("the attention left the context")
            F = frames[at]
            hits = [r for r in self._by_pos.get(state, []) if all(eval_lit(l, F, self._D, last) for l in r[1])]
            if len(hits) != 1: raise Refuse(f"{len(hits)} rules at {state}")
            nxt, tok = hits[0][2]
            if tok is None: pass
            elif tok[0] == "h": at += 1 if tok[1] == "down" else -1
            else: out.append(self.token_string(tok, F)); last = (tok, self.token_value(tok, F))
            state = nxt
        raise Refuse("loop")

    def token_value(self, tok, F):
        """the slot value a token was read from (None for a constant): with the token, the last morpheme as the rules read it"""
        return F.get(tok[1]) if tok[0] == "s" else F.get("verb") if tok[0] == "vf" else F.get("cop_adj") if tok[0] == "af" else None
