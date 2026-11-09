r"""
The rules I can state, as a transducer: English -> frame -> Japanese.

Everything is finite and table-driven so it compiles to the same two-layer ReLU cell
as before (compile_rules.py): the parser is a DFA over (phase, token class, a few
register predicates); the frame is a fixed set of registers; generation is a fixed
template whose every slot is a table lookup over register contents and feature bits.

What is NOT here is the point of the experiment: whatever the held-out Tanaka
sentences need that I could not write down as a rule.
"""
import re
import tables          # the rule tables live in tables/*.tbl, one file per table
from collections import Counter

# ============================================================================
# 1. Japanese morphology I know
# ============================================================================
GODAN_ROWS = tables.build_godan()          # tables/godan.tbl

class Pred:
    """A conjugable predicate: kind in v1 (ichidan), v5 (godan), suru, kuru, ia (i-adjective),
    na (na-adjective / noun + copula), fixed (already finite)."""
    def __init__(self, kind, stem, dic=None):
        self.kind, self.stem = kind, stem
        self.dic = dic or (stem + "る" if kind == "v1" else stem)

    # --- stems ---
    def nai_stem(self):
        k = self.kind
        if k == "v1": return self.stem
        if k == "v5": return self.stem[:-1] + GODAN_ROWS[self.stem[-1]][0]
        if k == "suru": return self.stem[:-2] + "し"
        if k == "kuru": return self.kuru_stem("こ")
        raise ValueError
    def masu_stem(self):
        k = self.kind
        if k == "v1": return self.stem
        if k == "v5": return self.stem[:-1] + GODAN_ROWS[self.stem[-1]][1]
        if k == "suru": return self.stem[:-2] + "し"
        if k == "kuru": return self.kuru_stem("き")
        raise ValueError
    def kuru_stem(self, kana):
        """来る keeps 来 in every stem; a compound written in kana (持ってくる) inflects こ / き — pinned by test_kuru_compounds"""
        return self.stem[:-1] if self.stem.endswith("来る") else self.stem[:-2] + kana
    def te(self):
        k = self.kind
        if k == "forms": return self.stem[0][0][:-1] + "って"
        if k == "v1": return self.stem + "て"
        if k == "v5":
            if self.stem.endswith("行く"): return self.stem[:-1] + "って"
            return self.stem[:-1] + GODAN_ROWS[self.stem[-1]][4]
        if k == "suru": return self.stem[:-2] + "して"
        if k == "kuru": return self.kuru_stem("き") + "て"
        if k == "ia": return self.stem + "くて"
        if k == "na": return self.stem + "で"
        raise ValueError
    def ta(self):
        t = self.te()
        return t[:-1] + ("だ" if t.endswith("で") else "た") if self.kind in ("v1", "v5", "suru", "kuru") else None

    # --- transforms that yield a new predicate ---
    def potential(self):
        k = self.kind
        if k == "v1": return Pred("v1", self.stem + "られ")
        if k == "v5": return Pred("v1", self.stem[:-1] + GODAN_ROWS[self.stem[-1]][2])
        if k == "suru": return Pred("v1", self.stem[:-2] + "でき")
        if k == "kuru": return Pred("v1", self.kuru_stem("こ") + "られ")
        raise ValueError
    def tai(self):          return Pred("ia", self.masu_stem() + "た")
    def causative(self):
        caus = {"v1": self.stem + "させ", "v5": self.nai_stem() + "せ", "suru": self.stem[:-2] + "させ", "kuru": self.kuru_stem("こ") + "させ" if self.kind == "kuru" else None}.get(self.kind)
        return Pred("v1", caus, dic=caus + "る") if caus else self
    def progressive(self):  return Pred("v1", self.te() + "い")
    def should(self):       return Pred("na", self.dic + "べき")
    def maybe(self):        return Pred("fixed", self.dic + "かもしれない")
    def intend(self):       return Pred("na", self.dic + "つもり")
    def volitional(self, polite):
        k = self.kind
        if polite: return self.masu_stem() + "ましょう"
        if k == "v1": return self.stem + "よう"
        if k == "v5": return self.stem[:-1] + GODAN_ROWS[self.stem[-1]][3] + "う"
        if k == "suru": return self.stem[:-2] + "しよう"
        if k == "kuru": return self.kuru_stem("こ") + "よう"
        raise ValueError

    # --- finite forms ---
    def finite(self, past=False, neg=False, polite=False):
        k = self.kind
        if k in ("v1", "v5", "suru", "kuru"):
            if polite:
                s = self.masu_stem()
                return s + ["ます", "ました", "ません", "ませんでした"][2 * neg + past]
            if not neg:
                return self.ta() if past else self.dic
            n = self.nai_stem() + ("なかった" if past else "ない")
            return n
        if k == "ia":
            base = [self.stem + "い", self.stem + "かった", self.stem + "くない", self.stem + "くなかった"][2 * neg + past]
            return base + ("です" if polite else "")
        if k == "na":
            if polite:
                return self.stem + ["です", "でした", "ではありません", "ではありませんでした"][2 * neg + past]
            return self.stem + ["だ", "だった", "ではない", "ではなかった"][2 * neg + past]
        if k == "forms":
            plain, pol = self.stem
            return (pol if polite else plain)[2 * neg + past]
        if k == "fixed":
            return self.stem + ("です" if polite else "")
        raise ValueError(k)

def make_pred(ja, cls):
    if cls == "aru":
        pre = ja[:-2] if ja.endswith("ある") else ""
        return Pred("forms", ([pre + f for f in ("ある", "あった", "ない", "なかった")], [pre + f for f in ("あります", "ありました", "ありません", "ありませんでした")]), dic=pre + "ある")
    if cls == "g":  return Pred("v5", ja)
    if cls == "i":  return Pred("v1", ja[:-1], dic=ja)
    if cls == "s":  return Pred("suru", ja)
    if cls == "k":  return Pred("kuru", ja)
    raise ValueError(cls)

# ============================================================================
# 2. Lexicon: the words I can state.  pos: v (verb), n (noun), a (i-adj), na (na-adj),
#    adv, t (time adverb, goes first), p (preposition), int (ignored interjection)
#    verbs: lemma|v|ja|conj|past|participle ; nouns: word|n|ja|class(h human,a animal,t thing,p place)
# ============================================================================
LEXICON_LINES = tables.build_lexicon_lines()   # tables/lexicon.tbl

# ============================================================================
# 2b. What one does with a noun: the noun's use-verbs, keyed by the English light verb
#     (Mel'cuk's lexical functions).  noun | light verbs (* = take/have/get/make) | ja | conj | particle
# ============================================================================
USE_LINES = tables.build_use_lines()           # tables/use.tbl
USE = {}          # noun -> {light verb: (ja, conj, particle)}
for line in USE_LINES:
    f = line.split("|")
    noun, verbs, ja, conj, part = f[:5]
    for v in verbs.split(","):
        USE.setdefault(noun, {})[v] = (ja, conj, part, len(f) > 5)
LIGHT = {v for d in USE.values() for v in d}

# ============================================================================
# 3. English morphology I know, used to look words up by lemma
# ============================================================================
LEX = {}          # word -> list of entries (pos, ja, extra...)
VERB_FORMS = {}   # surface -> (lemma, form) form in base/3sg/past/ing/pp
def add(word, entry):
    LEX.setdefault(word, []).append(entry)

def regular_forms(base):
    if base.endswith("e") and not base.endswith("ee"):
        past, ing = base + "d", base[:-1] + "ing"
    elif base.endswith("y") and base[-2] not in "aeiou":
        past, ing = base[:-1] + "ied", base + "ing"
    elif re.search(r"[^aeiou][aeiou][bdglmnprt]$", base) and base not in ("open", "visit", "listen", "happen", "enter", "answer", "remember", "order", "offer"):
        past, ing = base + base[-1] + "ed", base + base[-1] + "ing"
    else:
        past, ing = base + "ed", base + "ing"
    if base.endswith(("s", "sh", "ch", "x", "o")): s3 = base + "es"
    elif base.endswith("y") and base[-2] not in "aeiou": s3 = base[:-1] + "ies"
    else: s3 = base + "s"
    return s3, past, ing

for line in LEXICON_LINES:
    f = line.split("|")
    word, pos, ja = f[0], f[1], f[2]
    if pos == "v":
        conj = f[3]
        s3, past, ing = regular_forms(word)
        if len(f) > 4: past = f[4]
        pp = f[5] if len(f) > 5 else past
        add(word, ("v", ja, conj))
        for surface, form in ((word, "base"), (s3, "3sg"), (past, "past"), (ing, "ing"), (pp, "pp")):
            VERB_FORMS.setdefault(surface, []).append((word, form))
    elif pos == "n":
        add(word, ("n", ja, f[3]))
        if not word.endswith("s") and " " not in word:
            pl = word + ("es" if word.endswith(("s", "sh", "ch", "x")) else "s")
            if pl not in LEX: add(pl, ("n", ja, f[3]))
    else:
        add(word, (pos, ja))
# irregular English verb forms the regular rules get wrong
for surface, lemma, form in (("has", "have", "3sg"), ("does", "do", "3sg"), ("goes", "go", "3sg"),
                             ("am", "be", "base"), ("is", "be", "3sg"), ("are", "be", "base"), ("were", "be", "past"),
                             ("being", "be", "ing"), ("having", "have", "ing"), ("running", "run", "ing"),
                             ("swimming", "swim", "ing"), ("sitting", "sit", "ing"), ("getting", "get", "ing"),
                             ("cutting", "cut", "ing"), ("putting", "put", "ing"), ("hitting", "hit", "ing"),
                             ("beginning", "begin", "ing"), ("stopping", "stop", "ing"), ("planning", "plan", "ing"),
                             ("shopping", "shop", "ing"), ("dying", "die", "ing"), ("lying", "lie", "ing")):
    VERB_FORMS.setdefault(surface, []).append((lemma, form))

NOUN_WORDS = {f[0] for f, n in tables.rows("lexicon") if f[1] == "n"}      # the lexicon's own noun entries (not the derived plurals)
SG_PRON = {"he", "she", "it", "this", "that", "somebody", "someone", "everybody", "everyone", "nobody", "anybody", "anyone", "something", "everything", "nothing", "anything", "one"}
PL_PRON = {"i", "you", "we", "they", "these", "those", "both"}
def number_of(tok):
    """the grammatical number a subject's head token fixes for verb agreement: '3sg', 'non3sg', or None when the token
    does not decide it (a singular-shaped noun may be a plural like people, so only pronouns and derived plurals decide)"""
    if tok in SG_PRON: return "3sg"
    if tok in PL_PRON: return "non3sg"
    if tok and any(c == "noun" for c, v in classify(tok)) and tok.endswith("s"):
        if tok[:-1] in NOUN_WORDS or (tok.endswith("es") and tok[:-2] in NOUN_WORDS) or (tok.endswith("ies") and tok[:-3] + "y" in NOUN_WORDS): return "non3sg"
    return None

PRONOUNS = tables.build_pairs("pronouns", 3)   # tables/pronouns.tbl: surface -> (japanese, role)
DETS = tables.build_pairs("determiners")       # tables/determiners.tbl
PREPS = tables.build_pairs("prepositions", 3) # tables/prepositions.tbl: word -> (action particle, existence particle)
PHRASAL, PHRASAL_NP, PARTICLE_AFTER, VERB_IDIOM, IDIOM_TAIL = tables.build_idioms()   # tables/idioms.tbl
ADJ_PREP = tables.build_adjectives()        # tables/adjectives.tbl
WH = tables.build_pairs("wh", 3)                # tables/wh.tbl
MODALS = tables.build_pairs("modals")          # tables/modals.tbl
NUMBERS = tables.build_pairs("numbers")        # tables/numbers.tbl
MULTIWORD = sorted([w for w in list(LEX) + list(PREPS) + list(DETS) + list(WH) if " " in w], key=len, reverse=True)

SUBORD, COORD, HEARSAY, SOFTENER = tables.build_connectives()   # tables/connectives.tbl

def subord_form(inner, conj):
    """a parsed subordinate clause -> Japanese clause + connective, Japanese-first order"""
    if conj in ("if", "once"): inner.past = True
    if conj == "after": inner.past = True
    if conj == "unless": inner.neg = True
    if conj == "so that" and inner.modal in ("may", "can"): inner.modal = "can"
    if conj == "so that" and inner.modal == "will": inner.modal = None
    body = generate(inner, polite=False, embedded=True)[:-1]
    if conj in ("if", "once"): return body + "ら"
    if conj == "unless": return (body[:-2] + "なければ") if body.endswith("ない") else body + "なら"
    if conj == "while" and inner.verb and not inner.copula and inner.subj is None:
        ipred = predicate_form(inner, False); p = clause_pred(inner)
        if p is not None: return body[:-len(ipred)] + p.masu_stem() + "ながら"
    return body + SUBORD[conj]

def clause_pred(c):
    """the Pred object a clause's predicate is built from, or None for the special cases"""
    if c.copula is not None:
        kind, *rest = c.copula
        if kind == "adj": return adj_pred(rest[0][0], rest[0][1])
        if kind == "np": return Pred("na", rest[0])
        if kind in ("pp", "adv"): return Pred("v1", "い") if c.subj_cls in ("h", "a") else Pred("v5", "ある")
        return None
    if c.verb is None: return None
    lemma, ja, conj = c.verb
    if conj in ("fixed", "ia_fixed") or c.want or c.modal or c.perf or "+" in lemma: return None
    p = make_pred(ja, conj)
    if c.prog or lemma in STATIVE: p = p.progressive()
    return p

def clause_te(c):
    """clause as a te-form chain link: 私は質問があって"""
    body = generate(c, polite=False)[:-1]
    ipred = predicate_form(c, False); p = clause_pred(c)
    if p is None or c.neg or c.past and p.kind in ("ia", "na"):
        return body + "、"
    try:
        return body[:-len(ipred)] + p.te()
    except Exception:
        return body + "、"

DURATION = tables.build_pairs("duration")      # tables/duration.tbl

CONTRACTED = tables.build_pairs("contractions")   # tables/contractions.tbl
def classify(tok):
    """All readings of a token: list of (class, payload)."""
    if tok in CONTRACTED and tok not in ("can",): tok = CONTRACTED[tok]
    out = []
    if tok in (".", "?", "!"): return [("end", tok)]
    if tok in ("'t", "not"): return [("not", None)]
    if tok == "n't": return [("not", None)]
    if tok in ("'s",): return [("poss", None), ("aux_be3", None)]
    if tok == "'m": return [("aux_be", None)]
    if tok == "'re": return [("aux_be", None)]
    if tok == "'ve": return [("aux_have", None)]
    if tok in ("do", "does", "did"): out.append(("aux_do", "past" if tok == "did" else "pres"))
    if tok in ("am", "is", "are", "was", "were", "be", "been", "being"):
        out.append(("aux_be", "past" if tok in ("was", "were") else "pres"))
    if tok in ("have", "has", "had"): out.append(("aux_have", "past" if tok == "had" else "pres"))
    if tok in MODALS: out.append(("modal", (MODALS[tok], tok in ("could", "would", "might", "'d"))))
    if tok == "to": out.append(("to", None))
    if tok == "that": out.append(("that", None))
    if tok == "there": out.append(("there", None))
    if tok == "let": out.append(("let", None))
    if tok == "please": out.append(("please", None))
    if tok in ("yes", "no") : out.append(("yesno", "はい" if tok == "yes" else "いいえ"))
    if tok in ("and", "or", "but", "because", "if", "when", "so", "while", "as", "though", "although", "whether", "unless", "until", "once", "since", "after", "before"):
        out.append(("conj", tok))
    if tok in WH: out.append(("wh", WH[tok]))
    if tok in PRONOUNS: out.append(("pron", PRONOUNS[tok]))
    if tok in DETS: out.append(("det", DETS[tok]))
    if tok in NUMBERS: out.append(("num", NUMBERS[tok]))
    if re.fullmatch(r"\d+", tok): out.append(("num", tok.translate(str.maketrans("0123456789", "０１２３４５６７８９"))))
    if tok in PREPS: out.append(("prep", PREPS[tok]))
    for lemma, form in VERB_FORMS.get(tok, []):
        for e in LEX[lemma]:
            if e[0] == "v": out.append(("verb", (lemma, form, e[1], e[2])))
    for e in LEX.get(tok, []):
        if e[0] == "n": out.append(("noun", (e[1], e[2])))
        elif e[0] in ("a", "na", "fx"): out.append(("adj", (e[1], e[0])))
        elif e[0] == "adv": out.append(("adv", e[1]))
        elif e[0] == "loc": out.append(("loc", e[1]))
        elif e[0] == "t": out.append(("time", e[1]))
        elif e[0] == "int": out.append(("int", e[1]))
        elif e[0] == "conj": out.append(("conj", tok))
        elif e[0] == "det": out.append(("det", e[1]))
    if not any(c == "adj" for c, _ in out) and tok.endswith("er") and len(tok) > 4:
        for base in (tok[:-2], tok[:-1], tok[:-3] + "y", tok[:-3]):
            es = [e for e in LEX.get(base, []) if e[0] in ("a", "na")]
            if es:
                out.append(("adj", ("より" + es[0][1], es[0][0]))); out.append(("adv", "より" + (es[0][1][:-1] + "く" if es[0][0] == "a" and es[0][1].endswith("い") else es[0][1] + "に"))); break
    return out

# ============================================================================
# 4. The frame parser: a DFA over (phase, class) with a few register predicates
# ============================================================================





# ============================================================================
# 5. Generation: frame -> Japanese, a fixed template of table lookups
# ============================================================================
def predicate_form(c, polite):
    """the finite predicate of a clause from its registers"""
    if c.there:
        base = Pred("v1", "い") if c.subj_cls in ("h", "a") else make_pred("ある", "aru")
        return base.finite(c.past, c.neg, polite)
    if c.copula is not None:
        kind, *rest = c.copula
        if kind == "wh":
            return ""
        if kind == "idiom":
            ja, k = rest
            p = {"na": Pred("na", ja), "a": Pred("ia", ja[:-1]), "fx": adj_pred(ja, "fx"), "v": make_pred(ja, "i" if ja.endswith(("れる", "える", "ける")) else "g")}[k]
            return p.finite(c.past, c.neg, polite)
        if kind == "adj":
            ja, k = rest[0]
            if c.compare is not None and ja.startswith("より"): ja = ja[2:]
            p = adj_pred(ja, k)
        elif kind == "np":
            p = Pred("na", rest[0])
            if c.relative and not c.neg and not c.past: return rest[0] + "である"
        elif kind == "pp":
            prep, np = rest
            p = Pred("v1", "い") if c.subj_cls in ("h", "a") else make_pred("ある", "aru")
        elif kind == "adv":
            p = Pred("v1", "い") if c.subj_cls in ("h", "a") else make_pred("ある", "aru")
        if c.modal == "must": return p_must(p, c, polite, noun=(kind == "np"))
        return p.finite(c.past, c.neg, polite)
    if c.verb is None:
        return ""
    lemma, ja, conj = c.verb
    if conj == "fixed":
        p = Pred("fixed", ja)
    elif conj == "ia_fixed":
        p = Pred("ia", ja[:-1])
    else:
        p = make_pred(ja, conj)
    if c.want == "tai":          p = p.tai()
    elif c.want in ("promise", "try", "decide") and getattr(c, "neg_inner", False):
        p = Pred("fixed", p.nai_stem() + {"promise": "ないと約束する", "try": "ないようにする", "decide": "ないことにした"}[c.want])
    elif c.want == "wish":       p = p.tai()
    elif c.want in COMPLEMENTS:  p = complement_pred(c.want, p, c)
    elif c.want == "better":     p = Pred("ia", (p.nai_stem() + "ない" if c.neg else p.ta()) + "ほうがい"); c.neg = False
    elif c.want == "neednot":    return p.dic + "必要はない" if not polite else p.dic + "必要はありません"
    elif c.want == "mind_request": return p.te() + "もらえません"
    elif c.want == "request":      return p.te() + "くださいません"                          # would you please V ? -> Vてくださいませんか
    elif c.want == "be_to":      p = Pred("fixed", p.dic + "ことになっている")
    elif c.want == "canthave":   return p.ta() + "はずがない"
    if "+" in lemma:
        lemma = "__use__"
    if lemma in STATIVE and not c.want and not c.imp and not c.lets and c.modal not in ("can",) and not c.perf:
        c.prog = True
    if lemma == "like" and not c.want:  p = Pred("na", "好き")
    elif lemma == "want" and not c.want and c.obj: p = Pred("ia", "欲し")
    elif lemma == "need" and not c.want: p = Pred("na", "必要")
    elif lemma == "hate" and not c.want: p = Pred("na", "嫌い")
    elif lemma == "understand" and not c.want: p = Pred("v5", "分かる")
    if c.perf and c.neg: c.prog = True                         # haven't come yet -> まだ来ていない
    if c.prog and not c.want and p.kind in ("v1", "v5", "suru", "kuru"):     p = p.progressive()
    if c.modal == "can":          p = p.potential() if p.kind in ("v1", "v5", "suru", "kuru") else p
    elif c.modal == "must":       return p_must(p, c, polite)
    elif c.modal == "should":     p = p.should()
    elif c.modal == "may":        p = p.maybe()
    if c.lets:                    return p.volitional(polite=True) if p.kind in ("v1", "v5", "suru", "kuru") else p.finite(False, False, True)
    if c.imp:
        if c.causative and p.kind in ("v1", "v5", "suru", "kuru"):
            caus = {"v1": p.stem + "させ", "v5": p.nai_stem() + "せ", "suru": p.stem[:-2] + "させ", "kuru": p.kuru_stem("こ") + "させ" if p.kind == "kuru" else None}[p.kind]
            return caus + "てください"
        if c.neg: return p.nai_stem() + "ないでください"
        return p.te() + "ください"
    if c.please and c.verb and not c.imp:
        return p.te() + "ください"
    past = c.past and not (c.modal == "will")
    if c.modal == "will" and c.past and not c.question and not c.want:                       # would (conjecture): 言わないだろう
        return p.finite(False, c.neg, False) + ("でしょう" if polite else "だろう")
    return p.finite(past, c.neg, polite)

class _Forms(dict):
    """forms of the governed verb, computed only when a template asks for them"""
    def __init__(self, p, c): super().__init__(); self.p, self.c = p, c
    def __missing__(self, k):
        p = self.p
        v = {"dic": lambda: p.dic, "stem": p.masu_stem, "te": p.te, "ta": p.ta, "vol": lambda: p.volitional(False), "inner": lambda: _inner_dic(p, self.c)}[k]()
        self[k] = v; return v

def complement_pred(want, p, c):
    """a row of tables/complements.tbl applied to the governed verb's Pred"""
    tmpl, kind = COMPLEMENTS[want]
    s = tmpl.format_map(_Forms(p, c))
    if kind == "v1": return Pred("v1", s[:-1], dic=s)
    if kind == "ia": return Pred("ia", s[:-1])
    return Pred(kind, s)

def _inner_dic(p, c):
    """dictionary form of the complement verb, negated when the English had 'not to V'"""
    return (p.nai_stem() + "ない") if getattr(c, "neg_inner", False) else p.dic

def _copy_for_hoshii(c):
    """want someone to V: the main predicate becomes ～てほしい (an i-adjective)"""
    c.verb = ("want someone to", "ほしい", "ia_fixed")
    return c

def adj_pred(ja, k):
    if k == "fx":
        return Pred("v1", ja[:-1]) if ja.endswith("いる") else Pred("fixed", ja)
    if ja.endswith("い") and not ja.endswith(("きれい", "ない")) and k != "na" or (k == "a" and ja.endswith("い") and not ja.endswith("ない")):
        return Pred("ia", ja[:-1])
    if ja.endswith("な") and k in ("a", "na"):
        return Pred("na", ja[:-1])
    if ja.endswith("の") or ja.endswith("た") or ja.endswith("て"):
        return Pred("fixed", ja[:-1] + "だ" if ja.endswith("の") else ja)
    return Pred("na", ja)

VC = tables.build_verb_classes()            # tables/verb_classes.tbl: class name -> set of verbs
def adj_causative(ja, kind):
    """make NP ADJ -> the adjective as a causative verb: 悲しい -> 悲しくさせる, 幸せ -> 幸せにさせる (ichidan)"""
    if kind == "a": return ja[:-1] + "くさせる"
    return (ja[:-1] if ja.endswith("な") else ja) + "にさせる"
def lemma_of(verb):
    """the English lemma a verb value is classified by: the light verb before the use-verb feedback (put down+address -> put down)"""
    return verb[0].split("+")[0]
STATIVE = VC["stative"]
GA_VERBS = VC["ga"]
COMPLEMENTS = tables.build_complements()     # tables/complements.tbl: verb -> (template, class)
def p_must(p, c, polite, noun=False):
    """must: obligation on a verb (〜なければならない), inference on a predicate (〜はずだ: 優しいはずだ, ばかなはずだ, 医者のはずだ)"""
    if p.kind in ("v1", "v5", "suru", "kuru"):
        if c.neg: return p.te() + "はいけない" + ("です" if polite else "")
        return p.nai_stem() + "なければならない" + ("です" if polite else "")
    if p.kind == "na": return p.stem + ("の" if noun else "な") + "はずだ"
    return p.finite(False, False, False) + "はずだ"

def generate(c, polite=None, embedded=False):
    if polite is None:
        polite = bool(c.question)
    out = []
    if c.fixed_out is not None:
        return c.fixed_out
    if c.so_that is not None:
        inner = c.so_that; c.so_that = None
        return generate(c, polite=False)[:-1] + "ので" + generate(inner, polite)
    if c.howabout is not None:
        return c.howabout + "はどうですか。"
    if c.exclaim is not None:
        return (c.subj + "は" if c.subj else "") + "なんと" + c.exclaim + "なのだろう。"
    if c.adv and "決して" in c.adv: c.neg = True                                              # never: the adverb carries the negation (決して〜ない)
    if c.front: out.append(c.front)
    if c.pre_clause: out.append(c.pre_clause)
    if c.time: out.append(c.time)
    subj_particle = "が" if (embedded and not c.coord) or (c.wh and c.wh[1] == "s") or c.there else "は"
    if c.there:
        for p, np in c.pp: out.append(np + p[1])
        if c.wh and c.wh[1] == "count":
            out.append(c.subj + "は" + "いくつ"); out.append(predicate_form(c, polite)); out.append("か")
            return "".join(out) + "。"
        if c.subj.endswith(("何も", "誰も")): c.neg = True; out.append(c.subj)
        else: out.append(c.subj + "が")
        out.append(predicate_form(c, polite))
        if c.question: out.append("か")
    else:
        wh_cop = c.wh and c.copula is not None and c.copula[0] == "wh"
        if c.subj and c.subj.endswith(("誰も", "何も")): c.neg = True
        weather = c.subj == "それ" and (c.verb and c.verb[0] in ("rain", "snow") or c.infinitive and c.infinitive[0] == "extraposed")
        if (c.subj is not None or c.wh and c.wh[1] == "s") and not c.imp and not c.lets and not (wh_cop and c.wh[0] == "何時" and c.subj == "それ") and c.want not in ("mind_request", "request") and not weather:
            if c.wh and c.wh[1] == "s" and not wh_cop: out.append(c.wh[0] + "が")
            elif c.subj.endswith(("誰も", "何も")): out.append(c.subj)
            else: out.append(c.subj + ("は" if wh_cop and not embedded else subj_particle))
        if c.wh and c.wh[1] in ("t", "pp") and not wh_cop: out.append(c.wh[0] + ("に" if c.wh[0] == "何時" else ""))
        if c.wh and c.wh[1] == "loc" and not wh_cop: out.append("どこ" + ("に" if c.copula or c.verb and lemma_of(c.verb) in VC["ni_where"] else "で"))
        if c.wh and c.wh[1] == "adv" and not c.copula: out.append(c.wh[0] if c.wh[0] != "どう" else "どうやって")
        if c.wh and c.wh[1] == "count": out.append("いくつの" + c.wh[0] + "を")
        exist = c.copula is not None and c.copula[0] == "pp"
        for p, np in c.pp:
            out.append(np + p[1 if exist or (c.verb and lemma_of(c.verb) in VC["ni_location"]) else 0])
        if c.copula is not None and c.copula[0] == "pp":
            out.append(c.copula[2] + c.copula[1][1])
        if c.compare is not None:
            kind_c, np_c = c.compare
            out.append(np_c + ("より" if kind_c == "than" else "に劣らず" if kind_c == "noless" else ("ほど" if c.neg else "と同じくらい")))
        if c.manner is not None:
            inner = c.manner
            body = generate(inner, polite=False, embedded=True)[:-1]
            ipred = predicate_form(inner, False)
            lemma0, ja0, conj0 = inner.verb
            p = make_pred(ja0, conj0) if conj0 != "fixed" else Pred("fixed", ja0)
            out.append(body[:-len(ipred)] + p.masu_stem() + "ながら")
        if c.control is not None and c.causative in VC["causative"]:
            inner = c.control
            body = generate(inner, polite=False, embedded=True)[:-1]
            ipred = predicate_form(inner, False)
            lemma0, ja0, conj0 = inner.verb
            p = make_pred(ja0, conj0) if conj0 != "fixed" else Pred("fixed", ja0)
            out.append(inner.subj + ("に" if inner.obj else "を") + body[len(inner.subj) + 1:-len(ipred)] + p.causative().finite(c.past, c.neg, polite))
            c.verb = None; c.control = None
        if c.control is not None:
            inner = c.control
            want_to = c.verb[0] == "want"
            inner_pred = predicate_form(inner, False)
            lemma0, ja0, conj0 = inner.verb if inner.verb else (None, None, None)
            body = generate(inner, polite=False, embedded=True)[:-1]
            body = body[len(inner.subj) + 1:] if inner.subj else body       # drop the が-subject
            if want_to:
                p = make_pred(ja0, conj0) if conj0 != "fixed" else Pred("fixed", ja0)
                out.append(inner.subj + "に" + body[:-len(inner_pred)] + p.te() + "ほしい" if False else inner.subj + "に" + body[:-len(inner_pred)] + p.te())
                c = _copy_for_hoshii(c)
            else:
                out.append(inner.subj + "に" + body + "ように")
        if c.purpose_clause is not None:
            inner = c.purpose_clause
            body = generate(inner, polite=False, embedded=True)[:-1]
            inner_pred = predicate_form(inner, False)
            lemma0, ja0, conj0 = inner.verb
            p = make_pred(ja0, conj0) if conj0 != "fixed" else Pred("fixed", ja0)
            if inner.purpose == "tame":
                out.append(body + "ために")
            else:
                out.append(body[:-len(inner_pred)] + p.masu_stem() + "に")
        if c.iobj: out.append(c.iobj + "に")
        if c.obj and c.subj and c.subj in ("私", "彼", "彼女", "私たち", "彼ら", "あなた") and c.obj.startswith(c.subj + "の") and len(c.obj) > len(c.subj) + 1:
            c.obj = c.obj[len(c.subj) + 1:]
        if c.obj:
            neg_pol = c.obj.endswith(("何も", "誰も"))
            if neg_pol: c.neg = True
            out.append(c.obj + ("" if neg_pol else c.obj_particle if c.obj_particle is not None else ("が" if (c.verb and (c.verb[0] in GA_VERBS or c.modal == "can") and not c.want) else "を")))
        if c.wh and c.wh[1] == "o" and not c.copula:
            out.append(c.wh[0] + "を")
        if c.adv: out.append(c.adv)
        if c.embedded is not None:
            out.append(generate(c.embedded, polite=False, embedded=True)[:-1] + "と")
        if c.wh and c.copula is not None and c.copula[0] == "wh" and c.wh[1] in ("o", "s", "adv", "loc", "t", "count"):
            if embedded: out.append(c.wh[0] + ("だった" if c.past else "") + "か")
            else: out.append(Pred("na", c.wh[0]).finite(c.past, c.neg, polite) + "か")
        else:
            if c.copula is not None and c.copula[0] == "adv":
                out.append(c.copula[1])
            if c.infinitive is not None:
                mode, inner = c.infinitive
                body = generate(inner, polite=False, embedded=True)[:-1]
                lemma0, ja0, conj0 = inner.verb if inner.verb else (None, None, None)
                p0 = (make_pred(ja0, conj0) if conj0 not in ("fixed", "ia_fixed") else Pred("fixed", ja0)) if ja0 else None
                adjf = predicate_form(c, polite)
                ipred = predicate_form(inner, False)
                if mode == "extraposed":
                    if body.endswith("だ"): body = body[:-1] + "な"
                    out.append(body + "のは" + c.degree + adjf)
                elif mode == "of_to":
                    out.append(body + "とは" + c.degree + adjf)
                elif mode == "adj_to":
                    out.append(body[:-len(ipred)] + (p0.te() if p0 and p0.kind != "fixed" else ipred) + c.degree + adjf)
                elif mode == "enough":
                    out.append(body + "のに十分" + adjf)
                elif mode == "too":
                    kind0, *rest0 = c.copula
                    a = adj_pred(rest0[0][0], rest0[0][1]) if kind0 == "adj" else None
                    base = (a.stem + "すぎて") if a is not None else ""
                    out.append(base + (p0.potential().finite(False, True, polite) if p0 and p0.kind in ("v1", "v5", "suru", "kuru") else "できない"))
            else:
                out.append(c.degree + predicate_form(c, polite if not c.suffix else False) + c.suffix)
            if c.question and not c.tag: out.append("か")
            if c.tag: out.append("ね")
    return "".join(out) + "。"


from parser_rules import ParseFail, Overflow, Clause, Parser, parse_bounded, BUFFER, MAX_DEPTH, WIDTH, HOLD   # the parser layer (parser_rules.py)

def translate(tokens):
    toks = merge_multiword(tokens)          # lexical layer: longest match over the buffer
    frame = parse_bounded(toks)             # parser layer under the ambiguity budget
    return generate(frame)                  # semantic layer: slots -> string

def merge_multiword(tokens):
    tokens = [x for t in tokens for x in (("can", "'t") if t == "cannot" else (t,))]
    tokens = [("will" if t == "won" and i + 1 < len(tokens) and tokens[i + 1] == "'t" else t) for i, t in enumerate(tokens)]
    out, i = [], 0
    while i < len(tokens):
        for mw in MULTIWORD:
            n = mw.count(" ") + 1
            if " ".join(tokens[i:i + n]) == mw:
                out.append(mw); i += n; break
        else:
            out.append(tokens[i]); i += 1
    return out
