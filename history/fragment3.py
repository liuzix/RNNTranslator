r"""
Fragment v3: the noun-centric lexicon.  Nouns carry their features, their use-verb and its
particle (Mel'čuk's Oper1); English light verbs (play, take, have) apply the noun's use-verb;
content verbs with noun-dependent stems (break) select on noun feature bits.
Reference translator and sentence generator for the sparse compile in compile_nouns.py.
"""
import random
import en_ja_rnn as M

# --- nouns: ja, animacy class, feature bits, use-verb key
# use-verb keys: suru する/を  hiku 弾く/を  fuku 吹く/を  tataku 叩く/を  noru 乗る/に  hairu 入る/に
#                nomu 飲む/を  toru 撮る/を  kaze ひく/を  aru ある/が  miru 見る/を  abiru 浴びる/を
USE_VERBS = {
    "suru": (["する", "した", "しない", "しなかった"], "を"),
    "hiku": (["弾く", "弾いた", "弾かない", "弾かなかった"], "を"),
    "fuku": (["吹く", "吹いた", "吹かない", "吹かなかった"], "を"),
    "tataku": (["叩く", "叩いた", "叩かない", "叩かなかった"], "を"),
    "noru": (["乗る", "乗った", "乗らない", "乗らなかった"], "に"),
    "hairu": (["入る", "入った", "入らない", "入らなかった"], "に"),
    "nomu": (["飲む", "飲んだ", "飲まない", "飲まなかった"], "を"),
    "toru": (["撮る", "撮った", "撮らない", "撮らなかった"], "を"),
    "kaze": (["ひく", "ひいた", "ひかない", "ひかなかった"], "を"),
    "aru": (["ある", "あった", "ない", "なかった"], "が"),
    "miru": (["見る", "見た", "見ない", "見なかった"], "を"),
    "abiru": (["浴びる", "浴びた", "浴びない", "浴びなかった"], "を"),
}
NOUNS = dict(M.NOUNS)            # name -> (ja, class)
NOUNS.update({
    "trumpet": ("トランペット", "thing"), "flute": ("フルート", "thing"), "drums": ("太鼓", "thing"),
    "bath": ("風呂", "thing"), "medicine": ("薬", "thing"), "photo": ("写真", "thing"), "cold": ("風邪", "thing"),
    "fever": ("熱", "thing"), "dream": ("夢", "thing"), "nap": ("昼寝", "thing"), "walk": ("散歩", "thing"),
    "shower": ("シャワー", "thing"), "taxi": ("タクシー", "thing"), "glass": ("グラス", "thing"), "cup": ("コップ", "thing"),
    "machine": ("機械", "thing"), "stick": ("棒", "thing"), "branch": ("枝", "thing"), "pencil": ("鉛筆", "thing"),
})
for n in ("guitar", "piano", "violin", "soccer", "tennis", "baseball"):
    NOUNS[n] = (NOUNS[n][0], "thing")          # no instrument/sport class any more: the noun carries its use
USE = {"guitar": "hiku", "piano": "hiku", "violin": "hiku", "trumpet": "fuku", "flute": "fuku", "drums": "tataku",
       "soccer": "suru", "tennis": "suru", "baseball": "suru", "nap": "suru", "walk": "suru",
       "bath": "hairu", "medicine": "nomu", "photo": "toru", "cold": "kaze", "fever": "aru", "dream": "miru",
       "shower": "abiru", "train": "noru", "bus": "noru", "taxi": "noru", "car": "noru"}
FEAT = {"glass": {"glass", "cup", "window"}, "long": {"stick", "branch", "pencil", "pen"}}
HUMAN, ANIMATE = M.HUMAN, M.ANIMATE

# --- verbs: lemma -> (base, 3sg, past, forms or variants, complement?)
# light verbs apply the object's use-verb when it has one; otherwise their own default
VERBS = {l: v for l, v in M.VERBS.items() if l != "play"}
for l in VERBS:
    VERBS[l] = VERBS[l][:3] + ({"default": VERBS[l][3]["default"]},) + (VERBS[l][4],)
VERBS["play"] = ("play", "plays", "played", {"default": ["する", "した", "しない", "しなかった"],
                                            "none": ["遊ぶ", "遊んだ", "遊ばない", "遊ばなかった"]}, False)
VERBS["take"] = ("take", "takes", "took", {"default": ["取る", "取った", "取らない", "取らなかった"]}, False)
VERBS["have"] = ("have", "has", "had", {"default": ["持っている", "持っていた", "持っていない", "持っていなかった"]}, False)
VERBS["break"] = ("break", "breaks", "broke", {"default": ["壊す", "壊した", "壊さない", "壊さなかった"],
                                               "glass": ["割る", "割った", "割らない", "割らなかった"],
                                               "long": ["折る", "折った", "折らない", "折らなかった"]}, False)
LIGHT = {"play", "take", "have"}

VERB_FORM_TOKENS = {}
for lemma, (base, s3, past, _, _) in VERBS.items():
    VERB_FORM_TOKENS[base] = (lemma, 0); VERB_FORM_TOKENS[s3] = (lemma, 0); VERB_FORM_TOKENS[past] = (lemma, 1)
PRONOUNS, PRON_TOKENS, ADJS, PREPS, COPULA, AUX, DETS = M.PRONOUNS, M.PRON_TOKENS, M.ADJS, M.PREPS, M.COPULA, M.AUX, M.DETS
COP_FORMS, EXIST, FORM, adj_entry = M.COP_FORMS, M.EXIST, M.FORM, M.adj_entry
EN_TOKENS = sorted(set(NOUNS) | set(PRON_TOKENS) | set(ADJS) | set(VERB_FORM_TOKENS)
                   | set(PREPS) | set(COPULA) | set(AUX) | set(DETS) | {"not", "that", "."})

def predicate_forms(lemma, obj_n, has_obj):
    """forms of the predicate given the object (English noun or None for a pronoun) and the object's particle"""
    table = VERBS[lemma][3]
    if lemma in LIGHT and obj_n in USE:
        return USE_VERBS[USE[obj_n]]
    if not has_obj and "none" in table:
        return table["none"], "を"
    if lemma == "break" and obj_n is not None:
        for feat, members in FEAT.items():
            if obj_n in members: return table[feat], "を"
    return table["default"], "を"

def ref_translate(tokens):
    pos = 0; out = []
    def parse_np():
        nonlocal pos
        t = tokens[pos]
        if t in PRON_TOKENS:
            pos += 1; return [PRONOUNS[PRON_TOKENS[t]][2]], "human", None
        if t in NOUNS:
            pos += 1; return [NOUNS[t][0]], NOUNS[t][1], t
        assert t in DETS, t
        pos += 1; adj = None
        if tokens[pos] in ADJS: adj = tokens[pos]; pos += 1
        n = tokens[pos]; pos += 1
        jp, cls = NOUNS[n]
        return ([adj_entry(adj, cls)[0]] if adj else []) + [jp], cls, n
    def parse_clause(embedded):
        nonlocal pos
        subj_jp, subj_cls, _ = parse_np()
        subj = subj_jp + ["が" if embedded else "は"]
        t = tokens[pos]
        if t in COPULA:
            past = COPULA[t]; pos += 1; neg = 0
            if tokens[pos] == "not": neg = 1; pos += 1
            if tokens[pos] in ADJS:
                pred = [adj_entry(tokens[pos], subj_cls)[1][FORM(past, neg)]]; pos += 1
                return subj + pred
            if tokens[pos] in PREPS:
                p = tokens[pos]; pos += 1
                pp_jp, _, _ = parse_np()
                return subj + pp_jp + [PREPS[p][1]] + [EXIST["animate" if ANIMATE(subj_cls) else "inanimate"][FORM(past, neg)]]
            np_jp, _, _ = parse_np()
            return subj + np_jp + [COP_FORMS[FORM(past, neg)]]
        if t in AUX:
            past = AUX[t]; pos += 1; assert tokens[pos] == "not"; pos += 1; neg = 1
            lemma, _ = VERB_FORM_TOKENS[tokens[pos]]; pos += 1
        else:
            lemma, past = VERB_FORM_TOKENS[t]; pos += 1; neg = 0
        if VERBS[lemma][4]:
            assert tokens[pos] == "that"; pos += 1
            inner = parse_clause(True)
            return subj + inner + ["と"] + [VERBS[lemma][3]["default"][FORM(past, neg)]]
        obj, pp, obj_n = [], [], None
        if tokens[pos] in DETS or tokens[pos] in PRON_TOKENS or tokens[pos] in NOUNS:
            obj_jp, _, obj_n = parse_np()
            obj = obj_jp
        forms, part = predicate_forms(lemma, obj_n, bool(obj))
        if obj: obj = obj + [part]
        if tokens[pos] in PREPS:
            p = tokens[pos]; pos += 1
            pp_jp, _, _ = parse_np()
            pp = pp_jp + [PREPS[p][0]]
        return subj + pp + obj + [forms[FORM(past, neg)]]
    while pos < len(tokens):
        out += parse_clause(False)
        assert tokens[pos] == ".", tokens[pos]; pos += 1
    return out

# --- generator
article = M.article
def gen_np(kind, rng, role, nouns=None):
    if kind == "pron":
        pid = rng.choice(list(PRONOUNS)); sf, of, _ = PRONOUNS[pid]
        return [sf if role == "subj" else of], pid in ("3m", "3f")
    n = rng.choice(nouns or list(NOUNS))
    if n in ("soccer", "tennis", "baseball"):
        return [n], True
    first = rng.choice(list(ADJS)) if kind == "AN" else n
    det = rng.choice(["the", article(first)])
    return [det] + ([first] if kind == "AN" else []) + [n], True
def verb_tokens(lemma, past, neg, third_sg):
    base, s3, pst, _, _ = VERBS[lemma]
    if neg: return ["did" if past else ("does" if third_sg else "do"), "not", base]
    return [pst] if past else [s3 if third_sg else base]
copula_tokens = M.copula_tokens
def gen_clause(sig, rng):
    subj_kind, pred, past, neg, obj_kind, pp_kind, cop_kind = sig
    subj, third = gen_np(subj_kind, rng, "subj"); toks = list(subj)
    if pred == "cop":
        toks.append(copula_tokens(past, third, subj))
        if neg: toks.append("not")
        if cop_kind == "adj": toks.append(rng.choice(list(ADJS)))
        elif cop_kind == "loc": toks.append(rng.choice(list(PREPS))); toks += gen_np(pp_kind, rng, "obj")[0]
        else: toks += gen_np(cop_kind, rng, "obj")[0]
        return toks
    lemmas = [l for l in VERBS if not VERBS[l][4]]
    lemma = rng.choice(lemmas)
    toks += verb_tokens(lemma, past, neg, third)
    if obj_kind:
        nouns = None
        if lemma in LIGHT and rng.random() < 0.7: nouns = list(USE)
        if lemma == "break" and rng.random() < 0.7: nouns = sorted(FEAT["glass"] | FEAT["long"])
        toks += gen_np(obj_kind, rng, "obj", nouns)[0]
    if pp_kind:
        toks.append(rng.choice(list(PREPS))); toks += gen_np(pp_kind, rng, "obj")[0]
    return toks
all_clause_sigs = M.all_clause_sigs
def gen_sentence(outer, inner, rng):
    if outer is None: return gen_clause(inner, rng) + ["."]
    subj_kind, past, neg = outer
    subj, third = gen_np(subj_kind, rng, "subj")
    lemma = rng.choice([l for l in VERBS if VERBS[l][4]])
    return subj + verb_tokens(lemma, past, neg, third) + ["that"] + gen_clause(inner, rng) + ["."]

def test_sentences(rng, n_random=3000):
    sigs = all_clause_sigs(); outers = [None] + [(s, p, q) for s in ("pron", "N", "AN") for p in (0, 1) for q in (0, 1)]
    sents = [gen_sentence(o, s, rng) for o in outers for s in sigs]
    sents += [gen_sentence(rng.choice(outers), rng.choice(sigs), rng) for _ in range(n_random)]
    for nn in NOUNS:
        for s in (["the", nn, "ate", "the", "apple", "."], ["i", "saw", "the", nn, "."], ["i", "went", "to", "the", nn, "."],
                  ["the", "cat", "is", article(nn), nn, "."], ["the", "old", nn, "ran", "."], ["i", "saw", "the", "old", nn, "."],
                  ["the", nn, "is", "old", "."], ["the", nn, "is", "in", "the", "park", "."], ["i", "played", "the", nn, "."],
                  ["i", "took", "the", nn, "."], ["he", "has", "the", nn, "."], ["he", "did", "not", "have", "the", nn, "."],
                  ["i", "broke", "the", nn, "."], ["i", "did", "not", "break", "the", nn, "in", "the", "room", "."],
                  ["i", "took", "the", nn, "in", "the", "park", "."], ["i", "think", "that", "he", "played", "the", nn, "."]):
            sents.append(s)
    for l in VERBS:
        base, s3, past, _, comp = VERBS[l]; tail = ["that", "i", "slept", "."] if comp else ["."]
        for s in (["he", s3] + tail, ["they", base] + tail, ["he", past] + tail, ["he", "does", "not", base] + tail,
                  ["he", "did", "not", base] + tail, ["he", past, "him", "."] if not comp else ["he", past, "that", "she", "ran", "."]):
            sents.append(s)
    return sents
