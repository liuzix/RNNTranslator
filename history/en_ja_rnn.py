r"""
A hand-rolled RNN that translates a controlled fragment of English into Japanese,
with a small semantic feature system.

No training. The network is written as unit equations (each unit is a ReLU of a
linear threshold over named units) and compiled into two weight matrices. The
lexicon rows are generated from the tables below; the control logic is written
by hand in build_network(). Python only compiles, runs and verifies.

Cell (applied once per English token):
    z_t = ReLU(A @ [h_{t-1}; x_t] + a)      # scratch: copies, token features, events
    h_t = ReLU(B @ z_t + b)                 # state: phase, registers, output slots
Output of step t: the Japanese tokens in slots 1..8 of h_t, in slot order.

Bounded memory, as a human interpreter: the only things held are
    one predicate per clause level (lemma / adjective / noun + tense + polarity),
    one object noun phrase (adjective + noun), one preposition, one "embedded" bit,
    one pending attributive adjective (waiting for its noun),
    and a handful of feature bits: subject human / animate, object class, locative.
Depth is 2 clauses (main + one "that"-clause). Nothing is a stack.

Semantics = features on lexical entries + output rows selected by those features:
    "old"   -> 年配 (people) / 古い (things)          adjective selected by its noun
    is/are  -> いる (animate) / ある (inanimate)      existential copula by subject
    "play"  -> 弾く (instrument) / する (sport) / 遊ぶ (no object)   verb by object
    "in"    -> で (action) / に (existence)           postposition by predicate
Japanese word order makes three of these free: the noun or object arrives before
the adjective or verb is emitted, so the selecting feature is already in a register.

Fragment:   S  -> NP (V | AUX not V | COP [not] (ADJ | DET [ADJ] N | P NP)) [NP] [P NP] .
            S  -> NP V_comp that S'      (S' as above, no further embedding)
            NP -> PRON | DET [ADJ] N
"""
import numpy as np

# ----------------------------------------------------------------------------
# Lexicon (English -> Japanese). These tables become weight rows.
# ----------------------------------------------------------------------------
# noun -> (Japanese, class)      class in human / animal / thing / place / instrument / sport
NOUNS = {
    "apple": ("りんご", "thing"), "book": ("本", "thing"), "bread": ("パン", "thing"),
    "water": ("水", "thing"), "tea": ("お茶", "thing"), "coffee": ("コーヒー", "thing"),
    "car": ("車", "thing"), "letter": ("手紙", "thing"), "song": ("歌", "thing"),
    "movie": ("映画", "thing"), "door": ("ドア", "thing"), "window": ("窓", "thing"),
    "flower": ("花", "thing"), "tree": ("木", "thing"), "desk": ("机", "thing"),
    "chair": ("椅子", "thing"), "picture": ("絵", "thing"), "bag": ("かばん", "thing"),
    "pen": ("ペン", "thing"), "phone": ("電話", "thing"), "bus": ("バス", "thing"),
    "train": ("電車", "thing"),
    "cat": ("猫", "animal"), "dog": ("犬", "animal"), "fish": ("魚", "animal"),
    "bird": ("鳥", "animal"), "animal": ("動物", "animal"),
    "teacher": ("先生", "human"), "student": ("学生", "human"), "friend": ("友達", "human"),
    "child": ("子供", "human"), "man": ("男の人", "human"), "woman": ("女の人", "human"),
    "doctor": ("医者", "human"),
    "park": ("公園", "place"), "school": ("学校", "place"), "station": ("駅", "place"),
    "store": ("店", "place"), "house": ("家", "place"), "garden": ("庭", "place"),
    "city": ("町", "place"), "mountain": ("山", "place"), "river": ("川", "place"),
    "room": ("部屋", "place"), "library": ("図書館", "place"),
    "guitar": ("ギター", "instrument"), "piano": ("ピアノ", "instrument"), "violin": ("バイオリン", "instrument"),
    "soccer": ("サッカー", "sport"), "tennis": ("テニス", "sport"), "baseball": ("野球", "sport"),
}
NOUN_CLASSES = ["human", "animal", "thing", "place", "instrument", "sport"]
HUMAN = lambda cls: cls == "human"
ANIMATE = lambda cls: cls in ("human", "animal")

# pronoun id -> (subject form, object form, Japanese); all pronouns are human
PRONOUNS = {
    "1s": ("i", "me", "私"), "2": ("you", "you", "あなた"), "3m": ("he", "him", "彼"),
    "3f": ("she", "her", "彼女"), "1p": ("we", "us", "私たち"), "3p": ("they", "them", "彼ら"),
}

# adjective -> {feature: (attributive form, [present, past, negative, past negative])}
# feature "default" is required; "human" overrides it when the noun is a person.
I_ADJ = lambda stem: (stem + "い", [stem + "い", stem + "かった", stem + "くない", stem + "くなかった"])
NA_ADJ = lambda stem: (stem + "の", [stem + "だ", stem + "だった", stem + "ではない", stem + "ではなかった"])
ADJS = {
    "red": {"default": I_ADJ("赤")}, "big": {"default": I_ADJ("大き")}, "small": {"default": I_ADJ("小さ")},
    "old": {"default": I_ADJ("古"), "human": NA_ADJ("年配")},
    "new": {"default": I_ADJ("新し")}, "cheap": {"default": I_ADJ("安")}, "expensive": {"default": I_ADJ("高")},
    "long": {"default": I_ADJ("長")}, "delicious": {"default": I_ADJ("美味し")},
    "interesting": {"default": I_ADJ("面白")}, "busy": {"default": I_ADJ("忙し")}, "white": {"default": I_ADJ("白")},
}
ADJ_FEATURES = ["default", "human"]

# lemma -> (base, 3sg, past, {object feature: [present, past, negative, past negative]}, complement verb?)
# object features: "default" required; "instrument", "sport" by the object's class; "none" = no object.
VERBS = {
    "eat": ("eat", "eats", "ate", {"default": ["食べる", "食べた", "食べない", "食べなかった"]}, False),
    "see": ("see", "sees", "saw", {"default": ["見る", "見た", "見ない", "見なかった"]}, False),
    "write": ("write", "writes", "wrote", {"default": ["書く", "書いた", "書かない", "書かなかった"]}, False),
    "buy": ("buy", "buys", "bought", {"default": ["買う", "買った", "買わない", "買わなかった"]}, False),
    "drink": ("drink", "drinks", "drank", {"default": ["飲む", "飲んだ", "飲まない", "飲まなかった"]}, False),
    "make": ("make", "makes", "made", {"default": ["作る", "作った", "作らない", "作らなかった"]}, False),
    "open": ("open", "opens", "opened", {"default": ["開ける", "開けた", "開けない", "開けなかった"]}, False),
    "close": ("close", "closes", "closed", {"default": ["閉める", "閉めた", "閉めない", "閉めなかった"]}, False),
    "find": ("find", "finds", "found", {"default": ["見つける", "見つけた", "見つけない", "見つけなかった"]}, False),
    "sing": ("sing", "sings", "sang", {"default": ["歌う", "歌った", "歌わない", "歌わなかった"]}, False),
    "go": ("go", "goes", "went", {"default": ["行く", "行った", "行かない", "行かなかった"]}, False),
    "come": ("come", "comes", "came", {"default": ["来る", "来た", "来ない", "来なかった"]}, False),
    "sleep": ("sleep", "sleeps", "slept", {"default": ["寝る", "寝た", "寝ない", "寝なかった"]}, False),
    "run": ("run", "runs", "ran", {"default": ["走る", "走った", "走らない", "走らなかった"]}, False),
    "swim": ("swim", "swims", "swam", {"default": ["泳ぐ", "泳いだ", "泳がない", "泳がなかった"]}, False),
    "work": ("work", "works", "worked", {"default": ["働く", "働いた", "働かない", "働かなかった"]}, False),
    "wait": ("wait", "waits", "waited", {"default": ["待つ", "待った", "待たない", "待たなかった"]}, False),
    "walk": ("walk", "walks", "walked", {"default": ["歩く", "歩いた", "歩かない", "歩かなかった"]}, False),
    "cry": ("cry", "cries", "cried", {"default": ["泣く", "泣いた", "泣かない", "泣かなかった"]}, False),
    "laugh": ("laugh", "laughs", "laughed", {"default": ["笑う", "笑った", "笑わない", "笑わなかった"]}, False),
    "play": ("play", "plays", "played", {"default": ["する", "した", "しない", "しなかった"],
                                        "instrument": ["弾く", "弾いた", "弾かない", "弾かなかった"],
                                        "sport": ["する", "した", "しない", "しなかった"],
                                        "none": ["遊ぶ", "遊んだ", "遊ばない", "遊ばなかった"]}, False),
    "think": ("think", "thinks", "thought", {"default": ["思う", "思った", "思わない", "思わなかった"]}, True),
    "say": ("say", "says", "said", {"default": ["言う", "言った", "言わない", "言わなかった"]}, True),
    "know": ("know", "knows", "knew", {"default": ["知っている", "知っていた", "知らない", "知らなかった"]}, True),
    "believe": ("believe", "believes", "believed", {"default": ["信じる", "信じた", "信じない", "信じなかった"]}, True),
}
OBJ_FEATURES = ["default", "instrument", "sport", "none"]
# preposition -> (postposition with an action, postposition with existence)
PREPS = {"in": ("で", "に"), "at": ("で", "に"), "to": ("に", "に"), "with": ("と", "と"), "from": ("から", "から")}
COPULA = {"am": 0, "is": 0, "are": 0, "was": 1, "were": 1}          # token -> past?
AUX = {"do": 0, "does": 0, "did": 1}
DETS = ["the", "a", "an"]
COP_FORMS = ["だ", "だった", "ではない", "ではなかった"]
EXIST = {"animate": ["いる", "いた", "いない", "いなかった"], "inanimate": ["ある", "あった", "ない", "なかった"]}
FORM = lambda past, neg: 2 * neg + past          # index into a 4-form list
WITH_INSTRUMENTAL = False   # reference semantics for "with": True after the weight-space edit

VERB_FORM_TOKENS = {}   # English verb token -> (lemma, past?)
for lemma, (base, s3, past, _, _) in VERBS.items():
    VERB_FORM_TOKENS[base] = (lemma, 0)
    VERB_FORM_TOKENS[s3] = (lemma, 0)
    VERB_FORM_TOKENS[past] = (lemma, 1)
PRON_TOKENS = {}        # English pronoun token -> pronoun id
for pid, (sf, of, _) in PRONOUNS.items():
    PRON_TOKENS[sf] = pid
    PRON_TOKENS[of] = pid

EN_TOKENS = sorted(set(NOUNS) | set(PRON_TOKENS) | set(ADJS) | set(VERB_FORM_TOKENS)
                   | set(PREPS) | set(COPULA) | set(AUX) | set(DETS) | {"not", "that", "."})

def adj_entry(adj, cls):
    """(attributive, forms) of adj for a noun class."""
    table = ADJS[adj]
    return table["human"] if (HUMAN(cls) and "human" in table) else table["default"]

def verb_forms(lemma, obj_cls):
    """forms of lemma given the object's class (None = no object)."""
    table = VERBS[lemma][3]
    key = "none" if obj_cls is None else obj_cls
    return table.get(key, table["default"])

# ----------------------------------------------------------------------------
# Reference translator: parse to a tree, then generate. Independent of the RNN.
# ----------------------------------------------------------------------------
def ref_translate(tokens):
    pos = 0
    out = []

    def parse_np():
        """-> (japanese tokens, noun class)"""
        nonlocal pos
        t = tokens[pos]
        if t in PRON_TOKENS:
            pos += 1
            return [PRONOUNS[PRON_TOKENS[t]][2]], "human"
        if t in NOUNS:                                  # bare noun ("play tennis")
            pos += 1
            return [NOUNS[t][0]], NOUNS[t][1]
        assert t in DETS, t
        pos += 1
        adj = None
        if tokens[pos] in ADJS:
            adj = tokens[pos]; pos += 1
        n = tokens[pos]; pos += 1
        jp, cls = NOUNS[n]
        return ([adj_entry(adj, cls)[0]] if adj else []) + [jp], cls

    def parse_clause(embedded):
        nonlocal pos
        subj_jp, subj_cls = parse_np()
        subj = subj_jp + ["が" if embedded else "は"]
        t = tokens[pos]
        if t in COPULA:
            past = COPULA[t]; pos += 1
            neg = 0
            if tokens[pos] == "not":
                neg = 1; pos += 1
            if tokens[pos] in ADJS:
                pred = [adj_entry(tokens[pos], subj_cls)[1][FORM(past, neg)]]; pos += 1
                return subj + pred
            if tokens[pos] in PREPS:                       # existence: X is in Y
                p = tokens[pos]; pos += 1
                pp_jp, _ = parse_np()
                exist = EXIST["animate" if ANIMATE(subj_cls) else "inanimate"]
                return subj + pp_jp + [PREPS[p][1]] + [exist[FORM(past, neg)]]
            np_jp, _ = parse_np()
            return subj + np_jp + [COP_FORMS[FORM(past, neg)]]
        if t in AUX:
            past = AUX[t]; pos += 1
            assert tokens[pos] == "not"; pos += 1
            neg = 1
            lemma, _ = VERB_FORM_TOKENS[tokens[pos]]; pos += 1
        else:
            lemma, past = VERB_FORM_TOKENS[t]; pos += 1
            neg = 0
        if VERBS[lemma][4]:
            assert tokens[pos] == "that"; pos += 1
            inner = parse_clause(True)
            return subj + inner + ["と"] + [verb_forms(lemma, None)[FORM(past, neg)]]
        obj, pp, obj_cls = [], [], None
        if tokens[pos] in DETS or tokens[pos] in PRON_TOKENS or tokens[pos] in NOUNS:
            obj_jp, obj_cls = parse_np()
            obj = obj_jp + ["を"]
        if tokens[pos] in PREPS:
            p = tokens[pos]; pos += 1
            pp_jp, pp_cls = parse_np()
            post = PREPS[p][0]
            if p == "with" and WITH_INSTRUMENTAL and not ANIMATE(pp_cls):
                post = "で"                                   # instrumental: ペンで書く
            pp = pp_jp + [post]
        return subj + pp + obj + [verb_forms(lemma, obj_cls)[FORM(past, neg)]]

    while pos < len(tokens):
        out += parse_clause(False)
        assert tokens[pos] == ".", tokens[pos]
        pos += 1
    return out

# ----------------------------------------------------------------------------
# The compiler: named units, each a ReLU of a linear threshold.
# ----------------------------------------------------------------------------
class Units:
    def __init__(self):
        self.names, self.idx = [], {}
    def add(self, name):
        if name not in self.idx:
            self.idx[name] = len(self.names); self.names.append(name)
        return name
    def __len__(self):
        return len(self.names)

class Layer:
    """targets[i] = ReLU(sum coef * source + bias). Each target defined exactly once."""
    def __init__(self, src, tgt):
        self.src, self.tgt, self.rules, self.macros = src, tgt, {}, {}
    def unit(self, name, terms, bias=0.0):
        assert name not in self.rules, f"{name} defined twice"
        self.tgt.add(name)
        flat = []
        for c, s in terms:
            if s in self.src.idx:
                flat.append((c, s))
            elif s in self.macros:          # a named sum over sources, spliced in
                flat += [(c * c2, s2) for c2, s2 in self.macros[s]]
            else:
                raise AssertionError(f"unknown source {s} for {name}")
        self.rules[name] = (flat, bias)
    def macro(self, name, srcs):
        """Define `name` both as a scratch unit (sum of srcs) and as a macro usable in this layer."""
        self.macros[name] = [(1, s) for s in srcs]
        self.unit(name, [(1, s) for s in srcs])
    def AND(self, name, pos, neg=()):
        """1 iff every pos unit is 1 and every neg unit is 0 (all units are 0/1)."""
        self.unit(name, [(1, p) for p in pos] + [(-1, n) for n in neg], -(len(pos) - 1))
    def SUM(self, name, srcs):
        self.unit(name, [(1, s) for s in srcs])
    def build(self):
        W = np.zeros((len(self.tgt), len(self.src)))
        b = np.zeros(len(self.tgt))
        for name, (terms, bias) in self.rules.items():
            i = self.tgt.idx[name]
            for c, s in terms:
                W[i, self.src.idx[s]] += c
            b[i] += bias
        return W, b

PHASES = ["S0", "S1", "V0", "N0", "N1", "C0", "C1", "EE", "T0", "VV", "O1", "E", "PP0", "PP1", "E2"]
# S0 clause start, expect subject     S1 inside subject NP (after DET)
# V0 expect verb / AUX / COP          N0 after AUX, expect "not"      N1 after "not", expect base verb
# C0 after COP (expect not/ADJ/DET/P) C1 after COP DET (expect ADJ/N)  EE after a copula predicate
# T0 after complement verb, expect "that"
# VV after a verb: expect object / P / "."   O1 inside object NP    E after object: expect P / "."
# PP0 after P: expect DET / PRON      PP1 inside PP NP             E2 after PP: expect "."

def build_network():
    S, X, Z = Units(), Units(), Units()          # state, input, scratch
    for t in EN_TOKENS:
        X.add("x:" + t)
    L2 = Layer(Z, S)

    # --- state registers (declared up front so layer 1 can copy them) ---
    regs = []
    def reg(name):
        S.add(name); regs.append(name); return name
    for ph in PHASES:
        S.add("ph:" + ph)
    reg("emb")
    for lvl in (1, 2):
        for l in VERBS: reg(f"lem{lvl}:{l}")
        reg(f"tense{lvl}"); reg(f"neg{lvl}")
        for a in ADJS: reg(f"padj{lvl}:{a}")
        for n in NOUNS: reg(f"pnoun{lvl}:{n}")
        reg(f"subjhuman{lvl}"); reg(f"subjanim{lvl}")      # subject features
        reg(f"loc{lvl}")                                     # copula + preposition = existence
    for a in ADJS: reg(f"objadj:{a}")
    for n in NOUNS: reg(f"objnoun:{n}")
    for p in PRONOUNS: reg(f"objpron:{p}")
    for f in ("human", "instrument", "sport"): reg(f"objcls:{f}")   # object features, kept to the end
    reg("hadobj")
    for a in ADJS: reg(f"pattr:{a}")      # attributive adjective inside a noun predicate
    for a in ADJS: reg(f"attr:{a}")       # attributive adjective waiting for its noun (subject / PP)
    for p in PREPS: reg(f"prep:{p}")

    # layer-1 source space = state units followed by input units
    SX = Units()
    for n in S.names: SX.add(n)
    for n in X.names: SX.add(n)
    L1 = Layer(SX, Z)

    # ---------------- layer 1: copies ----------------
    for name in regs + ["ph:" + p for p in PHASES]:
        L1.unit("c:" + name, [(1, name)])
    L1.SUM("c:objpresent", [f"objnoun:{n}" for n in NOUNS] + [f"objpron:{p}" for p in PRONOUNS])
    L1.SUM("c:pnoun1any", [f"pnoun1:{n}" for n in NOUNS])
    L1.SUM("c:pnoun2any", [f"pnoun2:{n}" for n in NOUNS])
    L1.SUM("c:pnounhuman", [f"pnoun{lvl}:{n}" for lvl in (1, 2) for n in NOUNS if HUMAN(NOUNS[n][1])])
    L1.SUM("c:locany", ["loc1", "loc2"])

    # ---------------- layer 1: token features (the lexicon as rows) ----------------
    L1.macro("f:det", ["x:" + d for d in DETS])
    L1.macro("f:not", ["x:not"]); L1.macro("f:that", ["x:that"]); L1.macro("f:period", ["x:."])
    L1.macro("f:aux", ["x:" + t for t in AUX]); L1.macro("f:cop", ["x:" + t for t in COPULA])
    L1.macro("f:past", ["x:" + t for t, (_, p) in VERB_FORM_TOKENS.items() if p]
                     + ["x:" + t for t, p in COPULA.items() if p] + ["x:" + t for t, p in AUX.items() if p])
    L1.macro("f:adj_any", ["x:" + a for a in ADJS]); L1.macro("f:noun_any", ["x:" + n for n in NOUNS])
    L1.macro("f:pron_any", ["x:" + t for t in PRON_TOKENS]); L1.macro("f:prep_any", ["x:" + p for p in PREPS])
    L1.macro("f:verb_any", ["x:" + t for t, (l, _) in VERB_FORM_TOKENS.items() if not VERBS[l][4]])
    L1.macro("f:comp_any", ["x:" + t for t, (l, _) in VERB_FORM_TOKENS.items() if VERBS[l][4]])
    L1.macro("f:noun_human", ["x:" + n for n in NOUNS if HUMAN(NOUNS[n][1])])
    L1.macro("f:noun_anim", ["x:" + n for n in NOUNS if ANIMATE(NOUNS[n][1])])
    for f in ("instrument", "sport"):
        L1.macro(f"f:noun_{f}", ["x:" + n for n in NOUNS if NOUNS[n][1] == f])
    for a in ADJS: L1.macro(f"f:adj:{a}", ["x:" + a])
    for n in NOUNS: L1.macro(f"f:noun:{n}", ["x:" + n])
    for p, (sf, of, _) in PRONOUNS.items(): L1.macro(f"f:pron:{p}", sorted({"x:" + sf, "x:" + of}))
    for l in VERBS: L1.macro(f"f:lemma:{l}", ["x:" + t for t, (ll, _) in VERB_FORM_TOKENS.items() if ll == l])
    for p in PREPS: L1.macro(f"f:prep:{p}", ["x:" + p])

    # ---------------- layer 1: events = AND(previous phase, token class) ----------------
    ph = lambda p: "ph:" + p
    transitions = {   # event -> (phase, feature)
        "S0det": ("S0", "f:det"), "S0pron": ("S0", "f:pron_any"), "S0noun": ("S0", "f:noun_any"),
        "S1adj": ("S1", "f:adj_any"), "S1noun": ("S1", "f:noun_any"),
        "V0verb": ("V0", "f:verb_any"), "V0comp": ("V0", "f:comp_any"),
        "V0aux": ("V0", "f:aux"), "V0cop": ("V0", "f:cop"),
        "N0not": ("N0", "f:not"), "N1verb": ("N1", "f:verb_any"), "N1comp": ("N1", "f:comp_any"),
        "C0not": ("C0", "f:not"), "C0adj": ("C0", "f:adj_any"), "C0det": ("C0", "f:det"),
        "C0prep": ("C0", "f:prep_any"), "C0noun": ("C0", "f:noun_any"),
        "C1adj": ("C1", "f:adj_any"), "C1noun": ("C1", "f:noun_any"),
        "VVdet": ("VV", "f:det"), "VVpron": ("VV", "f:pron_any"), "VVprep": ("VV", "f:prep_any"),
        "VVnoun": ("VV", "f:noun_any"),
        "O1adj": ("O1", "f:adj_any"), "O1noun": ("O1", "f:noun_any"),
        "Eprep": ("E", "f:prep_any"),
        "PP0det": ("PP0", "f:det"), "PP0pron": ("PP0", "f:pron_any"), "PP0noun": ("PP0", "f:noun_any"),
        "PP1adj": ("PP1", "f:adj_any"), "PP1noun": ("PP1", "f:noun_any"),
        "T0that": ("T0", "f:that"),
    }
    for ev, (p, f) in transitions.items():
        L1.AND("e:" + ev, [ph(p), f])
    L1.unit("e:flush", [(1, ph("VV")), (1, ph("E")), (1, ph("E2")), (1, ph("EE")), (1, "f:period")], -1)
    L1.unit("e:subj_noun", [(1, ph("S0")), (1, ph("S1")), (1, "f:noun_any")], -1)   # subject NP ends on a noun
    L1.unit("e:pp_noun", [(1, ph("PP0")), (1, ph("PP1")), (1, "f:noun_any")], -1)   # PP NP ends on a noun

    # ---------------- layer 1: register writes, gated by clause level ----------------
    # level 1 = main clause (emb = 0), level 2 = embedded clause (emb = 1)
    def lvl_write(name, pos, lvl, extra=()):
        """AND(pos..., level == lvl); `extra` are further 0/1 terms OR-ed in pairs like (phase, feature)."""
        if lvl == 1:
            L1.unit(name, [(1, p) for p in pos] + [(-1, "emb")], -(len(pos) - 1))
        else:
            L1.unit(name, [(1, p) for p in pos] + [(1, "emb")], -len(pos))
    for lvl in (1, 2):
        emb_term = ((-1, 1)[lvl - 1], "emb")
        base_bias = -1 if lvl == 1 else -2
        for l in VERBS:   # verb lemma is consumed in V0 (inflected) or N1 (base form after "not")
            L1.unit(f"w:lem{lvl}:{l}", [(1, ph("V0")), (1, ph("N1")), (1, f"f:lemma:{l}"), emb_term], base_bias)
        lvl_write(f"w:tense{lvl}", [ph("V0"), "f:past"], lvl)
        L1.unit(f"w:neg{lvl}", [(1, ph("N0")), (1, ph("C0")), (1, "f:not"), emb_term], base_bias)
        for a in ADJS: lvl_write(f"w:padj{lvl}:{a}", [ph("C0"), f"f:adj:{a}"], lvl)
        for n in NOUNS:
            L1.unit(f"w:pnoun{lvl}:{n}", [(1, ph("C0")), (1, ph("C1")), (1, f"f:noun:{n}"), emb_term], base_bias)
        L1.unit(f"w:subjhuman{lvl}", [(1, ph("S0")), (1, ph("S1")), (1, "f:noun_human"), (1, "f:pron_any"), emb_term], base_bias)
        L1.unit(f"w:subjanim{lvl}", [(1, ph("S0")), (1, ph("S1")), (1, "f:noun_anim"), (1, "f:pron_any"), emb_term], base_bias)
        lvl_write(f"w:loc{lvl}", [ph("C0"), "f:prep_any"], lvl)
    for a in ADJS: L1.AND(f"w:objadj:{a}", [ph("O1"), f"f:adj:{a}"])
    for n in NOUNS: L1.unit(f"w:objnoun:{n}", [(1, ph("O1")), (1, ph("VV")), (1, f"f:noun:{n}")], -1)
    for p in PRONOUNS: L1.AND(f"w:objpron:{p}", [ph("VV"), f"f:pron:{p}"])
    L1.unit("w:objcls:human", [(1, ph("O1")), (1, ph("VV")), (1, "f:noun_human"), (1, "f:pron_any")], -1)
    for f in ("instrument", "sport"):
        L1.unit(f"w:objcls:{f}", [(1, ph("O1")), (1, ph("VV")), (1, f"f:noun_{f}")], -1)
    L1.unit("w:hadobj", [(1, ph("O1")), (1, ph("VV")), (1, "f:noun_any"), (1, "f:pron_any")], -1)
    for a in ADJS: L1.AND(f"w:pattr:{a}", [ph("C1"), f"f:adj:{a}"])
    for a in ADJS: L1.unit(f"w:attr:{a}", [(1, ph("S1")), (1, ph("PP1")), (1, f"f:adj:{a}")], -1)
    for p in PREPS: L1.unit(f"w:prep:{p}", [(1, ph("VV")), (1, ph("E")), (1, ph("C0")), (1, f"f:prep:{p}")], -1)

    # ---------------- layer 2: phases ----------------
    next_phase = {
        "S0": ["e:flush", "e:T0that"], "S1": ["e:S0det", "e:S1adj"], "V0": ["e:S0pron", "e:subj_noun"],
        "N0": ["e:V0aux"], "N1": ["e:N0not"], "C0": ["e:V0cop", "e:C0not"], "C1": ["e:C0det", "e:C1adj"],
        "EE": ["e:C0adj", "e:C1noun", "e:C0noun"], "T0": ["e:V0comp", "e:N1comp"], "VV": ["e:V0verb", "e:N1verb"],
        "O1": ["e:VVdet", "e:O1adj"], "E": ["e:VVpron", "e:O1noun", "e:VVnoun"], "PP0": ["e:VVprep", "e:Eprep", "e:C0prep"],
        "PP1": ["e:PP0det", "e:PP1adj"], "E2": ["e:PP0pron", "e:pp_noun"],
    }
    for p, evs in next_phase.items():
        L2.SUM(ph(p), evs)

    # ---------------- layer 2: registers persist until the sentence ends ----------------
    L2.unit("emb", [(1, "c:emb"), (1, "e:T0that"), (-1, "e:flush")])
    for name in regs:
        if name == "emb":
            continue
        terms = [(1, "c:" + name), (1, "w:" + name), (-1, "e:flush")]
        if name.startswith("objadj") or name.startswith("objnoun") or name.startswith("objpron"):
            terms += [(-1, "e:PP0pron"), (-1, "e:pp_noun")]    # the object is emitted early when a PP arrives
        if name.startswith("attr:"):
            terms += [(-1, "e:subj_noun"), (-1, "e:pp_noun")]  # the pending adjective is emitted with its noun
        L2.unit(name, terms)

    # ---------------- layer 2: output slots ----------------
    def out(slot, token, tag, terms, bias):
        L2.unit(f"o{slot}:{token}#{tag}", terms, bias)
    # the subject and the PP noun phrase stream out; a pending adjective is emitted with its noun,
    # in the form the noun's class selects
    noun_end = [(1, "e:subj_noun"), (1, "e:pp_noun")]
    for a in ADJS:
        for feat in ADJ_FEATURES:
            attr = adj_entry(a, "human" if feat == "human" else "thing")[0]
            if feat == "human":
                out(1, attr, f"attr_{feat}:{a}", noun_end + [(1, f"c:attr:{a}"), (1, "f:noun_human")], -2)
            else:
                out(1, attr, f"attr_{feat}:{a}", noun_end + [(1, f"c:attr:{a}"), (-1, "f:noun_human")], -1)
    for n, (jp, _) in NOUNS.items():
        out(2, jp, "subj_or_pp_noun", [(1, "c:ph:S0"), (1, "c:ph:S1"), (1, "c:ph:PP0"), (1, "c:ph:PP1"), (1, f"f:noun:{n}")], -1)
    for p, (_, _, jp) in PRONOUNS.items():
        out(2, jp, "subj_or_pp_pron", [(1, "c:ph:S0"), (1, "c:ph:PP0"), (1, f"f:pron:{p}")], -1)
    out(3, "は", "subj", [(1, "e:subj_noun"), (1, "e:S0pron"), (-1, "c:emb")], 0)
    out(3, "が", "subj", [(1, "e:subj_noun"), (1, "e:S0pron"), (1, "c:emb")], -1)
    # PP ends: postposition (selected by the predicate), then the held object
    ppfl = [(1, "e:pp_noun"), (1, "e:PP0pron")]
    for p, (act, exist) in PREPS.items():
        out(3, act, f"post_action:{p}", ppfl + [(1, f"c:prep:{p}"), (-1, "c:locany")], -1)
        out(3, exist, f"post_exist:{p}", ppfl + [(1, f"c:prep:{p}"), (1, "c:locany")], -2)
    def obj_adj_rules(slot, trigger, tag):
        for a in ADJS:
            attr_h = adj_entry(a, "human")[0]; attr_d = adj_entry(a, "thing")[0]
            out(slot, attr_h, f"{tag}_human:{a}", trigger + [(1, f"c:objadj:{a}"), (1, "c:objcls:human")], -2)
            out(slot, attr_d, f"{tag}_default:{a}", trigger + [(1, f"c:objadj:{a}"), (-1, "c:objcls:human")], -1)
    obj_adj_rules(4, ppfl, "obj_adj_at_pp")
    for n, (jp, _) in NOUNS.items():
        out(5, jp, "obj_noun_at_pp", ppfl + [(1, f"c:objnoun:{n}")], -1)
    for p, (_, _, jp) in PRONOUNS.items():
        out(5, jp, "obj_pron_at_pp", ppfl + [(1, f"c:objpron:{p}")], -1)
    out(6, "を", "at_pp", ppfl + [(1, "c:objpresent")], -1)
    # sentence ends: held object, innermost predicate, [と, main verb]
    fl = [(1, "e:flush")]
    obj_adj_rules(1, fl, "obj_adj_at_end")
    for n, (jp, _) in NOUNS.items():
        out(2, jp, "obj_noun_at_end", fl + [(1, f"c:objnoun:{n}")], -1)
    for p, (_, _, jp) in PRONOUNS.items():
        out(2, jp, "obj_pron_at_end", fl + [(1, f"c:objpron:{p}")], -1)
    out(3, "を", "at_end", fl + [(1, "c:objpresent")], -1)
    for a in ADJS:
        attr_h = adj_entry(a, "human")[0]; attr_d = adj_entry(a, "thing")[0]
        out(4, attr_h, f"pred_attr_human:{a}", fl + [(1, f"c:pattr:{a}"), (1, "c:pnounhuman")], -2)
        out(4, attr_d, f"pred_attr_default:{a}", fl + [(1, f"c:pattr:{a}"), (-1, "c:pnounhuman")], -1)

    def conjugated(slot, tag, lvl, key, forms, inner, extra=()):
        """Emit forms[FORM(past, neg)] of the held item `key` at level lvl when the sentence ends,
        additionally conditioned on `extra` = [(+1 | -1, unit), ...] (unit must be 1 | must be 0).
        inner=True: this level is the innermost clause (emb must equal lvl-1)."""
        for past in (0, 1):
            for neg in (0, 1):
                terms = fl + [(1, f"c:{key}")]
                bias = -1
                if inner:
                    terms.append(((-1, 1)[lvl - 1], "c:emb")); bias -= (lvl - 1)
                else:                      # outer verb: only when there is an embedded clause
                    terms.append((1, "c:emb")); bias -= 1
                terms.append(((-1, 1)[past], f"c:tense{lvl}")); bias -= past
                terms.append(((-1, 1)[neg], f"c:neg{lvl}")); bias -= neg
                for sign, unit in extra:
                    terms.append((sign, unit)); bias -= (sign == 1)
                out(slot, forms[FORM(past, neg)], f"{tag}{lvl}:{key}", terms, bias)
    # object-feature conditions for verb forms: which object classes select which form table
    OBJ_COND = {
        "instrument": [(1, "c:objcls:instrument")],
        "sport": [(1, "c:objcls:sport")],
        "none": [(-1, "c:hadobj")],
    }
    for lvl in (1, 2):
        for l, (_, _, _, table, _) in VERBS.items():
            keys = [k for k in OBJ_FEATURES if k in table]
            if keys == ["default"]:
                conjugated(5, "verb_inner", lvl, f"lem{lvl}:{l}", table["default"], inner=True)
                continue
            # default = had an object whose class selects nothing special
            default_cond = [(1, "c:hadobj")] + [(-1, f"c:objcls:{k}") for k in keys if k in ("instrument", "sport")]
            conjugated(5, "verb_inner_default", lvl, f"lem{lvl}:{l}", table["default"], inner=True, extra=default_cond)
            for k in keys:
                if k != "default":
                    conjugated(5, f"verb_inner_{k}", lvl, f"lem{lvl}:{l}", table[k], inner=True, extra=OBJ_COND[k])
        for a in ADJS:
            conjugated(5, "adj_pred_human", lvl, f"padj{lvl}:{a}", adj_entry(a, "human")[1], inner=True,
                       extra=[(1, f"c:subjhuman{lvl}")])
            conjugated(5, "adj_pred_default", lvl, f"padj{lvl}:{a}", adj_entry(a, "thing")[1], inner=True,
                       extra=[(-1, f"c:subjhuman{lvl}")])
        for n, (jp, _) in NOUNS.items():
            terms = fl + [(1, f"c:pnoun{lvl}:{n}"), ((-1, 1)[lvl - 1], "c:emb")]
            out(5, jp, f"noun_pred{lvl}", terms, -1 - (lvl - 1))
        conjugated(6, "copula", lvl, f"pnoun{lvl}any", COP_FORMS, inner=True)
        conjugated(5, "exist_animate", lvl, f"loc{lvl}", EXIST["animate"], inner=True, extra=[(1, f"c:subjanim{lvl}")])
        conjugated(5, "exist_inanimate", lvl, f"loc{lvl}", EXIST["inanimate"], inner=True, extra=[(-1, f"c:subjanim{lvl}")])
    out(7, "と", "quote", fl + [(1, "c:emb")], -1)
    for l, (_, _, _, table, _) in VERBS.items():
        conjugated(8, "verb_outer", 1, f"lem1:{l}", table["default"], inner=False)

    A0, a = L1.build()                  # columns indexed by SX (registers, phases, then tokens)
    B, b = L2.build()
    for name in S.names:
        assert name in L2.rules, f"state unit {name} has no update rule"
    # re-lay layer-1 columns over the full state vector (output slots were added to S after SX)
    A = np.zeros((len(Z), len(S) + len(X)))
    for name, j in SX.idx.items():
        col = S.idx[name] if name in S.idx else len(S) + X.idx[name]
        A[:, col] = A0[:, j]
    return dict(A=A, a=a, B=B, b=b, S=S, X=X, Z=Z)

# ----------------------------------------------------------------------------
# Running the network
# ----------------------------------------------------------------------------
NSLOTS = 8
relu = lambda v: np.maximum(v, 0.0)

class Translator:
    def __init__(self, weights=None):
        """weights=None: compile from the unit equations (names available, trace works).
        weights=dict from load_weights(): raw matrices + I/O layout only, no unit names."""
        if weights is None:
            net = build_network()
            self.A, self.a, self.B, self.b = net["A"], net["a"], net["B"], net["b"]
            self.S, self.X, self.Z = net["S"], net["X"], net["Z"]
            self.nS = len(self.S)
            self.in_tokens = [n[2:] for n in self.X.names]
            self.out_units = []                                   # (unit index, slot, token)
            for name, i in self.S.idx.items():
                if name.startswith("o") and ":" in name and "#" in name:
                    slot = int(name[1:name.index(":")])
                    token = name[name.index(":") + 1:name.index("#")]
                    self.out_units.append((i, slot, token))
            self.h0 = np.zeros(self.nS); self.h0[self.S.idx["ph:S0"]] = 1.0
        else:
            self.A, self.a, self.B, self.b = weights["A"], weights["a"], weights["B"], weights["b"]
            self.S = self.X = self.Z = None
            self.nS = self.B.shape[0]
            self.in_tokens = list(weights["in_tokens"])
            self.out_units = [(int(i), int(s), t) for i, s, t in weights["out_units"]]
            self.h0 = weights["h0"]
        self.in_idx = {t: i for i, t in enumerate(self.in_tokens)}
        self.slots = {k: [] for k in range(1, NSLOTS + 1)}     # slot -> [(unit index, token)]
        for i, slot, token in self.out_units:
            self.slots[slot].append((i, token))

    def step(self, h, token):
        x = np.zeros(len(self.in_tokens)); x[self.in_idx[token]] = 1.0
        z = relu(self.A @ np.concatenate([h, x]) + self.a)
        return relu(self.B @ z + self.b)

    def dump(self, path):
        """Raw weights plus the I/O layout: input column -> token, output unit -> (slot, token),
        initial state. No scratch or state unit names."""
        np.savez(path, A=self.A, a=self.a, B=self.B, b=self.b, h0=self.h0,
                 in_tokens=np.array(self.in_tokens, dtype=object),
                 out_units=np.array([(i, s, t) for i, s, t in self.out_units], dtype=object))

    def print_weights(self, path):
        """Human-readable listing of every nonzero weight."""
        nS = self.nS
        out_name = {i: f"out[slot{s}:{t}]" for i, s, t in self.out_units}
        with open(path, "w") as f:
            f.write(f"# layer 1: scratch z = ReLU(A @ [h; x] + a); A is {self.A.shape[0]} x {self.A.shape[1]}\n")
            f.write(f"# columns 0..{nS-1} = previous state h, columns {nS}..{nS+len(self.in_tokens)-1} = input token\n")
            f.write(f"# layer 2: state h' = ReLU(B @ z + b); B is {self.B.shape[0]} x {self.B.shape[1]}\n")
            f.write(f"# initial state: h0 has a 1 at index {int(np.argmax(self.h0))}\n#\n")
            for i in range(self.A.shape[0]):
                cols = np.nonzero(self.A[i])[0]
                if len(cols) == 0 and self.a[i] == 0:
                    continue
                parts = []
                for j in cols:
                    label = f"x[{self.in_tokens[j - nS]}]" if j >= nS else f"h[{j}]"
                    parts.append(f"{self.A[i, j]:+g}*{label}")
                f.write(f"z[{i}] = ReLU({' '.join(parts)} {self.a[i]:+g})\n")
            f.write("#\n")
            for i in range(self.B.shape[0]):
                cols = np.nonzero(self.B[i])[0]
                if len(cols) == 0 and self.b[i] == 0:
                    continue
                parts = [f"{self.B[i, j]:+g}*z[{j}]" for j in cols]
                f.write(f"{out_name.get(i, f'h[{i}]')} = ReLU({' '.join(parts)} {self.b[i]:+g})\n")


    def emitted(self, h):
        toks = []
        for k in range(1, NSLOTS + 1):
            hits = [t for i, t in self.slots[k] if h[i] > 0.5]
            assert len(set(hits)) <= 1, f"slot {k} holds {hits}"
            toks += hits[:1]
        return toks

    def translate(self, tokens, trace=False):
        h, out, rows = self.h0, [], []
        for t in tokens:
            h = self.step(h, t)
            e = self.emitted(h)
            out += e
            if trace:
                held = [n for n in self.S.names
                        if h[self.S.idx[n]] > 0.5 and not n.startswith("o") and not n.startswith("ph:")]
                phase = [p for p in PHASES if h[self.S.idx["ph:" + p]] > 0.5]
                rows.append((t, e, phase, held))
        return (out, rows) if trace else out

def load_weights(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

# ----------------------------------------------------------------------------
# Sentence generator over the fragment, with structural signatures
# ----------------------------------------------------------------------------
import random

def article(word):
    return "an" if word[0] in "aeiou" else "a"

def gen_np(kind, rng, role, nouns=None):
    """kind in {'pron','N','AN'}; returns (tokens, third_singular?)"""
    if kind == "pron":
        pid = rng.choice(list(PRONOUNS))
        sf, of, _ = PRONOUNS[pid]
        return [sf if role == "subj" else of], pid in ("3m", "3f")
    n = rng.choice(nouns or list(NOUNS))
    if NOUNS[n][1] == "sport":
        return [n], True
    first = rng.choice(list(ADJS)) if kind == "AN" else n
    det = rng.choice(["the", article(first)])
    return [det] + ([first] if kind == "AN" else []) + [n], True

def verb_tokens(lemma, past, neg, third_sg):
    base, s3, pst, _, _ = VERBS[lemma]
    if neg:
        aux = "did" if past else ("does" if third_sg else "do")
        return [aux, "not", base]
    if past:
        return [pst]
    return [s3 if third_sg else base]

def copula_tokens(past, third_sg, subj_tokens):
    if past:
        return "was" if (third_sg or subj_tokens == ["i"]) else "were"
    if subj_tokens == ["i"]:
        return "am"
    return "is" if third_sg else "are"

def gen_clause(sig, rng):
    """sig = (subj_kind, pred, past, neg, obj_kind, pp_kind, cop_kind)"""
    subj_kind, pred, past, neg, obj_kind, pp_kind, cop_kind = sig
    subj, third = gen_np(subj_kind, rng, "subj")
    toks = list(subj)
    if pred == "cop":
        toks.append(copula_tokens(past, third, subj))
        if neg:
            toks.append("not")
        if cop_kind == "adj":
            toks.append(rng.choice(list(ADJS)))
        elif cop_kind == "loc":
            toks.append(rng.choice(list(PREPS)))
            toks += gen_np(pp_kind, rng, "obj")[0]
        else:
            toks += gen_np(cop_kind, rng, "obj")[0]
        return toks
    lemmas = [l for l in VERBS if not VERBS[l][4]]
    if obj_kind == "pron":
        lemmas.remove("play")
    lemma = rng.choice(lemmas)
    toks += verb_tokens(lemma, past, neg, third)
    if obj_kind:
        nouns = [n for n in NOUNS if NOUNS[n][1] in ("instrument", "sport")] if lemma == "play" else None
        toks += gen_np(obj_kind, rng, "obj", nouns)[0]
    if pp_kind:
        toks.append(rng.choice(list(PREPS)))
        toks += gen_np(pp_kind, rng, "obj")[0]
    return toks

def all_clause_sigs():
    sigs = []
    for subj in ("pron", "N", "AN"):
        for past in (0, 1):
            for neg in (0, 1):
                for obj in (None, "pron", "N", "AN"):
                    for pp in (None, "pron", "N", "AN"):
                        sigs.append((subj, "verb", past, neg, obj, pp, None))
                for cop in ("adj", "N", "AN"):
                    sigs.append((subj, "cop", past, neg, None, None, cop))
                for pp in ("pron", "N", "AN"):
                    sigs.append((subj, "cop", past, neg, None, pp, "loc"))
    return sigs

def gen_sentence(outer, inner, rng):
    """outer: None or (subj_kind, past, neg) for a complement clause; inner: clause sig."""
    if outer is None:
        return gen_clause(inner, rng) + ["."]
    subj_kind, past, neg = outer
    subj, third = gen_np(subj_kind, rng, "subj")
    lemma = rng.choice([l for l in VERBS if VERBS[l][4]])
    return subj + verb_tokens(lemma, past, neg, third) + ["that"] + gen_clause(inner, rng) + ["."]

# ----------------------------------------------------------------------------
def verify(T, n_random=20000, n_streams=500):
    rng = random.Random(0)
    def check(tokens):
        got, want = T.translate(tokens), ref_translate(tokens)
        assert got == want, f"\n  {' '.join(tokens)}\n  rnn: {''.join(got)}\n  ref: {''.join(want)}"

    # 1. Every clause structure, as a main clause and inside every complement structure
    sigs = all_clause_sigs()
    outers = [None] + [(s, p, n) for s in ("pron", "N", "AN") for p in (0, 1) for n in (0, 1)]
    count = 0
    for outer in outers:
        for sig in sigs:
            for _ in range(3):
                check(gen_sentence(outer, sig, rng)); count += 1
    print(f"structures: {len(sigs)} clause shapes x {len(outers)} embeddings, "
          f"{count} sentences, all match the reference")

    # 2. Every lexical entry in every role it can take, and every semantic selection
    count = 0
    for n in NOUNS:
        for s in (["the", n, "ate", "the", "apple", "."], ["i", "saw", "the", n, "."],
                  ["i", "went", "to", "the", n, "."], ["the", "cat", "is", article(n), n, "."],
                  ["the", "old", n, "ran", "."], ["i", "saw", "the", "old", n, "."],
                  ["i", "ran", "with", "the", "old", n, "."], ["the", n, "is", "old", "."],
                  ["the", "cat", "is", "the", "old", n, "."], ["the", n, "is", "in", "the", "park", "."],
                  ["the", n, "was", "not", "at", "the", "station", "."],
                  ["i", "think", "that", "the", n, "is", "red", "."],
                  ["i", "think", "that", "the", n, "was", "not", "in", "the", "room", "."],
                  ["i", "played", "the", n, "."], ["i", "did", "not", "play", "the", n, "in", "the", "park", "."]):
            check(s); count += 1
    for a in ADJS:
        for s in (["the", a, "cat", "ran", "."], ["the", a, "teacher", "ran", "."],
                  ["i", "bought", "the", a, "car", "."], ["i", "saw", "the", a, "doctor", "."],
                  ["i", "bought", "the", a, "car", "in", "the", a, "store", "."],
                  ["the", "car", "is", a, "."], ["the", "car", "was", "not", a, "."],
                  ["the", "teacher", "is", a, "."], ["the", "teacher", "was", "not", a, "."],
                  ["he", "said", "that", "the", "car", "was", a, "."],
                  ["he", "said", "that", "the", "friend", "was", a, "."]):
            check(s); count += 1
    for pid, (sf, of, _) in PRONOUNS.items():
        for s in ([sf, "ran", "."], ["i", "saw", of, "."], ["i", "went", "with", of, "."],
                  ["i", "saw", of, "with", of, "."], ["i", "know", "that", sf, "slept", "."],
                  [sf, "is" if pid in ("3m", "3f") else ("am" if pid == "1s" else "are"), "old", "."],
                  [sf, "was" if pid in ("3m", "3f", "1s") else "were", "in", "the", "park", "."]):
            check(s); count += 1
    for l in VERBS:
        base, s3, past, _, comp = VERBS[l]
        tail = ["that", "i", "slept", "."] if comp else ["."]
        for s in (["he", s3] + tail, ["they", base] + tail, ["he", past] + tail,
                  ["he", "does", "not", base] + tail, ["he", "did", "not", base] + tail):
            check(s); count += 1
    for p in PREPS:
        check(["i", "ran", p, "the", "park", "."]); check(["i", "ate", "the", "bread", p, "the", "park", "."])
        check(["the", "cat", "is", p, "the", "park", "."]); check(["the", "book", "is", p, "the", "park", "."])
        count += 4
    print(f"lexicon: every entry in every role, {count} sentences, all match")

    # 3. Random sentences and multi-sentence streams (tests the end-of-sentence reset)
    for _ in range(n_random):
        check(gen_sentence(rng.choice(outers), rng.choice(sigs), rng))
    for _ in range(n_streams):
        stream = []
        for _ in range(rng.randint(2, 5)):
            stream += gen_sentence(rng.choice(outers), rng.choice(sigs), rng)
        check(stream)
    print(f"random: {n_random:,} sentences and {n_streams} multi-sentence streams match")


if __name__ == "__main__":
    T = Translator()
    print(f"state units {len(T.S)}, scratch units {len(T.Z)}, input tokens {len(T.X)}, "
          f"weights {T.A.size + T.B.size:,} (nonzero {np.count_nonzero(T.A) + np.count_nonzero(T.B):,})")
    verify(T)
    T.dump("en_ja_weights.npz")
    T.print_weights("en_ja_weights.txt")

    # 4. The four semantic selections
    print()
    for s in ["the old teacher did not write the letter .", "the old car is cheap .",
              "the teacher is old .", "the car is old .",
              "the cat is in the park .", "the book is in the room .",
              "i ate an apple in the park .", "i was in the park .",
              "she played the piano .", "she played tennis .",
              "the child played in the garden .",
              "he said that the old doctor was not at the station ."]:
        toks = s.split()
        print(f"{s}\n  -> {''.join(T.translate(toks))}")
    print("\ntrace: the old teacher is in the garden .")
    out, rows = T.translate("the old teacher is in the garden .".split(), trace=True)
    for t, e, phase, held in rows:
        print(f"  {t:>8}  emit {''.join(e) or '-':<12} phase {phase[0] if phase else '-':<4} "
              f"held {' '.join(held)}")
