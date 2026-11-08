"""
Sparse compile of the English->Japanese fragment (same function as en_ja_rnn.py) into a
three-layer ReLU cell, with every shared predicate computed once and the predicate
morphology factored into stem + suffix output units.

    z1 = ReLU(F x + f)              token-class features, from the input only
    z2 = ReLU(G [h; z1; x] + g)     events (phase AND class), register writes
    h' = ReLU(H [h; z1; z2; x] + k) registers persist through an identity edge; output units

The count that matters is nnz(F) + nnz(G) + nnz(H).  A lower bound is derived at the end
from the function itself, not from this construction.
"""
import os, sys, random
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import en_ja_rnn as M0
import fragment3 as M

NOUNS, PRONOUNS, ADJS, VERBS, PREPS = M.NOUNS, M.PRONOUNS, M.ADJS, M.VERBS, M.PREPS
COP_FORMS, EXIST, FORM, HUMAN, ANIMATE = M.COP_FORMS, M.EXIST, M.FORM, M.HUMAN, M.ANIMATE
USE, USE_VERBS, FEAT, LIGHT = M.USE, M.USE_VERBS, M.FEAT, M.LIGHT
EN = M.EN_TOKENS
x_of = {t: i for i, t in enumerate(EN)}

class Space:
    def __init__(self): self.names, self.idx = [], {}
    def add(self, n):
        if n not in self.idx: self.idx[n] = len(self.names); self.names.append(n)
        return n
    def __len__(self): return len(self.names)

class Lay:
    """one layer: targets = ReLU(sum coef*source + bias); sources are named in `spaces` with offsets"""
    def __init__(self, tgt, spaces):
        self.tgt, self.spaces, self.rules = tgt, spaces, {}
    def unit(self, name, terms, bias=0.0):
        assert name not in self.rules, name
        self.tgt.add(name)
        for c, s in terms:
            assert any(s in sp_.idx for sp_, _ in self.spaces), f"{s} not a source of {name}"
        self.rules[name] = (terms, bias)
    def AND(self, name, pos, neg=()):
        self.unit(name, [(1, p) for p in pos] + [(-1, n) for n in neg], -(len(pos) - 1))
    def col(self, s):
        for sp_, off in self.spaces:
            if s in sp_.idx: return off + sp_.idx[s]
    def build(self):
        ncol = sum(len(sp_) for sp_, _ in self.spaces)
        r, c, v, b = [], [], [], np.zeros(len(self.tgt))
        for name, (terms, bias) in self.rules.items():
            i = self.tgt.idx[name]; b[i] = bias
            for coef, s in terms:
                r.append(i); c.append(self.col(s)); v.append(coef)
        W = sp.csr_matrix((v, (r, c)), shape=(len(self.tgt), ncol)); W.sum_duplicates()
        return W, b

PH = ["S0", "S1", "V0", "N0", "N1", "C0", "C1", "EE", "T0", "VV", "O1", "E", "PP0", "PP1", "E2"]
GODAN_SUFFIX = {   # dictionary ending -> suffixes for [present, past, neg, past-neg]
    "う": ["う", "った", "わない", "わなかった"], "く": ["く", "いた", "かない", "かなかった"],
    "ぐ": ["ぐ", "いだ", "がない", "がなかった"], "す": ["す", "した", "さない", "さなかった"],
    "つ": ["つ", "った", "たない", "たなかった"], "ぬ": ["ぬ", "んだ", "なない", "ななかった"],
    "ぶ": ["ぶ", "んだ", "ばない", "ばなかった"], "む": ["む", "んだ", "まない", "まなかった"],
    "る": ["る", "った", "らない", "らなかった"],
}
def split_forms(forms):
    """stem + 4 suffixes for a 4-form paradigm, as the longest common prefix (the data decides the class)"""
    stem = forms[0]
    for f in forms[1:]:
        while not f.startswith(stem): stem = stem[:-1]
    return stem, [f[len(stem):] for f in forms]

def build():
    X, H, Z1, Z2 = Space(), Space(), Space(), Space()
    for t in EN: X.add("x:" + t)
    # ---------------- state ----------------
    for p in PH: H.add("ph:" + p)
    regs = []
    def reg(n): H.add(n); regs.append(n); return n
    reg("emb")
    for L in (1, 2):
        for l in VERBS: reg(f"lem{L}:{l}")
        reg(f"tense{L}"); reg(f"neg{L}")
        for a in ADJS: reg(f"padj{L}:{a}")
        for n in NOUNS: reg(f"pnoun{L}:{n}")
        reg(f"subjhuman{L}"); reg(f"subjanim{L}"); reg(f"loc{L}")
    for a in ADJS: reg(f"objadj:{a}")
    for n in NOUNS: reg(f"objnoun:{n}")
    for p in PRONOUNS: reg(f"objpron:{p}")
    reg("objcls:human")
    for u in USE_VERBS: reg(f"objuse:{u}")             # the object's use-verb, carried by the noun
    for pt in ("に", "が"): reg(f"objpart:{pt}")        # the object's particle when a light verb applies it
    for ft in FEAT: reg(f"objfeat:{ft}")                # noun feature bits read by content verbs
    reg("hadobj")
    for a in ADJS: reg(f"pattr:{a}")
    for a in ADJS: reg(f"attr:{a}")
    for p in PREPS: reg(f"prep:{p}")
    n_state_regs = len(H)

    # ---------------- layer 1: class features, each computed once ----------------
    L1 = Lay(Z1, [(X, 0)])
    xs = lambda toks: [(1, "x:" + t) for t in toks]
    L1.unit("f:det", xs(M0.DETS)); L1.unit("f:not", xs(["not"])); L1.unit("f:that", xs(["that"])); L1.unit("f:period", xs(["."]))
    L1.unit("f:aux", xs(M0.AUX)); L1.unit("f:cop", xs(M0.COPULA))
    L1.unit("f:past", xs([t for t, (_, p) in M.VERB_FORM_TOKENS.items() if p] + [t for t, p in M0.COPULA.items() if p] + [t for t, p in M0.AUX.items() if p]))
    L1.unit("f:adj_any", xs(ADJS)); L1.unit("f:noun_any", xs(NOUNS)); L1.unit("f:pron_any", xs(M0.PRON_TOKENS)); L1.unit("f:prep_any", xs(PREPS))
    L1.unit("f:verb_any", xs([t for t, (l, _) in M.VERB_FORM_TOKENS.items() if not VERBS[l][4]]))
    L1.unit("f:comp_any", xs([t for t, (l, _) in M.VERB_FORM_TOKENS.items() if VERBS[l][4]]))
    L1.unit("f:noun_human", xs([n for n in NOUNS if HUMAN(NOUNS[n][1])])); L1.unit("f:noun_anim", xs([n for n in NOUNS if ANIMATE(NOUNS[n][1])]))
    for u in USE_VERBS: L1.unit(f"f:use:{u}", xs([n for n in NOUNS if USE.get(n) == u]))
    for pt in ("に", "が"): L1.unit(f"f:part:{pt}", xs([n for n in NOUNS if n in USE and USE_VERBS[USE[n]][1] == pt]))
    for ft, members in FEAT.items(): L1.unit(f"f:feat:{ft}", xs(sorted(members)))
    for l in VERBS: L1.unit(f"f:lemma:{l}", xs([t for t, (ll, _) in M.VERB_FORM_TOKENS.items() if ll == l]))
    for p, (sf, of, _) in PRONOUNS.items(): L1.unit(f"f:pron:{p}", xs(sorted({sf, of})))
    f = lambda n: "f:" + n

    # ---------------- layer 2: events and writes ----------------
    L2 = Lay(Z2, [(H, 0), (Z1, len(H)), (X, len(H) + len(Z1))])   # H is sized later; offsets fixed after outputs are added
    ph = lambda p: "ph:" + p
    E = {"S0det": ("S0", "det"), "S0pron": ("S0", "pron_any"), "S0noun": ("S0", "noun_any"),
         "S1adj": ("S1", "adj_any"), "S1noun": ("S1", "noun_any"),
         "V0verb": ("V0", "verb_any"), "V0comp": ("V0", "comp_any"), "V0aux": ("V0", "aux"), "V0cop": ("V0", "cop"),
         "N0not": ("N0", "not"), "N1verb": ("N1", "verb_any"), "N1comp": ("N1", "comp_any"),
         "C0not": ("C0", "not"), "C0adj": ("C0", "adj_any"), "C0det": ("C0", "det"), "C0prep": ("C0", "prep_any"), "C0noun": ("C0", "noun_any"),
         "C1adj": ("C1", "adj_any"), "C1noun": ("C1", "noun_any"),
         "VVdet": ("VV", "det"), "VVpron": ("VV", "pron_any"), "VVprep": ("VV", "prep_any"), "VVnoun": ("VV", "noun_any"),
         "O1adj": ("O1", "adj_any"), "O1noun": ("O1", "noun_any"), "Eprep": ("E", "prep_any"),
         "PP0det": ("PP0", "det"), "PP0pron": ("PP0", "pron_any"), "PP0noun": ("PP0", "noun_any"),
         "PP1adj": ("PP1", "adj_any"), "PP1noun": ("PP1", "noun_any"), "T0that": ("T0", "that")}
    for ev, (p, c) in E.items(): L2.AND("e:" + ev, [ph(p), f(c)])
    L2.unit("e:flush", [(1, ph("VV")), (1, ph("E")), (1, ph("E2")), (1, ph("EE")), (1, f("period"))], -1)
    L2.unit("e:subj_noun", [(1, ph("S0")), (1, ph("S1")), (1, f("noun_any"))], -1)
    L2.unit("e:pp_noun", [(1, ph("PP0")), (1, ph("PP1")), (1, f("noun_any"))], -1)
    L2.unit("e:noun_end", [(1, ph("S0")), (1, ph("S1")), (1, ph("PP0")), (1, ph("PP1")), (1, f("noun_any"))], -1)
    L2.AND("e:PPend_pron", [ph("PP0"), f("pron_any")])
    # shared sums over registers, computed once
    L2.unit("s:pnounhuman", [(1, f"pnoun{L}:{n}") for L in (1, 2) for n in NOUNS if HUMAN(NOUNS[n][1])])
    for L in (1, 2): L2.unit(f"s:pnoun{L}any", [(1, f"pnoun{L}:{n}") for n in NOUNS])
    L2.unit("s:objpresent", [(1, f"objnoun:{n}") for n in NOUNS] + [(1, f"objpron:{p}") for p in PRONOUNS])
    lvl = lambda L: [((-1, 1)[L - 1], "emb")]
    lb = lambda L, k: -(k - 1) - (L - 1)         # bias for an AND of k positive terms gated to level L
    for L in (1, 2):
        for l in VERBS:
            L2.unit(f"w:lem{L}:{l}", [(1, ph("V0")), (1, ph("N1")), (1, f(f"lemma:{l}"))] + lvl(L), lb(L, 2))
        L2.unit(f"w:tense{L}", [(1, ph("V0")), (1, f("past"))] + lvl(L), lb(L, 2))
        L2.unit(f"w:neg{L}", [(1, ph("N0")), (1, ph("C0")), (1, f("not"))] + lvl(L), lb(L, 2))
        for a in ADJS: L2.unit(f"w:padj{L}:{a}", [(1, ph("C0")), (1, "x:" + a)] + lvl(L), lb(L, 2))
        for n in NOUNS: L2.unit(f"w:pnoun{L}:{n}", [(1, ph("C0")), (1, ph("C1")), (1, "x:" + n)] + lvl(L), lb(L, 2))
        L2.unit(f"w:subjhuman{L}", [(1, ph("S0")), (1, ph("S1")), (1, f("noun_human")), (1, f("pron_any"))] + lvl(L), lb(L, 2))
        L2.unit(f"w:subjanim{L}", [(1, ph("S0")), (1, ph("S1")), (1, f("noun_anim")), (1, f("pron_any"))] + lvl(L), lb(L, 2))
        L2.unit(f"w:loc{L}", [(1, ph("C0")), (1, f("prep_any"))] + lvl(L), lb(L, 2))
    for a in ADJS: L2.AND(f"w:objadj:{a}", [ph("O1"), "x:" + a])
    for n in NOUNS: L2.unit(f"w:objnoun:{n}", [(1, ph("O1")), (1, ph("VV")), (1, "x:" + n)], -1)
    for p in PRONOUNS: L2.AND(f"w:objpron:{p}", [ph("VV"), f(f"pron:{p}")])
    L2.unit("w:objcls:human", [(1, ph("O1")), (1, ph("VV")), (1, f("noun_human")), (1, f("pron_any"))], -1)
    for u in USE_VERBS: L2.unit(f"w:objuse:{u}", [(1, ph("O1")), (1, ph("VV")), (1, f(f"use:{u}"))], -1)
    for pt in ("に", "が"): L2.unit(f"w:objpart:{pt}", [(1, ph("O1")), (1, ph("VV")), (1, f(f"part:{pt}"))], -1)
    for ft in FEAT: L2.unit(f"w:objfeat:{ft}", [(1, ph("O1")), (1, ph("VV")), (1, f(f"feat:{ft}"))], -1)
    # a light verb is held, and the object carries a use-verb: the noun's verb replaces the verb's
    L2.unit("s:light", [(1, f"lem{L}:{l}") for L in (1, 2) for l in LIGHT])
    L2.unit("s:use", [(1, f"lem{L}:{l}") for L in (1, 2) for l in LIGHT] + [(1, f"objuse:{u}") for u in USE_VERBS], -1)
    L2.unit("s:nonwo", [(1, f"lem{L}:{l}") for L in (1, 2) for l in LIGHT] + [(1, "objpart:に"), (1, "objpart:が")], -1)
    L2.unit("w:hadobj", [(1, ph("O1")), (1, ph("VV")), (1, f("noun_any")), (1, f("pron_any"))], -1)
    for a in ADJS: L2.AND(f"w:pattr:{a}", [ph("C1"), "x:" + a])
    for a in ADJS: L2.unit(f"w:attr:{a}", [(1, ph("S1")), (1, ph("PP1")), (1, "x:" + a)], -1)
    for p in PREPS: L2.unit(f"w:prep:{p}", [(1, ph("VV")), (1, ph("E")), (1, ph("C0")), (1, "x:" + p)], -1)

    # ---------------- layer 3: next state and outputs ----------------
    L3 = Lay(H, [(H, 0), (Z1, 0), (Z2, 0), (X, 0)])      # offsets fixed below
    out_units = {}   # name -> (sort key, token)
    def out(slot, token, tag, terms, bias):
        name = f"o{slot}:{token}#{tag}"
        L3.unit(name, terms, bias); out_units[name] = (slot, token)
    NEXT = {"S0": ["e:flush", "e:T0that"], "S1": ["e:S0det", "e:S1adj"], "V0": ["e:S0pron", "e:subj_noun"],
            "N0": ["e:V0aux"], "N1": ["e:N0not"], "C0": ["e:V0cop", "e:C0not"], "C1": ["e:C0det", "e:C1adj"],
            "EE": ["e:C0adj", "e:C1noun", "e:C0noun"], "T0": ["e:V0comp", "e:N1comp"], "VV": ["e:V0verb", "e:N1verb"],
            "O1": ["e:VVdet", "e:O1adj"], "E": ["e:VVpron", "e:O1noun", "e:VVnoun"], "PP0": ["e:VVprep", "e:Eprep", "e:C0prep"],
            "PP1": ["e:PP0det", "e:PP1adj"], "E2": ["e:PPend_pron", "e:pp_noun"]}
    for p, evs in NEXT.items(): L3.unit(ph(p), [(1, e) for e in evs])
    ppfl = [(1, "e:pp_noun"), (1, "e:PPend_pron")]
    L3.unit("emb", [(1, "emb"), (1, "e:T0that"), (-1, "e:flush")])
    for r in regs:
        if r == "emb": continue
        terms = [(1, r), (1, "w:" + r), (-1, "e:flush")]
        if r.startswith(("objadj", "objnoun", "objpron")): terms += [(-1, "e:pp_noun"), (-1, "e:PPend_pron")]
        if r.startswith("attr:"): terms += [(-1, "e:subj_noun"), (-1, "e:pp_noun")]
        L3.unit(r, terms)
    # --- streaming emissions: subject / PP noun phrase, with a pending adjective selected by the noun's class
    for a in ADJS:
        attr_h, attr_d = M.adj_entry(a, "human")[0], M.adj_entry(a, "thing")[0]
        out(1, attr_h, f"attr_h:{a}", [(1, "e:noun_end"), (1, f"attr:{a}"), (1, f("noun_human"))], -2)
        out(1, attr_d, f"attr_d:{a}", [(1, "e:noun_end"), (1, f"attr:{a}"), (-1, f("noun_human"))], -1)
    for n, (jp, _) in NOUNS.items(): out(2, jp, "now", [(1, "e:noun_end"), (1, "x:" + n)], -1)
    for p, (_, _, jp) in PRONOUNS.items(): out(2, jp, "now", [(1, "e:S0pron"), (1, "e:PPend_pron"), (1, f(f"pron:{p}"))], -1)
    out(3, "は", "subj", [(1, "e:subj_noun"), (1, "e:S0pron"), (-1, "emb")], 0)
    out(3, "が", "subj", [(1, "e:subj_noun"), (1, "e:S0pron"), (1, "emb")], -1)
    for p, (act, ex) in PREPS.items():
        out(3, act, f"post_act:{p}", ppfl + [(1, f"prep:{p}"), (-1, "loc1"), (-1, "loc2")], -1)
        out(3, ex, f"post_ex:{p}", ppfl + [(1, f"prep:{p}"), (1, "loc1"), (1, "loc2")], -2)
    # --- the held object, emitted when a PP ends or the sentence ends (same units, two triggers)
    trig = ppfl + [(1, "e:flush")]
    for a in ADJS:
        attr_h, attr_d = M.adj_entry(a, "human")[0], M.adj_entry(a, "thing")[0]
        out(4, attr_h, f"obj_h:{a}", trig + [(1, f"objadj:{a}"), (1, "objcls:human")], -2)
        out(4, attr_d, f"obj_d:{a}", trig + [(1, f"objadj:{a}"), (-1, "objcls:human")], -1)
    for n, (jp, _) in NOUNS.items(): out(5, jp, "obj", trig + [(1, f"objnoun:{n}")], -1)
    for p, (_, _, jp) in PRONOUNS.items(): out(5, jp, "obj", trig + [(1, f"objpron:{p}")], -1)
    out(6, "を", "obj", trig + [(1, "s:objpresent"), (-1, "s:nonwo")], -1)
    for pt in ("に", "が"): out(6, pt, "obj_use", trig + [(1, f"objpart:{pt}"), (1, "s:light"), (1, "s:objpresent")], -3)
    # --- sentence end: predicate noun with its attributive adjective, factored predicate, と, outer verb
    fl = [(1, "e:flush")]
    for a in ADJS:
        attr_h, attr_d = M.adj_entry(a, "human")[0], M.adj_entry(a, "thing")[0]
        out(7, attr_h, f"pattr_h:{a}", fl + [(1, f"pattr:{a}"), (1, "s:pnounhuman")], -2)
        out(7, attr_d, f"pattr_d:{a}", fl + [(1, f"pattr:{a}"), (-1, "s:pnounhuman")], -1)
    # innermost predicate = level 2 if emb else level 1; the stem is chosen by the held lemma, the
    # suffix by (paradigm class, tense, polarity). Stems and suffixes are separate output units.
    def inner(L):    # terms making "this level is innermost"
        return [((-1, 1)[L - 1], "emb")], -(L - 1)
    simple = {l: split_forms(VERBS[l][3]["default"]) for l in VERBS if l not in LIGHT and l != "break"}
    vparadigms = {}
    for l, (stem, suf) in simple.items(): vparadigms.setdefault(tuple(suf), []).append(l)
    for L in (1, 2):
        g, gb = inner(L)
        tn = lambda past, neg: [((-1, 1)[past], f"tense{L}"), ((-1, 1)[neg], f"neg{L}")]
        # ordinary verbs: stem by lemma, suffix shared by paradigm
        for l, (stem, suf) in simple.items():
            out(8, stem, f"stem{L}:{l}", fl + [(1, f"lem{L}:{l}")] + g, -1 + gb)
        for suffixes, members in vparadigms.items():
            for past in (0, 1):
                for neg in (0, 1):
                    out(9, suffixes[FORM(past, neg)], f"suf{L}:{hash(suffixes) & 0xffff}:{past}{neg}",
                        fl + [(1, f"lem{L}:{l}") for l in members] + tn(past, neg) + g, -1 - past - neg + gb)
        # light verbs: the noun's use-verb when the object carries one, else the verb's own default
        uparadigms = {}
        for u, (forms, _) in USE_VERBS.items():
            stem, suf = split_forms(forms)
            out(8, stem, f"ustem{L}:{u}", fl + [(1, "s:use"), (1, f"objuse:{u}")] + g, -2 + gb)
            uparadigms.setdefault(tuple(suf), []).append(u)
        for suffixes, members in uparadigms.items():
            for past in (0, 1):
                for neg in (0, 1):
                    out(9, suffixes[FORM(past, neg)], f"usuf{L}:{hash(suffixes) & 0xffff}:{past}{neg}",
                        fl + [(1, "s:use")] + [(1, f"objuse:{u}") for u in members] + tn(past, neg) + g, -2 - past - neg + gb)
        lparadigms = {}
        for l in LIGHT:
            table = VERBS[l][3]
            for key, forms in table.items():
                stem, suf = split_forms(forms)
                cond, cb = ([(-1, "s:use")], 0) if key == "default" else ([(-1, "hadobj")], 0)
                if key == "default" and "none" in table: cond, cb = [(-1, "s:use"), (1, "hadobj")], -1
                out(8, stem, f"lstem{L}:{l}:{key}", fl + [(1, f"lem{L}:{l}")] + g + cond, -1 + gb + cb)
                lparadigms.setdefault(tuple(suf), []).append((l, key))
        for suffixes, members in lparadigms.items():
            for past in (0, 1):
                for neg in (0, 1):
                    for l, key in members:
                        cond, cb = ([(-1, "s:use")], 0) if key == "default" else ([(-1, "hadobj")], 0)
                        if key == "default" and "none" in VERBS[l][3]: cond, cb = [(-1, "s:use"), (1, "hadobj")], -1
                        out(9, suffixes[FORM(past, neg)], f"lsuf{L}:{l}:{key}:{past}{neg}",
                            fl + [(1, f"lem{L}:{l}")] + tn(past, neg) + g + cond, -1 - past - neg + gb + cb)
        # break: stem selected by the object's feature bits, suffix by the selected paradigm
        btable = VERBS["break"][3]
        for key, forms in btable.items():
            stem, suf = split_forms(forms)
            cond, cb = ([(1, f"objfeat:{key}")], -1) if key in FEAT else ([(-1, f"objfeat:{ft}") for ft in FEAT], 0)
            out(8, stem, f"bstem{L}:{key}", fl + [(1, f"lem{L}:break")] + g + cond, -1 + gb + cb)
            for past in (0, 1):
                for neg in (0, 1):
                    out(9, suf[FORM(past, neg)], f"bsuf{L}:{key}:{past}{neg}",
                        fl + [(1, f"lem{L}:break")] + tn(past, neg) + g + cond, -1 - past - neg + gb + cb)
        # adjective predicates: stem by adjective, suffix shared by paradigm; "old" selects by the subject
        aparadigms = {}
        for a in ADJS:
            same = M.adj_entry(a, "human") == M.adj_entry(a, "thing")
            feats = (("any", [], 0),) if same else (("human", [(1, f"subjhuman{L}")], -1), ("thing", [(-1, f"subjhuman{L}")], 0))
            for feat, hcond, hb in feats:
                stem, suffixes = split_forms(M.adj_entry(a, "thing" if feat == "any" else feat)[1])
                out(8, stem, f"astem{L}:{a}:{feat}", fl + [(1, f"padj{L}:{a}")] + g + hcond, -1 + gb + hb)
                aparadigms.setdefault((tuple(suffixes), feat), []).append(a)
        for (suffixes, feat), members in aparadigms.items():
            hcond, hb = {"any": ([], 0), "human": ([(1, f"subjhuman{L}")], -1), "thing": ([(-1, f"subjhuman{L}")], 0)}[feat]
            for past in (0, 1):
                for neg in (0, 1):
                    out(9, suffixes[FORM(past, neg)], f"asuf{L}:{feat}:{hash(suffixes) & 0xffff}:{past}{neg}",
                        fl + [(1, f"padj{L}:{a}") for a in members] + tn(past, neg) + g + hcond, -1 - past - neg + gb + hb)
        # noun predicate: the noun, then a copula suffix
        for n, (jp, _) in NOUNS.items(): out(8, jp, f"npred{L}", fl + [(1, f"pnoun{L}:{n}")] + g, -1 + gb)
        for past in (0, 1):
            for neg in (0, 1):
                out(9, COP_FORMS[FORM(past, neg)], f"cop{L}:{past}{neg}", fl + [(1, f"s:pnoun{L}any")] + tn(past, neg) + g, -1 - past - neg + gb)
        # existence
        for feat, cond, cb in (("animate", [(1, f"subjanim{L}")], -1), ("inanimate", [(-1, f"subjanim{L}")], 0)):
            for past in (0, 1):
                for neg in (0, 1):
                    out(9, EXIST[feat][FORM(past, neg)], f"exist{L}:{feat}:{past}{neg}",
                        fl + [(1, f"loc{L}")] + tn(past, neg) + g + cond, -1 - past - neg + gb + cb)
    out(10, "と", "quote", fl + [(1, "emb")], -1)
    # outer verb (level 1, only when embedded): stem + suffix
    for l, (_, _, _, table, _) in VERBS.items():
        stem, suffixes = split_forms(table["default"])
        out(11, stem, f"ostem:{l}", fl + [(1, "emb"), (1, f"lem1:{l}")], -2)
    oparadigms = {}
    for l, (_, _, _, table, _) in VERBS.items():
        stem, suffixes = split_forms(table["default"]); oparadigms.setdefault(tuple(suffixes), []).append(l)
    for suffixes, members in oparadigms.items():
        for past in (0, 1):
            for neg in (0, 1):
                out(12, suffixes[FORM(past, neg)], f"osuf:{hash(suffixes) & 0xffff}:{past}{neg}",
                    fl + [(1, "emb")] + [(1, f"lem1:{l}") for l in members] + [((-1, 1)[past], "tense1"), ((-1, 1)[neg], "neg1")], -2 - past - neg)

    # fix layer offsets now that H is complete
    L2.spaces = [(H, 0), (Z1, len(H)), (X, len(H) + len(Z1))]
    L3.spaces = [(H, 0), (Z1, len(H)), (Z2, len(H) + len(Z1)), (X, len(H) + len(Z1) + len(Z2))]
    F, fb = L1.build(); G, gb_ = L2.build(); Hm, hb = L3.build()
    return dict(F=F, f=fb, G=G, g=gb_, H=Hm, h=hb, X=X, Hs=H, Z1=Z1, Z2=Z2, out=out_units, n_regs=n_state_regs)

relu = lambda v: np.maximum(v, 0.0)
class Sparse:
    def __init__(self):
        self.net = build()
        n = self.net
        self.nX, self.nH, self.nZ1, self.nZ2 = len(n["X"]), len(n["Hs"]), len(n["Z1"]), len(n["Z2"])
        self.h0 = np.zeros(self.nH); self.h0[n["Hs"].idx["ph:S0"]] = 1.0
        self.out_idx = sorted((n["Hs"].idx[name], key) for name, key in n["out"].items())
        self.oi = np.array([i for i, _ in self.out_idx]); self.ok = [k for _, k in self.out_idx]
        self.active = np.zeros(self.nH + self.nZ1 + self.nZ2, dtype=bool)
    def nnz(self):
        n = self.net; return n["F"].nnz + n["G"].nnz + n["H"].nnz
    def translate(self, tokens):
        n = self.net; h = self.h0.copy(); out = []
        for t in tokens:
            x = np.zeros(self.nX); x[n["X"].idx["x:" + t]] = 1.0
            z1 = relu(n["F"] @ x + n["f"])
            z2 = relu(n["G"] @ np.concatenate([h, z1, x]) + n["g"])
            h = relu(n["H"] @ np.concatenate([h, z1, z2, x]) + n["h"])
            self.active |= np.concatenate([h, z1, z2]) > 0.5
            fired = [(self.ok[j][0], self.ok[j][1]) for j in range(len(self.oi)) if h[self.oi[j]] > 0.5]
            fired.sort()
            seen = set()
            for slot, tok in fired:
                assert slot not in seen, f"slot {slot} twice: {fired}"
                seen.add(slot); out.append(tok)
        return "".join(out)

if __name__ == "__main__":
    S = Sparse()
    n = S.net
    print(f"units: state {S.nH} (registers {n['n_regs']}, outputs {len(n['out'])}), scratch {S.nZ1} + {S.nZ2}")
    print(f"nonzero weights: F {n['F'].nnz} + G {n['G'].nnz} + H {n['H'].nnz} = {S.nnz()}   (v2 sparse: 5,242; dense: 8,279)")
    # same function as the reference, including the four semantic selections
    rng = random.Random(0)
    sents = M.test_sentences(rng, 3000)
    for e in sents:
        got, want = S.translate(e), "".join(M.ref_translate(e))
        assert got == want, f"\n  {' '.join(e)}\n  got  {got}\n  want {want}"
    print(f"verified identical to the reference on {len(sents)} sentences")
    # ---- lower bound from the function: every unit that ever fires needs an incoming edge;
    #      every unit that fires AND depends on two independent sources needs two.
    names = n["Hs"].names + n["Z1"].names + n["Z2"].names
    act = [nm for nm, a in zip(names, S.active) if a]
    print(f"units that ever fire: {len(act)} of {len(names)}")
