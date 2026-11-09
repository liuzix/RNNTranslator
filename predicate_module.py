"""parser_rules.Parser.predicate and copula_predicate translated by hand into rules with complete guards.

`predicate` reads what stands between the subject and the verb — adverbs, a time adverb, the auxiliary or the modal, had
better, need not, have to, the perfect — and hands over to the verb phrase (RET: the verb is the next token, read by the
extracted verb-phrase rules) or, after the copula, to the copula module (RETCOP, a second exit: what follows the module
depends on which it took, and the rules after it must know).  `copula_predicate` reads what follows "be": a
degree adverb, the superlative article, "going to", an adjective and its comparison, a prepositional phrase, a place or
an adverb, a noun phrase; it hands over to the PP machine (RET) or to the verb phrase (the progressive, "going to").
Every branch of the two methods is here in the parser's order, as an ordered decision at one position (pp_module.Decision);
what the frame cannot hold — an infinitive, an embedded clause, the idiom copula, a passive, "not A but B" — exits with
FAIL and says why.

States are module-local; parser_matrix.clause_module_rules maps them to ("M", "pred", name) / ("M", "cop", name).  The
copula module's one parameter, allow_empty (a question word before the copula may leave it empty), arrives as the slot
`param.allow_empty`, written by the entry transition and cleared on every exit.
"""
import rules_ja as R
from pp_module import Decision, lit, AND, OR, NOT, starts_np, adv_write, np_slots, clears

DEGREE_ADVS = ("とても", "本当に", "かなり", "も", "まだ", "もう", "いつも")        # degree adverbs taken whatever follows
PAST_MODALS = ("could", "would", "might", "'d")

def degree_write(src, k, nxt, state, const=None):
    """a degree adverb joins the degree register: degree when empty, else degree2, else refuse"""
    write = lambda slot: (slot, const, ("const",)) if const is not None else (slot, None, ("copy", src))
    return [(state, ("¬slot.degree",), (nxt, k, (write("degree"),))),
            (state, ("slot.degree", "¬slot.degree2"), (nxt, k, (write("degree2"),))),
            (state, ("slot.degree", "slot.degree2"), ("FAIL:three degree adverbs", 0, ()))]

# ------------------------------------------------------------------------------------------------ predicate
def build_pred():
    D = {}; rs = []
    PLAINADV0 = AND(lit("w0.class=adv"), lit("¬w0.class=verb"), lit("¬w0.class=aux_be"))
    ADVNV = {k: AND(lit(f"w{k}.class=adv"), lit(f"¬w{k}.class=verb")) for k in (0, 1, 2)}
    HAVE0 = OR(lit("w0.class=aux_have"), lit("w0.surf='ve"))
    PASTAUX0 = OR(lit("w0.surf=was"), lit("w0.surf=were"))
    PASTMODAL0 = OR(*[lit(f"w0.surf={w}") for w in PAST_MODALS])
    past, perf, had = ("past", True, ("const",)), ("perf", True, ("const",)), ("perf_had", True, ("const",))
    d = Decision("pred", "PR0", D)
    d.branch(PLAINADV0, "PR_ADV")                                                     # adverbs before the verb
    rs += d.otherwise("PR1") + adv_write("adv", 1, "PR0", "PR_ADV")
    d = Decision("pred", "PR1", D)
    d.branch(["w0.class=time", "¬slot.time"], "PR2", 1, [("time", None, ("copy", "time"))])
    d.branch(["w0.class=time", "slot.time"], "FAIL:two time adverbs")
    rs += d.otherwise("PR2")
    d = Decision("pred", "PR2", D)
    d.branch(lit("w0.class=aux_be"), "PR_AUXBE")                                      # the agreement check, then the tense
    d.branch(lit("w0.class=aux_be3"), "PR_BE", 1)
    d.branch(["w0.class=aux_do", "w1.class=not", "w0.surf=did"], "PR_DONOT", 2, [past, ("neg", True, ("const",))])
    d.branch(["w0.class=aux_do", "w1.class=not"], "PR_DONOT", 2, [("neg", True, ("const",))])
    d.branch(["w0.class=aux_do", "w1.class=verb", "w1.vform1=base", "¬w1.class=noun", "w0.surf=did"], "PR_EMPH", 1, [past])   # emphatic: did come
    d.branch(["w0.class=aux_do", "w1.class=verb", "w1.vform1=base", "¬w1.class=noun"], "PR_EMPH", 1)
    d.branch(["w0.surf=may", "w1.surf=as", "w2.surf=well"], "RET", 3, [("want", "better", ("const",))])
    d.branch(["w0.surf=had", "w1.surf=better"], "PR_BETTER", 2, [("want", "better", ("const",))])
    d.branch(["w0.surf='d", "w1.surf=better"], "PR_BETTER", 2, [("want", "better", ("const",))])
    d.branch(["w0.surf=need", "w1.class=not"], "RET", 2, [("want", "neednot", ("const",))])
    d.branch(AND(lit("w0.class=modal"), PASTMODAL0), "PR_MOD1", 1, [("modal", None, ("copy", "modal")), past])
    d.branch(lit("w0.class=modal"), "PR_MOD1", 1, [("modal", None, ("copy", "modal")), ("past", None, ("clear",))])
    d.branch(AND(HAVE0, lit("w1.class=not"), lit("w2.class=verb"), lit("w0.surf=had")), "PR_HAVENOT", 1, [past, perf, had])
    d.branch(AND(HAVE0, lit("w1.class=not"), lit("w2.class=verb")), "PR_HAVENOT", 1, [past, perf])
    d.branch(AND(HAVE0, lit("w1.vform=pp"), lit("¬w1.surf=to"), lit("w0.surf=had")), "PR_HAVENOT", 1, [past, perf, had])
    d.branch(AND(HAVE0, lit("w1.vform=pp"), lit("¬w1.surf=to")), "PR_HAVENOT", 1, [past, perf])
    d.branch(AND(HAVE0, ADVNV[1], lit("w2.vform=pp"), lit("w0.surf=had")), "PR_HAVEADV", 1, [past, perf, had])         # have just finished
    d.branch(AND(HAVE0, ADVNV[1], lit("w2.vform=pp")), "PR_HAVEADV", 1, [past, perf])
    d.branch(AND(HAVE0, ADVNV[1], ADVNV[2], lit("w3.vform=pp"), lit("w0.surf=had")), "PR_HAVEADV", 1, [past, perf, had])
    d.branch(AND(HAVE0, ADVNV[1], ADVNV[2], lit("w3.vform=pp")), "PR_HAVEADV", 1, [past, perf])
    d.branch(["w0.surf=had", "w1.surf=to"], "RET", 2, [past, ("modal", "must", ("const",))])
    d.branch(["w0.surf=have", "w1.surf=to"], "RET", 2, [("modal", "must", ("const",))])
    d.branch(["w0.surf=has", "w1.surf=to"], "RET", 2, [("modal", "must", ("const",))])
    d.branch(lit("w0.class=verb"), "RET")
    rs += d.otherwise("FAIL:predicate")
    D["d.pr.pastaux0"] = PASTAUX0
    rs += [("PR_AUXBE", ("agree.bad",), ("FAIL:agreement", 0, ())),                     # agree.bad: the cell's derived predicate, a literal of the rule itself
           ("PR_AUXBE", ("¬agree.bad", "d.pr.pastaux0"), ("PR_BE", 1, (past,))),
           ("PR_AUXBE", ("¬agree.bad", "¬d.pr.pastaux0"), ("PR_BE", 1, ()))]
    rs += Decision("pred", "PR_BE", D).branch(lit("w0.class=not"), "RETCOP", 1, [("neg", True, ("const",))]).otherwise("RETCOP")
    d = Decision("pred", "PR_DONOT", D); d.branch(ADVNV[0], "PR_DONOT_ADV")
    rs += d.otherwise("RET") + adv_write("adv", 1, "PR_DONOT", "PR_DONOT_ADV")
    rs += adv_write(None, 0, "RET", "PR_EMPH", const="確かに")
    rs += Decision("pred", "PR_BETTER", D).branch(lit("w0.class=not"), "RET", 1, [("neg", True, ("const",))]).otherwise("RET")
    rs += Decision("pred", "PR_MOD1", D).branch(lit("w0.class=not"), "PR_MOD2", 1, [("neg", True, ("const",))]).otherwise("PR_MOD2")
    rs += Decision("pred", "PR_MOD2", D).branch(lit("w0.surf=to"), "PR_MOD3", 1).otherwise("PR_MOD3")
    d = Decision("pred", "PR_MOD3", D); d.branch(PLAINADV0, "PR_MOD3_ADV")
    rs += d.otherwise("PR_MOD4") + adv_write("adv", 1, "PR_MOD3", "PR_MOD3_ADV")
    d = Decision("pred", "PR_MOD4", D)
    d.branch(["slot.modal=would", "w0.surf=mind", "w1.vform1=ing"], "RET", 1, [("want", "mind_request", ("const",)), ("modal", None, ("clear",))])
    d.branch(["w0.surf=have", "w1.vform=pp"], "PR_MODHAVE", 1, [past, perf])
    d.branch(["w0.class=aux_be", "¬w1.class=verb", "¬w1.class=END"], "RETCOP", 1)                  # will be late: the copula follows
    rs += d.otherwise("RET")
    rs += Decision("pred", "PR_MODHAVE", D).branch(["slot.modal=can", "flag.neg"], "RET", 0, [("want", "canthave", ("const",)), ("neg", None, ("clear",)), ("modal", None, ("clear",))]).otherwise("RET")
    rs += Decision("pred", "PR_HAVENOT", D).branch(lit("w0.class=not"), "RET", 1, [("neg", True, ("const",))]).otherwise("RET")
    d = Decision("pred", "PR_HAVEADV", D); d.branch(ADVNV[0], "PR_HAVEADV_ADV")
    rs += d.otherwise("RET") + adv_write("adv", 1, "PR_HAVEADV", "PR_HAVEADV_ADV")
    return rs, D

# ------------------------------------------------------------------------------------------------ copula_predicate
def build_cop():
    D = {}; rs = []
    SNP = {k: starts_np(k) for k in (0, 1, 2, 3)}
    DEGREEADV0 = OR(*([AND(lit("w0.class=adv"), lit(f"w0.adv={a}")) for a in DEGREE_ADVS] +
                      [AND(lit("w0.class=adv"), lit("w1.class=adj"), lit("¬w1.class=noun")), AND(lit("w0.class=adv"), lit("¬w0.class=adj"), lit("¬w1.class=noun"), SNP[1])]))
    VERBISH1 = OR(lit("w1.class=verb"), lit("w1.class=aux_be"), lit("w1.class=modal"), lit("w1.class=aux_do"))
    FLO1 = OR(lit("w1.surf=first"), lit("w1.surf=last"), lit("w1.surf=only"))
    d = Decision("cop", "CP0", D)
    d.branch(AND(lit("flag.imp"), lit("w0.class=adj"), NOT(SNP[0])), "FAIL:imperative adjective")
    d.branch(AND(lit("flag.imp"), lit("hit=adjprep"), NOT(SNP[0])), "FAIL:imperative adjective")
    rs += d.otherwise("CP1")
    d = Decision("cop", "CP1", D)
    d.branch(["w0.class=time", "¬w0.class=adv", "¬slot.time"], "CP1", 1, [("time", None, ("copy", "time"))])
    d.branch(["w0.class=time", "¬w0.class=adv", "slot.time"], "FAIL:two time adverbs")
    rs += d.otherwise("CP2")
    d = Decision("cop", "CP2", D)
    d.branch(AND(DEGREEADV0, lit("w0.surf=too")), "CP2", 1, [("degree", "すぎ", ("const",))])
    d.branch(DEGREEADV0, "CP2_DEG")
    rs += d.otherwise("CP3") + degree_write("adv", 1, "CP2", "CP2_DEG")
    rs += Decision("cop", "CP3", D).branch(["w0.surf=the", "w1.class=adj", "¬w2.class=noun", "¬w2.class=adj"], "CP4", 1).otherwise("CP4")
    rs += Decision("cop", "CP4", D).branch(["w0.surf=no", "w1.surf=less", "w2.class=adj", "w3.surf=than"], "CP5", 2, [("noless", True, ("const",))]).otherwise("CP5")
    d = Decision("cop", "CP5", D)
    d.branch(["w0.surf=going", "w1.surf=to", "w2.class=verb"], "RET", 2, [("want", "intend", ("const",))])
    d.branch(["w0.surf=to", "w1.class=verb", "w1.vform1=base"], "RET", 1, [("want", "be_to", ("const",))])
    d.branch(["w0.surf=about", "w1.surf=to", "w2.class=verb", "w2.vform1=base"], "RET", 2, [("want", "about_to", ("const",))])
    d.branch(AND(lit("w0.surf=as"), lit("w1.class=adj"), lit("w2.surf=as"), SNP[3]), "CP_AS1", 1)
    d.branch(["w0.surf=more", "w1.class=adj"], "CP5_MORE")
    rs += d.otherwise("CP6") + degree_write(None, 1, "CP6", "CP5_MORE", const="より")
    rs += [("CP_AS1", ("w0.class=adj",), ("CP_AS2", 1, (("cop_kind", "adj", ("const",)), ("cop_adj", None, ("copy", "adj_pair"))))),
           ("CP_AS1", ("¬w0.class=adj",), ("FAIL:adjective expected", 0, ())),
           ("CP_AS2", ("w0.surf=as",), (("NP0", "cmp"), 1, (("ret", "cop_cmp", ("const",)), ("cmp_kind", "as", ("const",))) + tuple(clears("cmp")))),
           ("CP_AS2", ("¬w0.surf=as",), ("FAIL:as expected", 0, ()))]
    d = Decision("cop", "CP6", D)
    d.branch(["w0.class=verb", "w0.vform1=ing"], "RET", 0, [("prog", True, ("const",))])
    d.branch(AND(lit("hit=adjprep"), OR(SNP[2], AND(lit("w2.class=verb"), lit("w2.vform1=ing")))), "FAIL:idiom copula")
    d.branch(AND(lit("w0.class=verb"), lit("w0.vform1=pp"), NOT(SNP[0])), "FAIL:passive")
    d.branch(["w0.class=adj", "¬w1.class=noun"], "CP_ADJ", 1, [("cop_kind", "adj", ("const",)), ("cop_adj", None, ("copy", "adj_pair"))])
    d.branch(lit("w0.class=prep"), "CP_PREP")
    d.branch(lit("w0.class=loc"), "RET", 1, [("cop_kind", "pp", ("const",)), ("cop_part", R.PREPS["at"], ("const",)), ("cop.noun", None, ("copy", "loc")), ("cop.cls", "t", ("const",))])
    d.branch(lit("w0.class=adv"), "RET", 1, [("cop_kind", "adv", ("const",)), ("cop_adv", None, ("copy", "adv"))])
    d.branch(["w0.class=end", "slot.param.allow_empty"], "RET")
    d.branch(AND(lit("w0.surf=the"), FLO1, lit("w2.surf=to"), lit("w3.class=verb")), "FAIL:infinitive")
    d.branch(SNP[0], ("NP0", "cop"), 0, [("ret", "cop_np", ("const",))] + clears("cop"))
    rs += d.otherwise("FAIL:copula predicate")
    for w in sorted(R.PREPS):                                                            # the copula's PP: the preposition, then the noun phrase
        rs.append(("CP_PREP", (f"w0.surf={w}",), (("NP0", "cop"), 1, (("ret", "cop_pp", ("const",)), ("cop_part", R.PREPS[w], ("const",))) + tuple(clears("cop")))))
    rs.append(("CP_PREP", tuple(f"¬w0.surf={w}" for w in sorted(R.PREPS)), ("FAIL:preposition expected", 0, ())))     # the table's otherwise
    # after the adjective: than, enough to, too … to, to V, of X to V, for X to V, that S
    d = Decision("cop", "CP_ADJ", D)
    d.branch(["w0.surf=than", "w1.class=time", "flag.noless"], "CP_THAN_T", 1, [("cmp_kind", "noless", ("const",))] + clears("cmp"))
    d.branch(["w0.surf=than", "w1.class=time"], "CP_THAN_T", 1, [("cmp_kind", "than", ("const",))] + clears("cmp"))
    d.branch(AND(lit("w0.surf=than"), SNP[1], lit("flag.noless")), ("NP0", "cmp"), 1, [("ret", "cop_cmp", ("const",)), ("cmp_kind", "noless", ("const",))] + clears("cmp"))
    d.branch(AND(lit("w0.surf=than"), SNP[1]), ("NP0", "cmp"), 1, [("ret", "cop_cmp", ("const",)), ("cmp_kind", "than", ("const",))] + clears("cmp"))
    d.branch(["w0.surf=enough", "w1.surf=to", "w2.class=verb"], "FAIL:infinitive")
    d.branch(["slot.degree=すぎ", "w0.surf=to", "w1.class=verb"], "FAIL:infinitive")
    d.branch(["w0.surf=to", "w1.class=verb"], "FAIL:infinitive")
    d.branch(AND(lit("w0.surf=of"), SNP[1], lit("subj.sore")), "FAIL:infinitive")
    d.branch(AND(lit("w0.surf=for"), SNP[1], lit("subj.sore")), "FAIL:infinitive")
    d.branch(lit("w0.surf=that"), "FAIL:embedded clause")
    d.branch(AND(SNP[0], VERBISH1), "FAIL:embedded clause")
    rs += d.otherwise("RET")
    rs += [("CP_THAN_T", ("w0.class=time",), ("RET", 1, (("cmp.noun", None, ("copy", "time")), ("cmp.cls", "t", ("const",))))),
           ("CP_THAN_T", ("¬w0.class=time",), ("FAIL:time adverb expected", 0, ()))]
    # the noun phrases come back
    rs.append(((("RET", "cmp")), ("reg.ret=cop_cmp",), ("RET", 0, (("ret", None, ("clear",)),))))
    rs.append(((("RET", "cop")), ("reg.ret=cop_pp",), ("RET", 0, (("ret", None, ("clear",)), ("cop_kind", "pp", ("const",))))))
    rs.append(((("RET", "cop")), ("reg.ret=cop_np",), ("CP_NP", 0, (("ret", None, ("clear",)), ("cop_kind", "np", ("const",))))))
    d = Decision("cop", "CP_NP", D)
    d.branch(AND(lit("flag.neg"), lit("w0.surf=but"), SNP[1]), "FAIL:not A but B")
    d.branch(["w0.surf=to", "w1.class=verb", "w1.vform1=base"], "FAIL:infinitive")
    d.branch(AND(lit("subj.sore"), lit("w0.surf=that"), SNP[1]), "FAIL:embedded clause")
    rs += d.otherwise("RET")
    # every exit clears the parameter slot the entry wrote
    rs = [(st, lits, ("CP_RET", k, acts) if nxt == "RET" else (nxt, k, acts)) for st, lits, (nxt, k, acts) in rs]
    rs.append(("CP_RET", (), ("RET", 0, (("param.allow_empty", None, ("clear",)),))))
    return rs, D

_pred = _cop = None
def pred_rules():
    global _pred
    if _pred is None: _pred = build_pred()
    return list(_pred[0])
def pred_derived(): pred_rules(); return dict(_pred[1])
def cop_rules():
    global _cop
    if _cop is None: _cop = build_cop()
    return list(_cop[0])
def cop_derived(): cop_rules(); return dict(_cop[1])
