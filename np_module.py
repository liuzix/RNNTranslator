"""parse_np translated by hand into a sub-automaton with complete guards.

The module reads the 4-token window and a few frame registers, writes the sub-slots of one noun-phrase role
(det, poss, num, adj1, adj2, noun, noun2, cls, mono, compound; number for a subject) and exits by RET or FAIL:<reason>.  Every branch of
parser_rules.Parser.parse_np is here, including its refusals; what the matrix cannot hold yet (an inner clause,
an of-phrase, a coordination, a possessive noun phrase) exits with FAIL and says why.

Two kinds of definitions:
  DERIVED  named predicates defined as a disjunction of conjunctions of base literals (compiled as AND units then an OR unit)
  rules    (state, literals, (next_state, tokens consumed, actions)); literals are predicate names, "¬" negates;
           a state with k=0 transitions is a sequential `if`: the chain NP4 -> NP5 -> NP6 -> NP7 is parse_np's post-head checks.
Base predicate names are parser_matrix.predicates(): w{k}.class=…, w{k}.surf=…, w{k}.prole=…, slot.<slot>, slot.<slot>=<value>,
reg.verb, reg.cop_kind, reg.subj, subj.sore, pend_prep=<w>.
"""
PERSONAL = ("it", "he", "she", "i", "you", "we", "they", "this", "that", "who", "what", "somebody", "someone", "everybody", "everyone", "nobody", "anybody", "anyone", "one", "there", "here")
CLOSED = ("aux_be", "aux_do", "aux_have", "modal", "prep", "adv", "det", "pron", "adj", "time", "conj", "num")
QUANTIFIERS = ("such", "all", "both")                                   # determiners that may precede an article (parser_rules.QUANTIFIERS)
INDEFINITES = ("something", "anything", "nothing", "everything")        # pronouns an adjective may follow (parser_rules.INDEFINITES)
TIME_PREPS = ("before", "after", "at", "until", "till", "around", "by")
VERBAL = ("verb", "aux_be", "aux_have", "modal")

def derived(role):
    """DERIVED[name] = list of conjunctions (lists of literals): the name is true when any conjunction is"""
    D = {}
    D["d.verbal1"] = [[f"w1.class={c}"] for c in VERBAL]
    D["d.aux1"] = [[f"w1.class={c}"] for c in ("aux_be", "aux_do", "aux_have", "modal")]
    D["d.relpat"] = [[f"w0.surf={w}", f"w1.class={c}"] for w in ("who", "which", "that") for c in VERBAL] + \
                    [["w0.class=pron", f"w0.prole={r}", f"w1.class={c}"] for r in ("s", "so") for c in VERBAL]
    D["d.relok"] = [["reg.verb"], ["reg.cop_kind"]] + ([["reg.subj"]] if role != "subj" else [])
    D["d.end1"] = [["w1.class=END"], ["w1.class=end"], ["w1.class=prep"], ["w1.surf=,"]]
    D["d.nominal1"] = [["w1.class=noun"], ["w1.class=adj", "w2.class=noun"], ["w1.class=adj", "w2.class=adj"], ["w1.class=num"], ["w1.class=adv", "w2.class=adj", "w3.class=noun"]]
    D["d.startsnp1"] = [["w1.class=pron"], ["w1.class=det"], ["w1.class=noun"], ["w1.class=num"], ["w1.class=adj", "¬w1.class=verb", "¬w1.class=adv", "w2.class=noun", "¬w2.class=verb"]]
    no_closed = [f"¬w0.class={c}" for c in CLOSED]
    D["d.compound0"] = [["w0.class=noun", "¬w0.class=verb"] + no_closed] + [["w0.class=noun", "w0.class=verb", f"w1.class={c}"] + no_closed for c in ("aux_be", "aux_do", "aux_have", "modal")]
    D["d.extraposed"] = [["subj.sore", "¬reg.verb", "¬reg.cop_kind"]]
    D["d.pend_time"] = [[f"pend_prep={w}"] for w in TIME_PREPS]
    D["d.personal0"] = [[f"w0.surf={w}"] for w in PERSONAL]
    D["d.posspron0"] = [["w0.class=pron", "w0.prole=p"], ["w0.class=pron", "w0.prole=op"]]
    D["d.quantart"] = [[f"w0.surf={q}", f"w1.surf={a}"] for q in QUANTIFIERS for a in ("a", "an", "the")]            # such a / all the: the article is dropped
    D["d.indefadj"] = [[f"w0.surf={w}", "w1.class=adj", "¬w1.class=noun", "¬w1.class=verb"] for w in INDEFINITES]   # something new -> 何か新しいもの
    D["d.lotof"] = [["w0.surf=a lot", "w1.surf=of"], ["w0.surf=lots", "w1.surf=of"], ["w0.surf=a", "w1.surf=lot", "w2.surf=of"]]
    D["d.one_pronoun"] = [["w0.surf=one", "slot." + role + ".det", "¬w1.class=noun"], ["w0.surf=one", "slot." + role + ".poss", "¬w1.class=noun"]]
    return D

def R(state, lits, nxt, k=1, acts=()):
    return (state, tuple(lits), (nxt, k, tuple(acts)))

def rules_for(role):
    s = lambda sub: f"{role}.{sub}"
    det_copy = (s("det"), None, ("copy", "det")); poss_copy = (s("poss"), None, ("copy", "pron_poss"))
    number = [(s("number"), None, ("copy", "nnum"))] if role == "subj" else []          # the head's number, for agreement (subjects only)
    pron_head = [(s("noun"), None, ("copy", "pron")), (s("cls"), "h", ("const",))] + number
    rules = []
    # NP0 — determiner / possessive / pronoun head
    rules += [R("NP0", ["¬slot." + s("det"), "¬slot." + s("poss"), "w0.surf=a lot", "w1.surf=of"], "NP1", 2, [(s("det"), "たくさんの", ("const",))]),
              R("NP0", ["¬slot." + s("det"), "¬slot." + s("poss"), "w0.surf=lots", "w1.surf=of"], "NP1", 2, [(s("det"), "たくさんの", ("const",))]),
              R("NP0", ["¬slot." + s("det"), "¬slot." + s("poss"), "w0.surf=a", "w1.surf=lot", "w2.surf=of"], "NP1", 3, [(s("det"), "たくさんの", ("const",))]),
              R("NP0", ["¬slot." + s("det"), "¬slot." + s("poss"), "¬d.lotof", "d.posspron0", "d.nominal1"], "NP1", 1, [poss_copy]),
              R("NP0", ["¬d.lotof", "w0.class=pron", "w0.prole=p", "¬d.nominal1"], "FAIL:possessive pronoun without noun", 0),
              R("NP0", ["¬slot." + s("det"), "¬slot." + s("poss"), "¬d.lotof", "¬d.posspron0", "w0.class=det", "¬w0.class=pron", "¬w0.surf=no", "¬d.quantart"], "NP1", 1, [det_copy]),
              R("NP0", ["¬slot." + s("det"), "¬slot." + s("poss"), "¬d.lotof", "¬d.posspron0", "w0.class=det", "¬w0.class=pron", "¬w0.surf=no", "d.quantart"], "NP1", 2, [det_copy]),
              R("NP0", ["¬slot." + s("det"), "¬slot." + s("poss"), "¬d.lotof", "¬d.posspron0", "w0.class=det", "¬w0.class=pron", "w0.surf=no"], "NP1", 1, [det_copy, ("neg", True, ("const",))]),
              R("NP0", ["¬slot." + s("det"), "¬slot." + s("poss"), "¬d.lotof", "¬d.posspron0", "w0.class=det", "w0.class=pron", "d.nominal1"], "NP1", 1, [det_copy])]
    # pronoun head: a non-possessive pronoun (or an 'op' pronoun / a det-pronoun) with nothing nominal after it
    head_ctx = [["¬d.lotof", "w0.class=pron", "¬w0.prole=p", "¬w0.prole=op", "¬w0.class=det"],
                ["¬d.lotof", "w0.class=pron", "¬w0.prole=p", "¬w0.prole=op", "w0.class=det", "¬d.nominal1"],
                ["¬d.lotof", "w0.class=pron", "w0.prole=op", "¬d.nominal1"]]
    if role == "subj":
        # subject case: an object-only or reflexive pronoun cannot be the subject (*me went, *himself saw john)
        for hc in head_ctx[:2]:
            for pr in ("o", "r"): rules.append(R("NP0", hc + [f"w0.prole={pr}"], "FAIL:subject case", 0))
        rules.append(R("NP0", head_ctx[2], "FAIL:subject case", 0))
        head_ctx = [hc + ["¬w0.prole=o", "¬w0.prole=r"] for hc in head_ctx[:2]]
    for hc in head_ctx:
        rules += [R("NP0", hc + ["¬w1.surf='s", "¬d.indefadj"], "RET", 1, pron_head),
                  R("NP0", hc + ["¬w1.surf='s", "d.indefadj"], "NPM", 1, pron_head),
                  R("NP0", hc + ["w1.surf='s", "d.personal0"], "RET", 1, pron_head),
                  R("NP0", hc + ["w1.surf='s", "¬d.personal0", "w2.class=noun"], "NP0", 2, [poss_copy]),
                  R("NP0", hc + ["w1.surf='s", "¬d.personal0", "¬w2.class=noun", "w2.class=adj"], "NP0", 2, [poss_copy]),
                  R("NP0", hc + ["w1.surf='s", "¬d.personal0", "¬w2.class=noun", "¬w2.class=adj"], "RET", 2,
                    [(s("noun"), None, ("copy", "pron")), (s("cls"), "t", ("const",)), (s("mono"), "のもの", ("const",))])]
    rules += [
              R("NP0", ["¬d.lotof", "¬w0.class=det", "¬w0.class=pron"], "NP1", 0)]
    # NPM — the adjective after an indefinite pronoun: something new -> 何か新しいもの
    rules += [R("NPM", ["w0.class=adj"], "RET", 1, [(s("mono"), None, ("copy", "adj_mono"))])]
    # a second determiner or possessive after "name 's": the rule parser concatenates, the slots cannot
    for filled in (["slot." + s("det")], ["¬slot." + s("det"), "slot." + s("poss")]):
        rules += [R("NP0", filled + ["d.lotof"], "FAIL:possessive noun phrase", 0),
                  R("NP0", filled + ["¬d.lotof", "d.posspron0", "d.nominal1"], "FAIL:possessive noun phrase", 0),
                  R("NP0", filled + ["¬d.lotof", "¬d.posspron0", "w0.class=det", "¬w0.class=pron"], "FAIL:possessive noun phrase", 0),
                  R("NP0", filled + ["¬d.lotof", "¬d.posspron0", "w0.class=det", "w0.class=pron", "d.nominal1"], "FAIL:possessive noun phrase", 0)]
    # NP1 — number
    rules += [R("NP1", ["w0.class=num", "¬d.one_pronoun"], "NP2", 1, [(s("num"), None, ("copy", "num"))]),
              R("NP1", ["w0.class=num", "d.one_pronoun"], "NP2", 0),
              R("NP1", ["¬w0.class=num"], "NP2", 0)]
    # NP2 — adjectives: adverb before an adjective, adjective before a noun or an adjective
    adj_trigger = [["w0.class=adv", "w1.class=adj"], ["¬w0.class=adv", "w0.class=adj", "w1.class=noun"], ["¬w0.class=adv", "w0.class=adj", "¬w1.class=noun", "w1.class=adj"],
                   ["w0.class=adv", "¬w1.class=adj", "w0.class=adj", "w1.class=noun"]]
    for trig in adj_trigger:
        src = ("copy", "adv") if trig[0] == "w0.class=adv" and trig[1] == "w1.class=adj" else ("copy", "adj")
        rules += [R("NP2", trig + ["¬slot." + s("adj1")], "NP2", 1, [(s("adj1"), None, src)]),
                  R("NP2", trig + ["slot." + s("adj1"), "¬slot." + s("adj2")], "NP2", 1, [(s("adj2"), None, src)]),
                  R("NP2", trig + ["slot." + s("adj1"), "slot." + s("adj2")], "FAIL:three adjectives", 0)]
    rules += [R("NP2", ["¬w0.class=adv", "¬w0.class=adj"], "NP3", 0),
              R("NP2", ["w0.class=adv", "¬w1.class=adj", "¬w0.class=adj"], "NP3", 0),
              R("NP2", ["w0.class=adv", "¬w1.class=adj", "w0.class=adj", "¬w1.class=noun"], "NP3", 0),
              R("NP2", ["¬w0.class=adv", "w0.class=adj", "¬w1.class=noun", "¬w1.class=adj"], "NP3", 0)]
    # NP3 — the head: a noun, or a bare number
    bare = ["¬w0.class=noun", "slot." + s("num"), "¬slot." + s("adj1"), "¬slot." + s("det"), "¬slot." + s("poss")]
    head = [(s("noun"), None, ("copy", "noun")), (s("cls"), None, ("copy", "ncls"))]
    measure = [(s("number"), "none", ("const",))] if role == "subj" else []        # ten years is a long time: a measure phrase takes either number
    rules += [R("NP3", ["w0.class=noun", "¬slot." + s("num")], "NP4", 1, head + number),
              R("NP3", ["w0.class=noun", "slot." + s("num")], "NP4", 1, head + measure),
              R("NP3", bare + ["d.pend_time"], "RET", 0, [(s("noun"), None, ("move_time", s("num"))), (s("num"), None, ("clear",)), (s("cls"), "t", ("const",))]),
              R("NP3", bare + ["¬d.pend_time"], "RET", 0, [(s("noun"), None, ("move", s("num"))), (s("num"), None, ("clear",)), (s("cls"), "t", ("const",))]),
              R("NP3", ["¬w0.class=noun", "¬slot." + s("num")], "FAIL:noun phrase", 0),
              R("NP3", ["¬w0.class=noun", "slot." + s("num"), "slot." + s("adj1")], "FAIL:noun phrase", 0),
              R("NP3", ["¬w0.class=noun", "slot." + s("num"), "¬slot." + s("adj1"), "slot." + s("det")], "FAIL:noun phrase", 0),
              R("NP3", ["¬w0.class=noun", "slot." + s("num"), "¬slot." + s("adj1"), "¬slot." + s("det"), "slot." + s("poss")], "FAIL:noun phrase", 0)]
    # NP4 — compound
    rules += [R("NP4", ["d.compound0", "¬slot." + s("noun2")], "NP4", 1, [(s("noun2"), None, ("copy", "noun")), (s("cls"), None, ("copy", "ncls")), (s("compound"), True, ("const",))] + number),
              R("NP4", ["d.compound0", "slot." + s("noun2")], "FAIL:three nouns", 0),
              R("NP4", ["¬d.compound0"], "NP5", 0)]
    # NP5 — relative clause (an inner frame: not representable here)
    rules += [R("NP5", ["d.relpat", "d.relok"], "FAIL:relative clause", 0),
              R("NP5", ["d.relpat", "¬d.relok"], "NP6", 0),
              R("NP5", ["¬d.relpat"], "NP6", 0)]
    # NP6 — 's
    rules += [R("NP6", ["w0.surf='s", "d.end1"], "RET", 1, [(s("mono"), "のもの", ("const",)), (s("cls"), "t", ("const",))]),
              R("NP6", ["w0.surf='s", "¬d.end1"], "FAIL:possessive noun phrase", 0),
              R("NP6", ["¬w0.surf='s"], "NP7", 0)]
    # NP7 — the fact that S, of-phrase, coordination, return
    fact = ["slot." + s("noun") + "=事実", "w0.surf=that", "d.startsnp1"]
    rules += [R("NP7", fact + ["¬d.extraposed"], "FAIL:the fact that", 0),
              R("NP7", fact + ["d.extraposed"], "RET", 0),
              R("NP7", ["¬slot." + s("noun") + "=事実", "w0.surf=of", "¬w1.class=END", "¬w1.surf=course"], "FAIL:of-phrase", 0),
              R("NP7", ["slot." + s("noun") + "=事実", "¬w0.surf=that", "w0.surf=of", "¬w1.class=END", "¬w1.surf=course"], "FAIL:of-phrase", 0),
              R("NP7", ["w0.surf=and", "d.startsnp1", "¬w1.class=verb"], "FAIL:coordination", 0),
              R("NP7", ["¬w0.surf=that", "¬w0.surf=of", "¬w0.surf=and"], "RET", 0),
              R("NP7", ["w0.surf=that", "¬slot." + s("noun") + "=事実"], "RET", 0),
              R("NP7", ["w0.surf=that", "slot." + s("noun") + "=事実", "¬d.startsnp1"], "RET", 0),
              R("NP7", ["w0.surf=of", "w1.class=END"], "RET", 0),
              R("NP7", ["w0.surf=of", "¬w1.class=END", "w1.surf=course"], "RET", 0),
              R("NP7", ["w0.surf=and", "¬d.startsnp1"], "RET", 0),
              R("NP7", ["w0.surf=and", "d.startsnp1", "w1.class=verb"], "RET", 0)]
    return rules

# ------------------------------------------------------------------ simulator (the reference for the exhaustive check and for the hybrid)
def eval_lit(lit, P, D):
    neg = lit.startswith("¬"); name = lit[1:] if neg else lit
    if name.startswith("d."):
        v = any(all(eval_lit(l, P, D) for l in conj) for conj in D[name])
    else:
        v = name in P
    return (not v) if neg else v

def step(rules, D, state, P):
    """the unique rule whose literals hold, or None / 'ambiguous'"""
    hits = [r for r in rules if r[0] == state and all(eval_lit(l, P, D) for l in r[1])]
    if len(hits) == 1: return hits[0][2]
    if not hits: return None
    return ("AMBIGUOUS", hits)

# ------------------------------------------------------------------ predicates for the module (base names shared with parser_matrix)
import rules_ja as RJ
from parser_matrix import window_features, readings_of, IDENTITY_WORDS, OPEN, CLOSED_CLASSES

def np_predicates(toks, i, slots, regs, pend):
    """base predicates at position i: window (4 slots), frame slots, a few registers, the pending preposition"""
    P = set()
    for k in range(4):
        t = toks[i + k] if i + k < len(toks) else None
        rs = readings_of(t); classes = sorted(set(c for c, _ in rs))
        for c in classes: P.add(f"w{k}.class={c}")
        if t is None or not (set(classes) & OPEN) or (set(classes) & CLOSED_CLASSES) or t in IDENTITY_WORDS: P.add(f"w{k}.surf={t}")
        for c, v in rs:
            if c == "pron": P.add(f"w{k}.prole={v[1]}")
    for slot, v in slots.items():
        if v not in (None, "", False): P.add(f"slot.{slot}"); P.add(f"slot.{slot}={v}")
    if regs.get("verb"): P.add("reg.verb")
    if regs.get("cop_kind"): P.add("reg.cop_kind")
    if regs.get("subj"): P.add("reg.subj")
    if regs.get("subj") == "それ": P.add("subj.sore")
    if pend: P.add(f"pend_prep={pend}")
    return P

def apply(slots, actions, toks, i, flags):
    tok = toks[i] if i < len(toks) else None
    rs = {}
    for c, v in readings_of(tok): rs.setdefault(c, []).append(v)
    for slot, v, src in actions:
        kind = src[0]
        if kind == "const":
            if slot == "neg": flags["neg"] = True
            else: slots[slot] = v
        elif kind == "clear": slots.pop(slot, None)
        elif kind == "move": slots[slot] = slots.get(src[1])
        elif kind == "move_time": slots[slot] = slots.get(src[1]) + "時"
        elif kind == "copy":
            c = src[1]
            if c == "det": slots[slot] = rs["det"][0]
            elif c == "pron": slots[slot] = rs["pron"][0][0]
            elif c == "pron_poss": slots[slot] = rs["pron"][0][0] if rs["pron"][0][1] == "p" else rs["pron"][0][0] + "の"
            elif c == "num": slots[slot] = rs["num"][0]
            elif c == "adj": ja, kind2 = rs["adj"][-1]; slots[slot] = ja + ("な" if kind2 == "na" and not ja.endswith("な") else "")
            elif c == "adv": slots[slot] = rs["adv"][-1]
            elif c == "noun": slots[slot] = rs["noun"][-1][0]
            elif c == "ncls": slots[slot] = rs["noun"][-1][1]
            elif c == "nnum": slots[slot] = RJ.number_of(tok) or "none"

def run_module(role, toks, i0, regs, pend, rules=None, D=None):
    """run the module from position i0; returns ("RET", i, slots, flags) or ("FAIL", reason, i)"""
    rules = rules or rules_for(role); D = D or derived(role)
    state, i, slots, flags = "NP0", i0, {}, {}
    for _ in range(40):
        P = np_predicates(toks, i, slots, regs, pend)
        out = step(rules, D, state, P)
        if out is None: return ("FAIL", f"dead at {state}", i)
        if out[0] == "AMBIGUOUS": return ("FAIL", f"ambiguous at {state}: " + "; ".join(str(r[1]) for r in out[1][:2]), i)
        nxt, k, actions = out
        apply(slots, actions, toks, i, flags); i += k
        if nxt == "RET": return ("RET", i, slots, flags)
        if nxt.startswith("FAIL:"): return ("FAIL", nxt[5:], i)
        state = nxt
    return ("FAIL", "loop", i)

def np_string(slots, role=None):
    g = (lambda k: slots.get(f"{role}.{k}")) if role else (lambda k: slots.get(k))
    s = "".join(g(k) or "" for k in ("det", "poss", "num", "adj1", "adj2", "noun"))
    if g("noun2"): s += "の" + g("noun2")
    if g("mono"): s += g("mono")
    return s
