"""The printing rules (gen_module) against generate(): on every sentence the rule parser parses, and on a sweep over the
predicates the rules test.

Corpus: for each parsed sentence, the rules are stepped on the frame and their tokens joined; the item `printer <split>
<citation>` is AGREE when the string is generate()'s, GAP when the rules refuse (a register the frame cannot hold, or no
rule), MISFIRE when they print something else.  Sweep: frames are built from a product of the predicates the rules read
(the kind of predicate, the flags, the modal, the complement; the arguments, the particles, the question words) and
rendered by both; a frame generate() itself cannot print (it raises) is skipped; the rules may refuse a frame, but where
they print they must print what generate() prints — the mismatch count is the item `printer.sweep` (MISMATCHES < MATCH).
"""
import sys, os, json, argparse, collections, itertools, time
sys.path.insert(0, os.getcwd())
import rules_ja as R, run_rules as H, gen_module as G
from parser_rules import ParseFail, Clause, parse_bounded

def corpus(pr):
    out = {}; per = {}
    for split in ("dev", "test"):
        outputs = {}; cnt = collections.Counter(); reasons = collections.Counter(); examples = []
        for line, en, ja in H.load_indexed(split):
            toks = R.merge_multiword(en); key = f"{line}: {' '.join(en)}"
            try: frame = parse_bounded(toks)
            except ParseFail: continue
            try: got = "".join(pr.render(frame))
            except G.Refuse as e: got = ""; reasons[str(e)[:40]] += 1
            want = R.generate(frame)
            outputs[key] = got
            level = "AGREE" if got == want else ("GAP" if not got else "MISFIRE"); cnt[level] += 1
            if level == "MISFIRE" and len(examples) < 12: examples.append({"sentence": " ".join(toks), "rules": got, "generate": want})
        out[split] = outputs; per[split] = {"agree": cnt["AGREE"], "gap": cnt["GAP"], "misfire": cnt["MISFIRE"], "refusals": dict(reasons.most_common(12)), "misfire_examples": examples}
    return out, per

# ------------------------------------------------------------------------------------------------ the sweep
def verb_examples():
    """one verb value per conjugation class, the special lemmas, a use-verb, the fixed verb"""
    vals = []
    for lemma in ("eat", "go", "study", "come", "like", "want", "need", "understand", "have", "intend"):
        for e in R.LEX.get(lemma, []):
            if e[0] == "v": vals.append((lemma, e[1], e[2])); break
    vals.append(("take+rest", "休む", "g"))
    return vals

def base_clause():
    c = Clause(); c.subj = "彼"; c.subj_cls = "h"; return c

def predicate_frames():
    """(label, Clause) for the predicate sweep: kinds x flags x modal, and kinds x fewer flags x want"""
    kinds = [("verb", v) for v in verb_examples()]
    kinds += [("adj", ("忙しい", "a")), ("adj", ("貴重", "na")), ("adj", ("疲れている", "fx")), ("adj", ("違う", "fx")), ("adj", ("裸の", "a")), ("adj", ("より良い", "a")),
              ("np", ("先生", "h")), ("pp", (("で", "に"), "家")), ("adv", "下に"), ("wh", ("何", "o")), ("there", None)]
    flags_all = ("past", "neg", "prog", "perf", "perf_had", "question", "imp", "please", "lets", "tag")
    flags_few = ("past", "neg", "question", "perf", "imp")
    wants = [None] + sorted(R.COMPLEMENTS) + ["tai", "wish", "better", "neednot", "mind_request", "request", "canthave"]
    def make(kind, val, flags, modal, want, animate):
        c = base_clause()
        if not animate: c.subj = "それ"; c.subj_cls = "t"
        if kind == "verb": c.verb = val; c.obj = "本" if val[0] in ("want", "eat") else None
        elif kind == "adj": c.copula = ("adj", val)
        elif kind == "np": c.copula = ("np", val[0], val[1])
        elif kind == "pp": c.copula = ("pp", val[0], val[1])
        elif kind == "adv": c.copula = ("adv", val)
        elif kind == "wh": c.copula = ("wh",); c.wh = val; c.question = True
        elif kind == "there": c.there = True
        for f in flags: setattr(c, f, True)
        c.modal = modal; c.want = want
        if want in ("promise", "try", "decide", "refuse", "agree", "pretend", "mean", "choose", "prefer") and "neg" in flags: c.neg_inner = True
        return c
    for kind, val in kinds:
        for animate in ((True, False) if kind in ("pp", "adv", "there") else (True,)):
            for n in range(len(flags_all) + 1):
                pass
            for bits in itertools.product((False, True), repeat=len(flags_all)):
                flags = [f for f, b in zip(flags_all, bits) if b]
                if "perf_had" in flags and "perf" not in flags: continue
                for modal in (None, "will", "can", "should", "must", "may"):
                    yield (kind, str(val), flags, modal, None), make(kind, val, flags, modal, None, animate)
            if kind != "verb": continue
            for bits in itertools.product((False, True), repeat=len(flags_few)):
                flags = [f for f, b in zip(flags_few, bits) if b]
                for want in wants:
                    if want is None: continue
                    yield (kind, str(val), flags, None, want), make(kind, val, flags, None, want, animate)

def template_frames():
    """the arguments and their particles, the question words, the comparison, around a fixed predicate"""
    verbs = [("eat", "食べる", "i"), ("like", "好む", "g"), ("enter", "入る", "g"), ("go", "行く", "g"), ("rain", "雨が降る", "g"), None]
    subjs = [None, ("彼", "h"), ("私", "s"), ("それ", "t"), ("誰も", "h"), ("私の母", "h")]
    objs = [None, "本", "私の本", "何も", "彼の本"]
    parts = [None, "", "を", "に", "も"]
    for verb in verbs:
        for subj in subjs:
            for obj in objs:
                for part in parts:
                    for iobj in (None, "彼女"):
                        for flags in ((), ("neg",), ("question",), ("imp",), ("neg", "question")):
                            c = Clause(); c.verb = verb; c.obj = obj; c.obj_particle = part; c.iobj = iobj
                            if subj: c.subj, c.subj_cls = subj
                            for f in flags: setattr(c, f, True)
                            if verb is None: c.copula = ("adj", ("忙しい", "a"))
                            yield ("args", str(verb), subj, obj, part, iobj, flags), c
    whs = [None, ("何", "o"), ("誰", "s"), ("どこ", "loc"), ("いつ", "t"), ("なぜ", "t"), ("何時", "t"), ("どう", "adv"), ("どのくらい", "adv"), ("いくつ", "count"), ("どこから", "pp"), ("一体何", "s")]
    cops = [None, ("adj", ("忙しい", "a")), ("np", "先生", "h"), ("pp", ("で", "に"), "家"), ("adv", "下に"), ("wh",)]
    for wh in whs:
        for cop in cops:
            for verb in (None, ("go", "行く", "g"), ("stay", "滞在する", "s")):
                if cop and verb and cop[0] != "adj": continue
                for subj in (None, ("彼", "h"), ("それ", "t")):
                    for there in (False, True):
                        for flags in ((), ("past",), ("neg",), ("question", "past")):
                            c = Clause(); c.wh = wh; c.copula = cop; c.verb = verb; c.there = there
                            if subj: c.subj, c.subj_cls = subj
                            if wh: c.question = True
                            for f in flags: setattr(c, f, True)
                            yield ("wh", wh, cop, str(verb), subj, there, flags), c
    pps = [[], [(("で", "に"), "学校")], [(("で", "に"), "学校"), (("と", "と"), "友達")], [(("で", "に"), "学校"), (("と", "と"), "友達"), (("から", "から"), "駅")]]
    cmps = [None, ("than", "彼"), ("as", "私の母"), ("noless", "彼女"), ("as", "彼の母")]
    for pp in pps:
        for cmp in cmps:
            for verb in [("go", "行く", "g"), ("eat", "食べる", "i"), None]:
                for cop in ([None] if verb else [("adj", ("より良い", "a")), ("adj", ("忙しい", "a")), ("pp", ("で", "に"), "家"), ("np", "先生", "h")]):
                    for subj in (("私", "s"), ("彼", "h"), ("それ", "t")):
                        for there in ((False, True) if verb is None and cop is None else (False,)):
                            for extra in ((), ("neg",), ("time",), ("adv",), ("never",), ("front",), ("degree",), ("tag",)):
                                c = Clause(); c.pp = list(pp); c.compare = cmp; c.verb = verb; c.copula = cop; c.there = there
                                c.subj, c.subj_cls = subj
                                if "neg" in extra: c.neg = True
                                if "time" in extra: c.time = "昨日"
                                if "adv" in extra: c.adv = "すぐに"
                                if "never" in extra: c.adv = "決して"
                                if "front" in extra: c.front = "実は、"
                                if "degree" in extra: c.degree = "とても"
                                if "tag" in extra: c.tag = True
                                yield ("pp", len(pp), cmp, str(verb), str(cop), subj, there, extra), c

def sweep(pr):
    t0 = time.time(); cnt = collections.Counter(); examples = {}; per_group = collections.defaultdict(collections.Counter)
    for label, c in itertools.chain(predicate_frames(), template_frames()):
        group = label[0]
        try: got = "".join(pr.render(c))
        except G.Refuse as e: got = None; why = str(e)
        try: want = R.generate(c)
        except Exception as e: cnt["generate_crash"] += 1; per_group[group]["generate_crash"] += 1; continue
        if got is None: cnt["refused"] += 1; per_group[group]["refused"] += 1; examples.setdefault(("refused", why[:30]), str(label)); continue
        if got == want: cnt["match"] += 1; per_group[group]["match"] += 1
        else:
            cnt["mismatch"] += 1; per_group[group]["mismatch"] += 1
            key = ("mismatch", group, want[-6:], got[-6:])
            if key not in examples: examples[key] = {"frame": str(label), "rules": got, "generate": want}
    mism = [v for k, v in examples.items() if k[0] == "mismatch"]
    return {"frames": sum(cnt.values()), "match": cnt["match"], "mismatches": cnt["mismatch"], "refused": cnt["refused"], "generate_crash": cnt["generate_crash"],
            "per_group": {g: dict(v) for g, v in per_group.items()}, "refusal_reasons": {k[1]: v for k, v in examples.items() if k[0] == "refused"},
            "mismatch_examples": mism[:40], "seconds": round(time.time() - t0, 1)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="result.json"); ap.add_argument("--no-sweep", action="store_true"); a = ap.parse_args()
    pr = G.PrintRules()
    outputs, per = corpus(pr)
    res = {"per_split": per, "outputs": outputs, "positions": len(pr.positions()), "rules": len(pr.rules()), "derived": len(pr.derived())}
    res["agree"] = sum(p["agree"] for p in per.values()); res["gap"] = sum(p["gap"] for p in per.values()); res["misfire"] = sum(p["misfire"] for p in per.values())
    res["sweep"] = sweep(pr) if not a.no_sweep else {"frames": 0, "match": 0, "mismatches": 0, "refused": 0, "generate_crash": 0}
    sw = res["sweep"]
    res["summary"] = (f"printer vs generate: agree {res['agree']}, gap {res['gap']}, misfire {res['misfire']} on the corpus; sweep {sw['frames']} frames, "
                      f"{sw['match']} match, {sw['mismatches']} mismatches, {sw['refused']} refused, {sw['generate_crash']} generate() cannot print; {res['rules']} rules at {res['positions']} positions")
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(res["summary"])
    for e in sw.get("mismatch_examples", [])[:15]: print("  ", e)

if __name__ == "__main__":
    main()
