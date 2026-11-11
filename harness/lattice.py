"""The lattice of dominance: the state of the project as a product of small chains, one per item, so that a change is an
improvement exactly when no item moves down and the frontier is the set of items not yet at the top of their chain.

Items and their chains (levels rise left to right; a lower index is worse):
  sentence <split> <citation>   the rule transducer's output for a corpus sentence
        WRONG < REFUSED < STIFF < RIGHT, and UNKNOWN below RIGHT but incomparable with WRONG, REFUSED and STIFF
        RIGHT    = the reference, or an output judged right — natural Japanese that says what the English says; for an input
                   judged ungrammatical, a refusal
        STIFF    = an output judged stiff — grammatical and faithful, but not how it is said
        WRONG    = an output judged wrong; any output for an ungrammatical input
        REFUSED  = no output (and the input is not adjudicated ungrammatical)
        UNKNOWN  = an output nobody has judged — a change away from it, to a refusal or to another unjudged output,
                   needs a verdict on the unjudged output(s) (harness/adjudications.json)
  matrix <split> <citation>     the matrix parser against the rule parser on the same sentence
        MISFIRE (parses where the rules refuse, or a different frame) < GAP (refuses where the rules parse) < AGREE
  negative <sentence>           ACCEPTED < REFUSED            (the rule parser on an ungrammatical sentence)
  control <sentence>            REFUSED < ACCEPTED            (the rule parser on its grammatical control)
  negative.matrix <sentence>    ACCEPTED < REFUSED            (the matrix parser on an ungrammatical sentence)
  np.exhaustive                 MISMATCHES < MATCH            (np_module against parse_np: the mismatch count, 0 is the top)
  suite <class> <sentence>      WRONG < REFUSED < RIGHT       (an acceptance case: RIGHT = the output equals the suite's expectation)
  torch.exact                   DIVERGING < EXACT             (the torch runner against the scipy runner: sentences whose states differ, 0 is the top)
  printer <split> <citation>    MISFIRE < GAP < AGREE         (the printing rules against generate() on a sentence the rule parser parses)
  printer.sweep                 MISMATCHES < MATCH            (the printing rules against generate() on the sweep over the predicates they test)

Coverage and quality are views of this state, not gates: coverage = |RIGHT ∪ UNKNOWN| / N, quality = chrF over the outputs.
"""
import json, os, collections

# the poset of sentence levels: pairs (lower, higher)
SENTENCE_ORDER = {("WRONG", "REFUSED"), ("WRONG", "STIFF"), ("WRONG", "RIGHT"), ("REFUSED", "STIFF"), ("REFUSED", "RIGHT"), ("STIFF", "RIGHT"), ("UNKNOWN", "RIGHT")}
VERDICT_LEVEL = {"right": "RIGHT", "stiff": "STIFF", "wrong": "WRONG"}
# what a stiff or wrong verdict blames — the grouping of the frontier's judged outputs (schemas/adjudications.schema.json)
CAUSES = ("sense", "preposition", "idiom", "case", "giving", "structure", "tense", "morphology", "it", "quantity", "negation", "pronoun", "register",
          "omission", "literal", "order")
CHAINS = {"matrix": ["MISFIRE", "GAP", "AGREE"], "negative": ["ACCEPTED", "REFUSED"], "control": ["REFUSED", "ACCEPTED"],
          "negative.matrix": ["ACCEPTED", "REFUSED"], "np.exhaustive": ["MISMATCHES", "MATCH"], "suite": ["WRONG", "REFUSED", "RIGHT"],
          "torch.exact": ["DIVERGING", "EXACT"], "printer": ["MISFIRE", "GAP", "AGREE"], "printer.sweep": ["MISMATCHES", "MATCH"]}
TOP = {"sentence": "RIGHT", "matrix": "AGREE", "negative": "REFUSED", "control": "ACCEPTED", "negative.matrix": "REFUSED", "np.exhaustive": "MATCH", "suite": "RIGHT",
       "torch.exact": "EXACT", "printer": "AGREE", "printer.sweep": "MATCH"}

def cmp(kind, old, new):
    """-> 'up' | 'down' | 'same' | 'incomparable'"""
    if old == new: return "same"
    if kind == "sentence":
        if (old, new) in SENTENCE_ORDER: return "up"
        if (new, old) in SENTENCE_ORDER: return "down"
        return "incomparable"
    chain = CHAINS[kind]
    return "up" if chain.index(new) > chain.index(old) else "down"

def load_adjudications(path, suites_dir=None):
    """-> ({citation: {output: verdict}}, {citation: reason for an ungrammatical input}); the suites' expectations for cited
    corpus lines count as right verdicts (suites_dir defaults to harness/suites next to the adjudications file)"""
    verdicts = collections.defaultdict(dict); ungrammatical = {}
    if os.path.exists(path):
        doc = json.load(open(path, encoding="utf-8"))
        for a in doc.get("sentences", []):
            key = a["citation"]
            if a["verdict"] == "input_ungrammatical": ungrammatical[key] = a["reason"]
            else: verdicts[key][a["output"]] = a["verdict"]
    import glob
    for sp in sorted(glob.glob(os.path.join(suites_dir or os.path.join(os.path.dirname(path), "suites"), "*.json"))):
        for case in json.load(open(sp, encoding="utf-8")).get("cases", []):
            if case.get("citation"): verdicts[f"{case['citation']}: {case['sentence']}"].setdefault(case["expect"], "right")
    return verdicts, ungrammatical

def load_causes(path):
    """-> {citation: {output: cause}}: what each stiff or wrong verdict blames"""
    causes = collections.defaultdict(dict)
    if os.path.exists(path):
        for a in json.load(open(path, encoding="utf-8")).get("sentences", []):
            if a.get("cause"): causes[a["citation"]][a["output"]] = a["cause"]
    return causes

def sentence_level(output, reference, key, verdicts, ungrammatical):
    if key in ungrammatical: return "RIGHT" if not output else "WRONG"
    if not output: return "REFUSED"
    if output == reference: return "RIGHT"
    return VERDICT_LEVEL.get(verdicts.get(key, {}).get(output), "UNKNOWN")

def matrix_level(rules_out, matrix_out):
    if rules_out and matrix_out == rules_out: return "AGREE"
    if not rules_out and not matrix_out: return "AGREE"
    if rules_out and not matrix_out: return "GAP"
    return "MISFIRE"

def levels(rules_outputs, matrix_outputs, negatives_rows, np_mismatches, references, verdicts, ungrammatical, suites_rows=None, torch_diverging=None,
           printer_outputs=None, printer_sweep=None):
    """every item -> (kind, level, detail).  rules_outputs / matrix_outputs / printer_outputs: {split: {citation: output}}; references: {split: {citation: ja}};
    torch_diverging: the sentences on which the torch runner's states differ from the scipy runner's; printer_sweep: the sweep's mismatch count
    (None when that action did not run)"""
    L = {}
    for split in ("dev", "test"):
        for key, out in rules_outputs.get(split, {}).items():
            item = f"{split} {key}"
            L[f"sentence {item}"] = ("sentence", sentence_level(out, references[split].get(key), item, verdicts, ungrammatical), out)
            if split in matrix_outputs and key in matrix_outputs[split]:
                L[f"matrix {item}"] = ("matrix", matrix_level(out, matrix_outputs[split][key]), matrix_outputs[split][key])
            if out and printer_outputs and key in printer_outputs.get(split, {}):
                p = printer_outputs[split][key]; L[f"printer {item}"] = ("printer", "AGREE" if p == out else ("GAP" if not p else "MISFIRE"), p)
    for row in negatives_rows or []:
        L[f"negative {row['sentence']}"] = ("negative", "ACCEPTED" if row.get("sentence.rules") else "REFUSED", row.get("sentence.rules"))
        L[f"control {row['control']}"] = ("control", "ACCEPTED" if row.get("control.rules") else "REFUSED", row.get("control.rules.reason"))
        L[f"negative.matrix {row['sentence']}"] = ("negative.matrix", "ACCEPTED" if row.get("sentence.matrix") else "REFUSED", row.get("sentence.matrix"))
    L["np.exhaustive"] = ("np.exhaustive", "MATCH" if np_mismatches == 0 else "MISMATCHES", np_mismatches)
    for row in suites_rows or []:
        L[f"suite {row['class']} {row['sentence']}"] = ("suite", row["level"], row["output"])
    if torch_diverging is not None: L["torch.exact"] = ("torch.exact", "EXACT" if torch_diverging == 0 else "DIVERGING", torch_diverging)
    if printer_sweep is not None: L["printer.sweep"] = ("printer.sweep", "MATCH" if printer_sweep == 0 else "MISMATCHES", printer_sweep)
    return L

def compare_levels(old, new):
    """-> dict with lists of (item, old level, new level, detail) under 'up', 'down', 'incomparable', 'removed', 'added';
    an unjudged output replaced by a different unjudged output is 'incomparable' too (UNKNOWN -> UNKNOWN with another string)"""
    out = {"up": [], "down": [], "incomparable": [], "removed": [], "added": []}
    for item, (kind, lv, det) in new.items():
        if item not in old: out["added"].append((item, None, lv, det)); continue
        okind, olv, odet = old[item]
        c = cmp(kind, olv, lv)
        if c == "same" and lv == "UNKNOWN" and odet != det: c = "incomparable"
        if c in ("up", "down", "incomparable"): out[c].append((item, olv, lv, (odet, det)))
    for item, (kind, lv, det) in old.items():
        if item not in new: out["removed"].append((item, lv, None, det))
    return out

def summary(L):
    """counts per kind and level, in chain order"""
    by = collections.defaultdict(collections.Counter)
    for item, (kind, lv, det) in L.items(): by[kind][lv] += 1
    order = {"sentence": ["RIGHT", "STIFF", "UNKNOWN", "REFUSED", "WRONG"]}
    return {k: {lv: c[lv] for lv in order.get(k, list(reversed(CHAINS.get(k, [])))) if c[lv]} for k, c in by.items()}

def frontier(L, refusal_reasons=None, causes=None):
    """the items not at the top of their chain, grouped by what blocks them: a refusal by where the parser stopped, a judged
    output by the cause its verdict names (causes: load_causes)"""
    groups = collections.defaultdict(list); reasons = refusal_reasons or {}; causes = causes or {}
    for item, (kind, lv, det) in L.items():
        if lv == TOP[kind]: continue
        if kind == "sentence":
            if lv == "REFUSED": groups[f"sentence refused: {reasons.get(item.split(' ', 1)[1], 'unknown reason')}"].append(item)
            elif lv == "UNKNOWN": groups["sentence output unjudged"].append(item)
            else: groups[f"sentence output {lv.lower()}: {causes.get(item.split(' ', 1)[1], {}).get(det) or 'no cause recorded'}"].append(item)
        elif kind == "matrix": groups[f"matrix {lv.lower()}"].append(item)
        elif kind == "suite": groups[f"suite {item.split(' ')[1]} {lv.lower()}"].append(item)
        else: groups[f"{kind} {lv.lower()}"].append(item)
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))
