"""The lattice of dominance: the state of the project as a product of small chains, one per item, so that a change is an
improvement exactly when no item moves down and the frontier is the set of items not yet at the top of their chain.

Items and their chains (levels rise left to right; a lower index is worse):
  sentence <split> <citation>   the rule transducer's output for a corpus sentence
        WRONG < REFUSED < RIGHT, and UNKNOWN below RIGHT but incomparable with WRONG and REFUSED
        RIGHT    = the reference, or an output adjudicated right; for an input adjudicated ungrammatical, a refusal
        WRONG    = an output adjudicated wrong; any output for an ungrammatical input
        REFUSED  = no output (and the input is not adjudicated ungrammatical)
        UNKNOWN  = an output nobody has judged — a change away from it needs a verdict (harness/adjudications.json)
  matrix <split> <citation>     the matrix parser against the rule parser on the same sentence
        MISFIRE (parses where the rules refuse, or a different frame) < GAP (refuses where the rules parse) < AGREE
  negative <sentence>           ACCEPTED < REFUSED            (the rule parser on an ungrammatical sentence)
  control <sentence>            REFUSED < ACCEPTED            (the rule parser on its grammatical control)
  negative.matrix <sentence>    ACCEPTED < REFUSED            (the matrix parser on an ungrammatical sentence)
  np.exhaustive                 MISMATCHES < MATCH            (np_module against parse_np: the mismatch count, 0 is the top)

Coverage and quality are views of this state, not gates: coverage = |RIGHT ∪ UNKNOWN| / N, quality = chrF over the outputs.
"""
import json, os, collections

# the poset of sentence levels: pairs (lower, higher)
SENTENCE_ORDER = {("WRONG", "REFUSED"), ("WRONG", "RIGHT"), ("REFUSED", "RIGHT"), ("UNKNOWN", "RIGHT")}
CHAINS = {"matrix": ["MISFIRE", "GAP", "AGREE"], "negative": ["ACCEPTED", "REFUSED"], "control": ["REFUSED", "ACCEPTED"],
          "negative.matrix": ["ACCEPTED", "REFUSED"], "np.exhaustive": ["MISMATCHES", "MATCH"]}
TOP = {"sentence": "RIGHT", "matrix": "AGREE", "negative": "REFUSED", "control": "ACCEPTED", "negative.matrix": "REFUSED", "np.exhaustive": "MATCH"}

def cmp(kind, old, new):
    """-> 'up' | 'down' | 'same' | 'incomparable'"""
    if old == new: return "same"
    if kind == "sentence":
        if (old, new) in SENTENCE_ORDER: return "up"
        if (new, old) in SENTENCE_ORDER: return "down"
        return "incomparable"
    chain = CHAINS[kind]
    return "up" if chain.index(new) > chain.index(old) else "down"

def load_adjudications(path):
    """-> ({citation: {output: verdict}}, {citation: reason for an ungrammatical input})"""
    verdicts = collections.defaultdict(dict); ungrammatical = {}
    if not os.path.exists(path): return verdicts, ungrammatical
    doc = json.load(open(path, encoding="utf-8"))
    for a in doc.get("sentences", []):
        key = a["citation"]
        if a["verdict"] == "input_ungrammatical": ungrammatical[key] = a["reason"]
        else: verdicts[key][a["output"]] = a["verdict"]
    return verdicts, ungrammatical

def sentence_level(output, reference, key, verdicts, ungrammatical):
    if key in ungrammatical: return "RIGHT" if not output else "WRONG"
    if not output: return "REFUSED"
    if output == reference: return "RIGHT"
    v = verdicts.get(key, {}).get(output)
    return "RIGHT" if v == "right" else "WRONG" if v == "wrong" else "UNKNOWN"

def matrix_level(rules_out, matrix_out):
    if rules_out and matrix_out == rules_out: return "AGREE"
    if not rules_out and not matrix_out: return "AGREE"
    if rules_out and not matrix_out: return "GAP"
    return "MISFIRE"

def levels(rules_outputs, matrix_outputs, negatives_rows, np_mismatches, references, verdicts, ungrammatical):
    """every item -> (kind, level, detail).  rules_outputs / matrix_outputs: {split: {citation: output}}; references: {split: {citation: ja}}"""
    L = {}
    for split in ("dev", "test"):
        for key, out in rules_outputs.get(split, {}).items():
            item = f"{split} {key}"
            L[f"sentence {item}"] = ("sentence", sentence_level(out, references[split].get(key), item, verdicts, ungrammatical), out)
            if split in matrix_outputs and key in matrix_outputs[split]:
                L[f"matrix {item}"] = ("matrix", matrix_level(out, matrix_outputs[split][key]), matrix_outputs[split][key])
    for row in negatives_rows or []:
        L[f"negative {row['sentence']}"] = ("negative", "ACCEPTED" if row.get("sentence.rules") else "REFUSED", row.get("sentence.rules"))
        L[f"control {row['control']}"] = ("control", "ACCEPTED" if row.get("control.rules") else "REFUSED", row.get("control.rules.reason"))
        L[f"negative.matrix {row['sentence']}"] = ("negative.matrix", "ACCEPTED" if row.get("sentence.matrix") else "REFUSED", row.get("sentence.matrix"))
    L["np.exhaustive"] = ("np.exhaustive", "MATCH" if np_mismatches == 0 else "MISMATCHES", np_mismatches)
    return L

def compare_levels(old, new):
    """-> dict with lists of (item, old level, new level, detail) under 'up', 'down', 'incomparable', 'removed', 'added'"""
    out = {"up": [], "down": [], "incomparable": [], "removed": [], "added": []}
    for item, (kind, lv, det) in new.items():
        if item not in old: out["added"].append((item, None, lv, det)); continue
        okind, olv, odet = old[item]
        c = cmp(kind, olv, lv)
        if c in ("up", "down", "incomparable"): out[c].append((item, olv, lv, (odet, det)))
    for item, (kind, lv, det) in old.items():
        if item not in new: out["removed"].append((item, lv, None, det))
    return out

def summary(L):
    """counts per kind and level, in chain order"""
    by = collections.defaultdict(collections.Counter)
    for item, (kind, lv, det) in L.items(): by[kind][lv] += 1
    order = {"sentence": ["RIGHT", "UNKNOWN", "REFUSED", "WRONG"]}
    return {k: {lv: c[lv] for lv in order.get(k, list(reversed(CHAINS.get(k, [])))) if c[lv]} for k, c in by.items()}

def frontier(L, refusal_reasons=None):
    """the items not at the top of their chain, grouped by what blocks them"""
    groups = collections.defaultdict(list); reasons = refusal_reasons or {}
    for item, (kind, lv, det) in L.items():
        if lv == TOP[kind]: continue
        if kind == "sentence":
            if lv == "REFUSED": groups[f"sentence refused: {reasons.get(item.split(' ', 1)[1], 'unknown reason')}"].append(item)
            elif lv == "UNKNOWN": groups["sentence output unjudged"].append(item)
            else: groups["sentence output wrong"].append(item)
        elif kind == "matrix": groups[f"matrix {lv.lower()}"].append(item)
        else: groups[f"{kind} {lv.lower()}"].append(item)
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))
