"""Schema checks: every declared data file against its JSON Schema (draft 2020-12), plus the rules a schema cannot state.

    tables/*.tbl            each row, as the record view in tables.COLUMNS, against schemas/tables/<name>.schema.json;
                            then schemas/tables/integrity.json: uniqueness and cross-table references
    harness/corpus.lock.json against schemas/corpus-lock.schema.json; the pinned files' line counts when the corpus is present
    harness/baseline.json    against schemas/baseline.schema.json; every cited sentence must be in the pinned corpus split
                            (the same filter the harness applies), each split cited completely, and the metrics must agree
                            with the outputs they summarise
    harness/adjudications.json against schemas/adjudications.schema.json; every citation must be a pinned corpus line

Exit 1 on any error; reference violations declared level "warn" are listed and do not fail.
    python harness/check_schemas.py [--root DIR] [--corpus DIR] [--quiet]
"""
import sys, os, json, argparse, collections
from jsonschema import Draft202012Validator

def load(path): return json.load(open(path, encoding="utf-8"))

def validate(instance, schema_path, label):
    """-> list of error strings"""
    v = Draft202012Validator(load(schema_path))
    return [f"{label}: {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message[:160]}" for e in sorted(v.iter_errors(instance), key=lambda e: list(e.absolute_path))]

def check_tables(root):
    sys.path.insert(0, root)
    import tables as T
    errors, warnings = [], []
    recs = {}
    for name in T.COLUMNS:
        recs[name] = T.records(name)
        v = Draft202012Validator(load(os.path.join(root, "schemas", "tables", f"{name}.schema.json")))
        for rec, n in recs[name]:
            for e in v.iter_errors(rec):
                errors.append(f"tables/{name}.tbl line {n}: {'/'.join(str(p) for p in e.absolute_path) or 'row'}: {e.message[:140]}")
    integ_path = os.path.join(root, "schemas", "tables", "integrity.json"); integ = load(integ_path)
    errors += validate(integ, os.path.join(root, "schemas", "integrity.schema.json"), "schemas/tables/integrity.json")
    def key_of(rec, field, token=None):
        v = rec.get(field)
        if v is None: return []
        vals = v if isinstance(v, list) else [v]
        if token is not None: vals = [x.split()[token] for x in vals if x.split()]
        return vals
    for u in integ.get("unique", []):
        seen = {}
        for rec, n in recs.get(u["table"], []):
            k = tuple(json.dumps(rec.get(f), ensure_ascii=False) for f in u["fields"])
            if k in seen: errors.append(f"tables/{u['table']}.tbl line {n}: duplicate {dict(zip(u['fields'], k))} (first at line {seen[k]})")
            else: seen[k] = n
    for r in integ.get("references", []):
        to = r["to"]; where = to.get("where", {})
        targets = set()
        for rec, n in recs.get(to["table"], []):
            if all(rec.get(k) == v for k, v in where.items()): targets |= set(key_of(rec, to["field"], to.get("token")))
        missing = []
        for rec, n in recs.get(r["from"]["table"], []):
            for val in key_of(rec, r["from"]["field"], r["from"].get("token")):
                if val not in targets: missing.append((val, n))
        if missing:
            msg = (f"tables/{r['from']['table']}.tbl: {len(missing)} value(s) of {r['from']['field']} not in {to['table']}.{to['field']}"
                   + (f" where {where}" if where else "") + ": " + ", ".join(f"{v!r} (line {n})" for v, n in missing[:8]) + (" …" if len(missing) > 8 else ""))
            (errors if r["level"] == "error" else warnings).append(msg)
    return errors, warnings, sum(len(v) for v in recs.values())

def corpus_citations(corpus_dir, split):
    """{'<line>: <sentence>'} for the lines of <split>.en the harness keeps (run_rules.keep: one sentence ending in . ? !)"""
    en = [l.split() for l in open(os.path.join(corpus_dir, split + ".en"), encoding="utf-8")]
    return {f"{i}: {' '.join(e)}" for i, e in enumerate(en, 1) if e and e[-1] in (".", "?", "!") and not any(t in (".", "?", "!") for t in e[:-1])}

def check_corpus_lock(root, corpus_dir):
    path = os.path.join(root, "harness", "corpus.lock.json"); lock = load(path)
    errors = validate(lock, os.path.join(root, "schemas", "corpus-lock.schema.json"), "harness/corpus.lock.json")
    if corpus_dir and os.path.isdir(corpus_dir):
        for fname, spec in lock["files"].items():
            p = os.path.join(corpus_dir, fname)
            if os.path.exists(p):
                n = sum(1 for _ in open(p, encoding="utf-8"))
                if n != spec["lines"]: errors.append(f"harness/corpus.lock.json: {fname} has {n} lines, lock says {spec['lines']}")
    return errors, lock

def check_baseline(root, corpus_dir):
    path = os.path.join(root, "harness", "baseline.json")
    if not os.path.exists(path): return ["harness/baseline.json: missing"]
    base = load(path)
    errors = validate(base, os.path.join(root, "schemas", "baseline.schema.json"), "harness/baseline.json")
    if errors: return errors
    for split in ("dev", "test"):
        outs = base["outputs"]["rules"][split]; det = base["detail"]["rules"][split]
        covered = sum(1 for v in outs.values() if v)
        if covered != det["covered"]: errors.append(f"baseline {split}: {covered} non-empty outputs but detail.covered = {det['covered']}")
        if len(outs) != det["n"]: errors.append(f"baseline {split}: {len(outs)} outputs but detail.n = {det['n']}")
        if base["metrics"][f"rules.{split}.covered"] != det["covered"]: errors.append(f"baseline {split}: metrics.covered != detail.covered")
        if abs(base["metrics"][f"rules.{split}.bleu_all"] - det["bleu_all"]) > 1e-9: errors.append(f"baseline {split}: metrics.bleu_all != detail.bleu_all")
        if corpus_dir and os.path.exists(os.path.join(corpus_dir, split + ".en")):
            cites = corpus_citations(corpus_dir, split)
            foreign = [s for s in outs if s not in cites]
            if foreign: errors.append(f"baseline {split}: {len(foreign)} citation(s) do not match the pinned corpus line: " + "; ".join(repr(s) for s in foreign[:3]))
            uncited = cites - set(outs)
            if uncited: errors.append(f"baseline {split}: {len(uncited)} corpus line(s) have no output: " + "; ".join(repr(s) for s in sorted(uncited)[:3]))
            for part in ("matrix", "refusals"):
                bad = [s for s in base["outputs"][part].get(split, {}) if s not in cites]
                if bad: errors.append(f"baseline {split}: {len(bad)} {part} citation(s) do not match the pinned corpus line")
        refused = {k for k, v in outs.items() if not v}
        if set(base["outputs"]["refusals"][split]) != refused: errors.append(f"baseline {split}: the refusal reasons do not cover exactly the refused sentences")
    m = base["detail"]["matrix"]
    if base["metrics"]["matrix.identical"] != m["identical"]: errors.append("baseline: metrics.matrix.identical != detail")
    if base["metrics"]["matrix.silent_misfires"] != m["wrong_frame"] + m["matrix_parses_refused"]: errors.append("baseline: matrix.silent_misfires != wrong_frame + matrix_parses_refused")
    if base["metrics"]["np.mismatches"] != base["detail"]["np"]["mismatches"]: errors.append("baseline: metrics.np.mismatches != detail")
    return errors

def check_adjudications(root, corpus_dir):
    path = os.path.join(root, "harness", "adjudications.json")
    if not os.path.exists(path): return []
    doc = load(path)
    errors = validate(doc, os.path.join(root, "schemas", "adjudications.schema.json"), "harness/adjudications.json")
    if errors: return errors
    cites = {}
    for a in doc["sentences"]:
        split, _, rest = a["citation"].partition(" ")
        if corpus_dir and os.path.exists(os.path.join(corpus_dir, split + ".en")):
            if split not in cites: cites[split] = corpus_citations(corpus_dir, split)
            if rest not in cites[split]: errors.append(f"adjudication cites a line that is not in the pinned corpus: {a['citation']!r}")
    seen = collections.Counter((a["citation"], a.get("output")) for a in doc["sentences"])
    for k, n in seen.items():
        if n > 1: errors.append(f"adjudication repeated: {k}")
    return errors

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--corpus", help="directory with the pinned corpus files (default: .cache/corpus/<commit> under the repo)"); ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true", help="the baseline is about to be rewritten")
    a = ap.parse_args(); root = os.path.abspath(a.root)
    errs, lock = check_corpus_lock(root, a.corpus)
    corpus = a.corpus or os.path.join(root, ".cache", "corpus", lock.get("commit", ""))
    if a.corpus is None and not os.path.isdir(corpus):
        # the checked tree may be a staged export: fall back to the live repo's cache
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); corpus = os.path.join(here, ".cache", "corpus", lock.get("commit", ""))
    errs += check_corpus_lock(root, corpus)[0] if a.corpus is None else []
    t_err, t_warn, nrows = check_tables(root)
    if os.path.exists(os.path.join(root, "harness", "negatives.json")):
        errs += validate(load(os.path.join(root, "harness", "negatives.json")), os.path.join(root, "schemas", "negatives.schema.json"), "harness/negatives.json")
    b_err = [] if a.skip_baseline else check_baseline(root, corpus)
    b_err += check_adjudications(root, corpus)
    errors = list(dict.fromkeys(errs + t_err + b_err))
    for w in t_warn: print("warning:", w)
    for e in errors[:20]: print("error:", e)
    if len(errors) > 20: print(f"… {len(errors) - 20} more error(s)")
    if not a.quiet or errors:
        print(f"schemas: {nrows} table rows, corpus lock, baseline — {len(errors)} error(s), {len(t_warn)} warning(s)")
    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
