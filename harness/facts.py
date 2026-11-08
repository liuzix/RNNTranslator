"""The facts README.md may cite.  Every number in the README is rendered from one of these by name (see
harness/check_readme.py), so each traces to the committed baseline, to a constant in the code, or to a count in the tree.

    uv run python harness/facts.py        print them
"""
import os, sys, json, glob, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def facts(root=ROOT):
    sys.path.insert(0, root)
    import rules_ja, parser_rules as PR        # rules_ja first: the two modules import each other
    f = {}
    base = json.load(open(os.path.join(root, "harness", "baseline.json"), encoding="utf-8"))
    lock = json.load(open(os.path.join(root, "harness", "corpus.lock.json"), encoding="utf-8"))
    for split in ("dev", "test"):
        d = base["detail"]["rules"][split]
        f[f"rules.{split}.n"] = d["n"]; f[f"rules.{split}.covered"] = d["covered"]
        f[f"rules.{split}.coverage_pct"] = round(100 * d["covered"] / d["n"])
        f[f"rules.{split}.bleu_covered"] = d["bleu_covered"]; f[f"rules.{split}.bleu_all"] = d["bleu_all"]
    m = base["detail"]["matrix"]
    for k in ("identical", "wrong_frame", "matrix_parses_refused", "refused_parseable", "both_refuse"): f[f"matrix.{k}"] = m[k]
    f["matrix.silent_misfires"] = m["wrong_frame"] + m["matrix_parses_refused"]
    f["matrix.sentences"] = sum(m[k] for k in ("identical", "wrong_frame", "matrix_parses_refused", "refused_parseable", "both_refuse"))
    cell = m["cell"]
    f["cell.rules"] = cell["rules"]; f["cell.nnz"] = cell["nnz"]; f["cell.state_units"] = cell["state_units"]; f["cell.input_units"] = cell["input_units"]
    f["cell.layers"] = len(cell["layers"])
    for kind, n in cell["rules_by_kind"].items(): f[f"cell.rules.{kind}"] = n
    n = base["detail"]["np"]
    f["np.cases"] = n["cases"]; f["np.max_len"] = n["max_len"]; f["np.mismatches"] = n["mismatches"]; f["np.alphabet"] = n["alphabet"]
    g = base["detail"]["negatives"]
    f["negatives.count"] = g["count"]; f["negatives.accepted"] = g["accepted"]; f["negatives.refused"] = g["count"] - g["accepted"]
    f["negatives.controls_accepted"] = g["count"] - g["controls_refused_by_rules"]; f["negatives.controls_matrix"] = g["count"] - g["controls_refused_by_matrix"]
    f["negatives.matrix_refused"] = g["count"] - g["matrix_accepted"]; f["negatives.matrix_accepted"] = g["matrix_accepted"]
    su = base["detail"]["suites"]
    f["suites.classes"] = len(su["classes"]); f["suites.cases"] = su["cases"]; f["suites.right"] = su["right"]; f["suites.closed"] = su["closed"]; f["suites.open"] = su["open"]
    for cls, c in su["classes"].items():
        f[f"suite.{cls}.title"] = c["title"]; f[f"suite.{cls}.cases"] = c["cases"]; f[f"suite.{cls}.right"] = c["right"]; f[f"suite.{cls}.status"] = "closed" if c["closed"] else "open"
    f["baseline.taken"] = base["taken"]
    # the lattice (harness/lattice.py): levels of every item in the committed baseline
    import importlib.util
    spec = importlib.util.spec_from_file_location("lattice", os.path.join(root, "harness", "lattice.py")); LT = importlib.util.module_from_spec(spec); spec.loader.exec_module(LT)
    verdicts, ungrammatical = LT.load_adjudications(os.path.join(root, "harness", "adjudications.json"))
    candidates = [os.environ.get("ENJA_DATA") or "", os.path.join(root, ".cache", "corpus", lock["commit"]), os.path.join(ROOT, ".cache", "corpus", lock["commit"]),
                  os.path.join(os.getcwd(), ".cache", "corpus", lock["commit"])]          # a staged export has no cache: the live repo's (the runner's cwd)
    corpus = next((c for c in candidates if c and os.path.isdir(c)), candidates[1])
    refs = {}
    for split in ("dev", "test"):
        en = [l.split() for l in open(os.path.join(corpus, split + ".en"), encoding="utf-8")]
        ja = ["".join(l.split()) for l in open(os.path.join(corpus, split + ".ja"), encoding="utf-8")]
        refs[split] = {f"{i}: {' '.join(e)}": j for i, (e, j) in enumerate(zip(en, ja), 1) if e and e[-1] in (".", "?", "!") and not any(t in (".", "?", "!") for t in e[:-1])}
    L = LT.levels(base["outputs"]["rules"], base["outputs"]["matrix"], base["outputs"]["negatives"], base["detail"]["np"]["mismatches"], refs, verdicts, ungrammatical)
    S = LT.summary(L)
    for kind, counts in S.items():
        for lv, n in counts.items(): f[f"lattice.{kind}.{lv.lower()}"] = n
    for kind in ("sentence", "matrix", "negative", "control", "negative.matrix", "np.exhaustive"):
        for lv in (["RIGHT", "UNKNOWN", "REFUSED", "WRONG"] if kind == "sentence" else LT.CHAINS[kind]): f.setdefault(f"lattice.{kind}.{lv.lower()}", 0)
    F = LT.frontier(L, {f"{s} {k}": r for s, m in base["outputs"]["refusals"].items() for k, r in m.items()})
    f["lattice.frontier.items"] = sum(len(v) for v in F.values()); f["lattice.frontier.groups"] = len(F)
    f["lattice.items"] = len(L); f["lattice.adjudications"] = len(verdicts) + len(ungrammatical)
    # constants of the parser
    f["parser.window"] = PR.BUFFER; f["parser.max_depth"] = PR.MAX_DEPTH; f["parser.width"] = PR.WIDTH; f["parser.hold"] = PR.HOLD
    # counts in the tree and the locks
    f["tables.count"] = len(glob.glob(os.path.join(root, "tables", "*.tbl")))
    f["schemas.tables.count"] = len(glob.glob(os.path.join(root, "schemas", "tables", "*.schema.json")))
    f["corpus.commit"] = lock["commit"][:12]; f["corpus.files"] = len(lock["files"])
    dl = json.load(open(os.path.join(root, "docs", "architecture.lock.json"), encoding="utf-8"))
    f["plantuml.version"] = dl["plantuml"]["version"]
    schema_url = json.load(open(os.path.join(root, "schemas", "baseline.schema.json"), encoding="utf-8"))["$schema"]
    f["schema.draft"] = re.search(r"draft/([\w-]+)/", schema_url).group(1)
    runner = open(os.path.join(root, "harness", "run.py"), encoding="utf-8").read()
    f["np.full_len"] = int(re.search(r"^FULL_NP_LEN = (\d+)", runner, re.M).group(1))
    uvlock = open(os.path.join(root, "uv.lock"), encoding="utf-8").read()
    f["deps.count"] = len(re.findall(r'^source = \{ registry', uvlock, re.M))
    return f

def render(text, f):
    """{{f:name}} or {{f:name|,}} / {{f:name|.1f}} -> the fact, formatted"""
    def sub(m):
        name, _, fmt = m.group(1).partition("|")
        if name not in f: raise KeyError(f"unknown fact {name!r}")
        return format(f[name], fmt) if fmt else str(f[name])
    return re.sub(r"\{\{f:([^}]+)\}\}", sub, text)

if __name__ == "__main__":
    for k, v in facts().items(): print(f"{k:28s} {v}")
