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
    f["baseline.taken"] = base["taken"]
    # constants of the parser
    f["parser.window"] = PR.BUFFER; f["parser.max_depth"] = PR.MAX_DEPTH; f["parser.width"] = PR.WIDTH; f["parser.hold"] = PR.HOLD
    # counts in the tree and the locks
    f["tables.count"] = len(glob.glob(os.path.join(root, "tables", "*.tbl")))
    f["schemas.tables.count"] = len(glob.glob(os.path.join(root, "schemas", "tables", "*.schema.json")))
    lock = json.load(open(os.path.join(root, "harness", "corpus.lock.json"), encoding="utf-8"))
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
