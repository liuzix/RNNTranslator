"""Loader for tables/*.tbl — the rule system's tables, one file per table.

Every file is the same little language: '#' lines are the header (the file's own DSL, in words), blank lines are
skipped, every other line is a row of '|'-separated fields (surrounding spaces ignored). Each build_* function below
turns one file's rows into the Python structure the parser reads, and validates it.
"""
import os

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
VERB_CLASSES = {"g", "i", "s", "k", "aru", "fixed"}
ADJ_CLASSES = {"a", "na", "fx", "v"}

def rows(name):
    """the rows of tables/<name>.tbl as lists of stripped fields"""
    out = []
    with open(os.path.join(DIR, name + ".tbl"), encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"): continue
            out.append(([x.strip() for x in s.split("|")], n))
    return out

# The record view of a table, for the schema checker (harness/check_schemas.py): a row becomes an object whose keys are
# the column names below, in order; a column marked ":list" is comma-separated; a row shorter than the column list simply
# lacks the trailing keys; fields beyond the column list go into "extra" (only lexicon.tbl has a pos-dependent tail).
# The shape of each record is then stated once, formally, in schemas/tables/<name>.schema.json (JSON Schema 2020-12),
# and the cross-table keys in schemas/tables/integrity.json.
COLUMNS = {
    "lexicon": ["word", "pos", "japanese"],
    "use": ["noun", "verbs:list", "japanese", "class", "particle", "stative"],
    "idioms": ["kind", "pattern", "japanese", "class", "particle"],
    "adjectives": ["pattern", "japanese", "class", "particle"],
    "complements": ["english", "template", "class"],
    "connectives": ["kind", "english", "japanese"],
    "godan": ["ending", "a", "i", "e", "o", "te"],
    "verb_classes": ["verb", "classes:list"],
    "pronouns": ["surface", "japanese", "role"],
    "determiners": ["surface", "japanese"],
    "prepositions": ["surface", "action", "state"],
    "wh": ["surface", "japanese", "role"],
    "modals": ["surface", "class"],
    "numbers": ["surface", "japanese"],
    "contractions": ["surface", "auxiliary"],
    "duration": ["unit", "counter"],
}

def records(name):
    """tables/<name>.tbl as a list of (record dict, line number) under COLUMNS[name]"""
    cols = COLUMNS[name]; out = []
    for f, n in rows(name):
        rec = {}
        for i, col in enumerate(cols):
            if i >= len(f): break
            key, _, kind = col.partition(":")
            rec[key] = [x.strip() for x in f[i].split(",") if x.strip()] if kind == "list" else f[i]
        if len(f) > len(cols): rec["extra"] = f[len(cols):]
        out.append((rec, n))
    return out

def _dup(name, key, seen, n):
    if key in seen: raise ValueError(f"{name}.tbl line {n}: duplicate key {key!r}")
    seen.add(key)

def build_lexicon_lines():
    """lexicon rows exactly as the parser's morphology loop wants them (word|pos|ja|...)"""
    return ["|".join(f) for f, n in rows("lexicon")]

def build_use_lines():
    return ["|".join(f) for f, n in rows("use")]

def build_idioms():
    """idioms.tbl -> PHRASAL, PHRASAL_NP (phrasal verbs that keep their preposition before an object), PARTICLE_AFTER, VERB_IDIOM, IDIOM_TAIL"""
    phrasal, phrasal_np, after, idiom, tail = {}, set(), {}, {}, {}
    for f, n in rows("idioms"):
        kind, pattern, ja, conj = f[:4]; part = f[4] if len(f) > 4 else None
        if conj not in VERB_CLASSES: raise ValueError(f"idioms.tbl line {n}: unknown class {conj!r}")
        toks = pattern.split()
        lem = toks[0]
        if kind in ("phrasal", "phrasal+np"):
            if len(toks) != 2 or "NP" in toks: raise ValueError(f"idioms.tbl line {n}: a phrasal pattern is 'verb particle'")
            phrasal[(lem, toks[1])] = (ja, conj)
            if kind == "phrasal+np": phrasal_np.add((lem, toks[1]))
        elif kind == "after":
            if len(toks) != 3 or toks[1] != "NP": raise ValueError(f"idioms.tbl line {n}: an 'after' pattern is 'verb NP particle'")
            after[(lem, toks[2])] = (ja, conj)
        elif kind == "idiom":
            if toks[-1] != "NP" or len(toks) < 3 or part is None: raise ValueError(f"idioms.tbl line {n}: an 'idiom' pattern is 'verb words... NP' with a particle")
            idiom[(lem, tuple(toks[1:-1]))] = (ja, conj, part)
        elif kind == "tail":
            if toks[1] != "NP" or len(toks) < 3 or part is None: raise ValueError(f"idioms.tbl line {n}: a 'tail' pattern is 'verb NP words...' with a particle")
            tail[(lem, tuple(toks[2:]))] = (ja, conj, part)
        else:
            raise ValueError(f"idioms.tbl line {n}: unknown kind {kind!r}")
    return phrasal, phrasal_np, after, idiom, tail

def build_adjectives():
    out = {}
    for f, n in rows("adjectives"):
        pat, ja, k, part = f[:4]
        if k not in ADJ_CLASSES: raise ValueError(f"adjectives.tbl line {n}: unknown class {k!r}")
        adj, prep = pat.rsplit(" ", 1)
        out[(adj, prep)] = (ja, k, part)
    return out

def build_pairs(name, ncols=2):
    """word | value   (or word | v1 | v2 -> tuple)"""
    out, seen = {}, set()
    for f, n in rows(name):
        _dup(name, f[0], seen, n)
        out[f[0]] = f[1] if ncols == 2 else tuple(f[1:ncols])
    return out

def build_connectives():
    subord, coord, hearsay, softener = {}, {}, {}, {}
    for f, n in rows("connectives"):
        kind, en, ja = f[:3]
        {"subord": subord, "coord": coord, "hearsay": hearsay, "softener": softener}[kind][en] = ja
    return subord, coord, hearsay, softener

def build_godan():
    return {f[0]: tuple(f[1:6]) for f, n in rows("godan")}

def build_verb_classes():
    cls = {}
    for f, n in rows("verb_classes"):
        for c in f[1].split(","):
            if c: cls.setdefault(c, set()).add(f[0])
    return cls

def build_complements():
    """complements.tbl -> {want: (template, class)}"""
    out, seen = {}, set()
    for f, n in rows("complements"):
        want, tmpl, kind = f[:3]
        _dup("complements", want, seen, n)
        if kind not in ("v1", "v5", "na", "fixed", "ia"): raise ValueError(f"complements.tbl line {n}: unknown class {kind!r}")
        out[want] = (tmpl, kind)
    return out
