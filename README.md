# RNNTranslator

An English → Japanese translator written as a circuit, not trained: a hand-written rule transducer whose parser
is being compiled, module by module, into sparse ReLU matrices that run one token at a time. The thesis under
test: a language is a small circuit of tables and bounded control; such a circuit can be written down, verified
exhaustively, and then compiled into weights whose every nonzero has a name.

Evaluated on the Tanaka-corpus dev (496) and test (499) splits of
[small_parallel_enja](https://github.com/odashi/small_parallel_enja), pinned by commit and hash.

## Architecture

```
  English tokens
        │
        ▼
 ┌─────────────────┐  tables/*.tbl: lexicon, idioms, use-verbs, verb classes, …
 │  lexical layer  │  merge multiword tokens → readings per token → a 4-token window
 └────────┬────────┘
          │ window of readings            ▲  feedback: the parser's phase gates which readings survive
          ▼                               │
 ┌─────────────────┐  parser.py — one interface, two implementations
 │  parser layer   │    RuleParser    parser_rules.py   bounded transducer: frame stack ≤ 2, 2 live readings × 4 tokens, no rewinds
 │                 │    MatrixParser  parser_matrix.py  the same machine as five sparse ReLU layers per step:
 │                 │                                    clause control extracted from traces; parse_np from np_module.py
 └────────┬────────┘
          │ frame (Clause: subject, object, verb, tense, …)      ▲  feedback: noun class and valency from the tables
          ▼                                                      │
 ┌─────────────────┐
 │ semantic layer  │  rules_ja.generate: slot order per clause type, conjugation paradigms, connectives
 └────────┬────────┘
          ▼
  Japanese
```

Both parser implementations must agree exactly inside the matrix's scope; where the matrix cannot represent a
parse (an inner clause, a register with no block) it refuses rather than guesses. The harness holds the line:
every commit is checked against a baseline of coverage, quality and agreement, hermetically and by content hash.

## Layout

<!-- tree:begin -->
```
.
├── .githooks/
│   └── pre-commit                    uv lock --check, then harness/run.py check --staged
├── archive/
│   ├── README.md                     what each blob holds and why it retired
│   └── retired-2026-11-08.tar.xz     fragment-era experiments, pre-hybrid compiler, old checks
├── harness/
│   ├── baseline.json                 metrics + every rule output at the last accepted commit
│   ├── check_schemas.py              tables, corpus lock and baseline against their JSON Schemas
│   ├── check_tree.py                 this file tree against the tracked files
│   ├── corpus.lock.json              the pinned corpus: commit, sha256 and line count per file
│   ├── eval_matrix.py                action: hybrid cell compiled on dev+test, cross-tab vs the rule parser
│   ├── eval_np.py                    action: np_module against parse_np over all windows
│   ├── eval_rules.py                 action: coverage and char-BLEU of the transducer
│   ├── run.py                        the action cache, sandbox, baseline comparison, corpus fetch
│   └── trace_rules.py                step the hybrid rule set through one sentence
├── schemas/
│   ├── tables/
│   │   ├── *.schema.json             one JSON Schema per table's row
│   │   └── integrity.json            uniqueness and cross-table references
│   ├── baseline.schema.json
│   ├── corpus-lock.schema.json
│   └── integrity.schema.json         the format of integrity.json
├── tables/
│   └── *.tbl                         the 16 tables, each a small DSL file with its own header
├── .gitattributes
├── .gitignore
├── .ignore                           keeps search tools out of archive/
├── .python-version
├── LICENSE
├── README.md
├── np_module.py                      parse_np translated by hand: derived predicates + rules, complete guards
├── parser.py                         the parser interface and its two implementations
├── parser_matrix.py                  the compiler to matrices and the MatrixParser
├── parser_rules.py                   the rule parser
├── pyproject.toml
├── rules_ja.py                       lexical layer, frame, generator, translate()
├── run_rules.py                      evaluation on dev/test with samples
├── tables.py                         table loader and record view
└── uv.lock                           every dependency with its sha256
```
<!-- tree:end -->

## Setup

```
uv sync --locked                                   # the environment exactly as locked
uv run python harness/run.py fetch-corpus          # the pinned corpus into .cache/corpus/, hashes verified
uv run python harness/run.py install-hooks         # git runs the checks on every commit
uv run python run_rules.py                         # the transducer on dev, with samples
uv run python harness/run.py check                 # everything the hook runs
```

## Harness

`run.py check` runs the static checks, then three cached actions, then compares with `harness/baseline.json`:

| check | what | guards |
| --- | --- | --- |
| `check_schemas` | every table row, `corpus.lock.json` and `baseline.json` against their JSON Schemas (draft 2020-12); uniqueness and cross-table references; every sentence the baseline cites is in the pinned corpus | shape |
| `check_tree` | the README file tree against the tracked files | drift |
| `rules` | coverage and char-BLEU of the transducer on dev and test, every output recorded | coverage, BLEU-over-all |
| `matrix` | the hybrid cell compiled on dev+test, cross-tabulated against the rule parser | agreement, silent misfires |
| `np` | `np_module` against `parse_np` over every window ≤ 2 tokens (`--full`: ≤ 3) | mismatches |

An action's key is the hash of its declared inputs, `uv.lock` and the interpreter; it runs in a sandbox holding only
those inputs, with a fixed hash seed and a minimal environment, and is cached under `.cache/actions/`. The pre-commit
hook runs `check --staged` on the index and refuses a regression against the baseline as staged; an intended change is
accepted with `run.py baseline --staged`. Retired files leave the tree (git keeps them); what is worth reading later
goes into `archive/` as a compressed blob with a manifest, so searches stay quiet.

## Where it stands (2026-11-08)

Rule transducer: test coverage 80% (400/499), dev 62.5% (310/496); char-BLEU inside coverage ≈ 35.

Matrix parser against the rule parser, dev+test (995 sentences; `harness/baseline.json` has the current figures):

| | extraction only | hybrid: `parse_np` as a module |
| --- | --- | --- |
| both parse, identical | 461 | 498 |
| silent misfires (wrong frame, or parses what the rules refuse) | 116 | 31 |
| rules parse, matrix refuses | 177 | 198 |
| rules in the cell / nonzeros | 946 / 52,069 | 1,458 / 180,046 |

`np_module` is checked exhaustively against `parse_np`: 0 mismatches over 2.3M windows of length ≤ 3; at length 4
the only divergence is the adjective-slot capacity (adverb + two adjectives), where the module refuses loudly.
The remaining misfires are all in the extracted clause-level rules; the next modules to translate by hand are the
PP machine, `predicate` and `copula_predicate`.

## License

MIT (see `LICENSE`). The corpus is the Tanaka Corpus via small_parallel_enja, under its own terms; it is not vendored.
