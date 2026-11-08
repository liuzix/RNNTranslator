# RNNTranslator

RNNTranslator is an English-to-Japanese translator written as a circuit, not trained: a hand-written rule transducer whose parser is compiled, module by module, into sparse ReLU matrices that run one token at a time. Hypothesis: a language is a small circuit of tables and bounded control, and such a circuit can be written down, verified exhaustively, and compiled into weights whose every nonzero has a name. It is evaluated on the Tanaka-corpus dev (496 sentences) and test (499 sentences) splits of small_parallel_enja, pinned by commit and hash.

## Architecture

![Architecture](docs/architecture.svg)

The lexical layer merges multiword tokens and gives each token its readings from the 16 tables, and the semantic layer generates the Japanese from the frame. The parser layer has one interface and two implementations, `RuleParser` and `MatrixParser`, and the matrix implementation is meant to agree with the rule implementation wherever the matrix can represent the parse. `RuleParser` is a bounded transducer: a 4-token window, a frame stack of depth at most 2, at most 2 live readings resolved within 4 tokens, and no rewinds. `MatrixParser` runs the same machine as 5 sparse ReLU layers per step, with the clause-level control extracted from traces of the rule parser and the noun-phrase module translated by hand from `parse_np`. Two feedback edges cross the layers: the noun's use-verb rewrites the verb (she took a rest → 彼女は休んだ), and the parser's state decides which reading of a token is used (i rest → 休む, i need a rest → 休息).

## Layout

<!-- tree:begin -->
```
.
├── .githooks/
│   └── pre-commit                    uv lock --check, then harness/run.py check --staged
├── archive/
│   ├── README.md                     what each blob holds and why it retired
│   └── retired-2026-11-08.tar.xz     fragment-era experiments, pre-hybrid compiler, old checks
├── docs/
│   ├── README.tmpl.md                the source of README.md; its numbers are fact placeholders
│   ├── architecture.lock.json        PlantUML jar hash, source and SVG hashes
│   ├── architecture.puml             the architecture diagram, source
│   └── architecture.svg              the architecture diagram, as rendered
├── harness/
│   ├── baseline.json                 metrics + every rule output at the last accepted commit
│   ├── check_diagram.py              the diagram files against their lock
│   ├── check_readme.py               README.md against the template, the facts and the appendix
│   ├── check_schemas.py              tables, corpus lock and baseline against their JSON Schemas
│   ├── check_tree.py                 this file tree against the tracked files
│   ├── corpus.lock.json              the pinned corpus: commit, sha256 and line count per file
│   ├── eval_matrix.py                action: hybrid cell compiled on dev+test, cross-tab vs the rule parser
│   ├── eval_np.py                    action: np_module against parse_np over all windows
│   ├── eval_rules.py                 action: coverage and char-BLEU of the transducer
│   ├── facts.py                      the facts the README may cite
│   ├── probe_sandbox.py              a probe action for the sandbox test
│   ├── render_diagram.py             renders docs/*.puml with the pinned PlantUML jar
│   ├── run.py                        the action cache, sandbox, baseline comparison, corpus fetch
│   ├── test_claims.py                the tests the README's claims are pinned to
│   └── trace_rules.py                step the hybrid rule set through one sentence
├── schemas/
│   ├── tables/
│   │   ├── *.schema.json             one JSON Schema per table's row
│   │   └── integrity.json            uniqueness and cross-table references
│   ├── baseline.schema.json
│   ├── corpus-lock.schema.json
│   ├── diagram-lock.schema.json
│   └── integrity.schema.json         the format of integrity.json
├── tables/
│   └── *.tbl                         the tables, each a small DSL file with its own header
├── .gitattributes
├── .gitignore
├── .ignore                           keeps search tools out of archive/
├── .python-version
├── CLAUDE.md                         the rules for working here: read first, pin every claim, keep everything reproducible
├── LICENSE
├── README.md                         rendered from docs/README.tmpl.md — do not edit by hand
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
uv run python harness/run.py readme                # re-render README.md after editing docs/README.tmpl.md
```

## Harness

Every commit runs the harness on the staged tree through the pre-commit hook, and a regression against the committed baseline refuses the commit. Each action declares its inputs and runs in a sandbox that holds only those inputs, cached under `.cache/actions/` by the hash of the inputs, `uv.lock` and the interpreter. Every dependency is pinned with its sha256 in `uv.lock` (7 packages). The corpus is pinned to commit 312408a11c5f with a sha256 and line count per file, and is not vendored. Every table row, the corpus lock and the baseline are validated against JSON Schemas (draft 2020-12), with uniqueness and cross-table references declared in `schemas/tables/integrity.json`, and every sentence the baseline cites is verified against the pinned corpus line. The file tree above is compared with the tracked files. The architecture diagram is rendered from `docs/architecture.puml` by PlantUML 1.2025.10, pinned by the jar's sha256, and the committed SVG must match the hash recorded at render time. Every number in this README is rendered from a named fact, and every sentence is either pinned to a test or declared a hypothesis in the appendix below. Retired files leave the tree, and what is worth reading later goes into `archive/` as a compressed blob that no text search sees into. `CLAUDE.md` states the rules every agent and human follows here, and every file and check it names exists.

The `rules` action measures coverage and char-BLEU of the transducer on dev and test and records every output. The `matrix` action compiles the hybrid cell on dev and test and cross-tabulates it against the rule parser. The `np` action checks `np_module` against `parse_np` over every window of at most 2 tokens, or 3 with `--full`.

## Where it stands

The rule transducer translates 400 of 499 test sentences (80%) and 310 of 496 dev sentences (62%), with char-BLEU 35.8 and 34.5 inside coverage. On the 995 dev and test sentences, the two parser implementations parse 498 identically; the matrix misfires silently on 31 (14 wrong frames and 17 parses of sentences the rules refuse) and refuses 198 sentences the rules parse. The compiled cell has 1437 rules (333 extracted, 100 entry, 169 exit, 76 refusal, 759 from the noun-phrase module), 12,592 state units, 1,547 input units and 171,184 nonzeros. `np_module` agrees with `parse_np` on all 27,888 windows of at most 2 tokens over 83 token classes. Its two adjective slots refuse an adverb followed by two adjectives, where `parse_np` concatenates. Hypothesis: the remaining misfires come from the extracted clause-level rules, so the next modules to translate by hand are the PP machine, `predicate` and `copula_predicate`.

## License

The code is MIT-licensed, and the corpus is the Tanaka Corpus via small_parallel_enja under its own terms.

## Appendix: claims and their pins

Every sentence above is quoted here with what pins it: a test in `harness/test_claims.py`, an action or a check in `harness/run.py` (all run by the pre-commit hook), or `Hypothesis`. `harness/check_readme.py` verifies the quotes, the pins and the coverage.

- "RNNTranslator is an English-to-Japanese translator written as a circuit, not trained: a hand-written rule transducer whose parser is compiled, module by module, into sparse ReLU matrices that run one token at a time.": Pinned by `test_cell_weights_are_named_integers`
- "Hypothesis: a language is a small circuit of tables and bounded control, and such a circuit can be written down, verified exhaustively, and compiled into weights whose every nonzero has a name.": Hypothesis
- "It is evaluated on the Tanaka-corpus dev (496 sentences) and test (499 sentences) splits of small_parallel_enja, pinned by commit and hash.": Pinned by `test_corpus_pinned`
- "The lexical layer merges multiword tokens and gives each token its readings from the 16 tables, and the semantic layer generates the Japanese from the frame.": Pinned by `test_layers_roundtrip`
- "The parser layer has one interface and two implementations, `RuleParser` and `MatrixParser`, and the matrix implementation is meant to agree with the rule implementation wherever the matrix can represent the parse.": Pinned by `test_parser_interface_two_implementations`
- "`RuleParser` is a bounded transducer: a 4-token window, a frame stack of depth at most 2, at most 2 live readings resolved within 4 tokens, and no rewinds.": Pinned by `test_rule_parser_bounds`
- "`MatrixParser` runs the same machine as 5 sparse ReLU layers per step, with the clause-level control extracted from traces of the rule parser and the noun-phrase module translated by hand from `parse_np`.": Pinned by `test_matrix_cell_shape`
- "Two feedback edges cross the layers: the noun's use-verb rewrites the verb (she took a rest → 彼女は休んだ), and the parser's state decides which reading of a token is used (i rest → 休む, i need a rest → 休息).": Pinned by `test_feedback_edges`
- "Every commit runs the harness on the staged tree through the pre-commit hook, and a regression against the committed baseline refuses the commit.": Pinned by `test_precommit_hook_runs_check`
- "Each action declares its inputs and runs in a sandbox that holds only those inputs, cached under `.cache/actions/` by the hash of the inputs, `uv.lock` and the interpreter.": Pinned by `test_sandbox_holds_only_declared_inputs`
- "Every dependency is pinned with its sha256 in `uv.lock` (7 packages).": Pinned by `test_dependencies_hash_pinned`
- "The corpus is pinned to commit 312408a11c5f with a sha256 and line count per file, and is not vendored.": Pinned by `test_corpus_pinned`
- "Every table row, the corpus lock and the baseline are validated against JSON Schemas (draft 2020-12), with uniqueness and cross-table references declared in `schemas/tables/integrity.json`, and every sentence the baseline cites is verified against the pinned corpus line.": Pinned by `check_schemas`
- "The file tree above is compared with the tracked files.": Pinned by `check_tree`
- "The architecture diagram is rendered from `docs/architecture.puml` by PlantUML 1.2025.10, pinned by the jar's sha256, and the committed SVG must match the hash recorded at render time.": Pinned by `check_diagram`
- "Every number in this README is rendered from a named fact, and every sentence is either pinned to a test or declared a hypothesis in the appendix below.": Pinned by `check_readme`
- "Retired files leave the tree, and what is worth reading later goes into `archive/` as a compressed blob that no text search sees into.": Pinned by `test_archive_is_opaque`
- "`CLAUDE.md` states the rules every agent and human follows here, and every file and check it names exists.": Pinned by `test_claude_md_references_exist`
- "The `rules` action measures coverage and char-BLEU of the transducer on dev and test and records every output.": Pinned by `rules`
- "The `matrix` action compiles the hybrid cell on dev and test and cross-tabulates it against the rule parser.": Pinned by `matrix`
- "The `np` action checks `np_module` against `parse_np` over every window of at most 2 tokens, or 3 with `--full`.": Pinned by `np`
- "The rule transducer translates 400 of 499 test sentences (80%) and 310 of 496 dev sentences (62%), with char-BLEU 35.8 and 34.5 inside coverage.": Pinned by `rules`
- "On the 995 dev and test sentences, the two parser implementations parse 498 identically; the matrix misfires silently on 31 (14 wrong frames and 17 parses of sentences the rules refuse) and refuses 198 sentences the rules parse.": Pinned by `matrix`
- "The compiled cell has 1437 rules (333 extracted, 100 entry, 169 exit, 76 refusal, 759 from the noun-phrase module), 12,592 state units, 1,547 input units and 171,184 nonzeros.": Pinned by `matrix`
- "`np_module` agrees with `parse_np` on all 27,888 windows of at most 2 tokens over 83 token classes.": Pinned by `np`
- "Its two adjective slots refuse an adverb followed by two adjectives, where `parse_np` concatenates.": Pinned by `test_np_two_adjective_slots`
- "Hypothesis: the remaining misfires come from the extracted clause-level rules, so the next modules to translate by hand are the PP machine, `predicate` and `copula_predicate`.": Hypothesis
- "The code is MIT-licensed, and the corpus is the Tanaka Corpus via small_parallel_enja under its own terms.": Pinned by `test_license_is_mit`
