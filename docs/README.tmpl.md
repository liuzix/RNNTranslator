# RNNTranslator

RNNTranslator is an English-to-Japanese translator written as a circuit, not trained: a hand-written rule transducer whose parser is compiled, module by module, into sparse ReLU matrices that run one token at a time. Hypothesis: a language is a small circuit of tables and bounded control, and such a circuit can be written down, verified exhaustively, and compiled into weights whose every nonzero has a name. It is evaluated on the Tanaka-corpus dev ({{f:rules.dev.n}} sentences) and test ({{f:rules.test.n}} sentences) splits of small_parallel_enja, pinned by commit and hash.

## Architecture

![Architecture](docs/architecture.svg)

The lexical layer merges multiword tokens and gives each token its readings from the {{f:tables.count}} tables, and the semantic layer generates the Japanese from the frame. The parser layer has one interface and two implementations, `RuleParser` and `MatrixParser`, and the matrix implementation is meant to agree with the rule implementation wherever the matrix can represent the parse. `RuleParser` is a bounded transducer: a {{f:parser.window}}-token window, a frame stack of depth at most {{f:parser.max_depth}}, at most {{f:parser.width}} live readings resolved within {{f:parser.hold}} tokens, and no rewinds. The rule parser refuses an object-form or reflexive pronoun as a subject and a finite verb that disagrees in number with a pronoun or plural subject. `MatrixParser` runs the same machine as {{f:cell.layers}} sparse ReLU layers per step, with the noun-phrase module, the PP machine, the predicate and the copula translated by hand and the rest of the clause-level control extracted from traces of the rule parser. Two feedback edges cross the layers: the noun's use-verb rewrites the verb (she took a rest → 彼女は休んだ), and the parser's state decides which reading of a token is used (i rest → 休む, i need a rest → 休息). The second edge is the lexical layer's part inside the cell: the table rows that join the frame's verb with the window — the idiom tails, the particle after a verb, the use nouns — are AND units over the verb block and the window's table keys, ORed into the hit predicates the rules read, so the runner hands the cell the window alone.

## The cell

![The cell](docs/cell.svg)

![The cell, unit by unit](docs/cell_neurons.svg)

The compiled cell is a recurrent network: the state is a register file of one-hot blocks (the control point, the slots of the frame, the pending preposition, the shift and emit units, and the output morpheme), the input is one-hot over the readings of the window, and one step is {{f:cell.layers}} ReLU layers followed by a readout of zeros and ones into the next state. Every weight is plus or minus one, except a large negative weight that silences a rule whose literal the cell cannot see, and every bias is an integer threshold. The second picture is the same cell unit by unit, in the textbook style: circles are units, columns are the layers, arrows are the weights, one write into the object block is traced in colour with its mirror in the decoder — a printing position, the emit unit, one slot read and one form lookup into the output — in orange, and the grey tag slots hold the outer predicate while a that-clause is read in the same slots. A slot write is a gated copy: the rules that write a block are ORed into a block gate and, per source, a source gate; the next layer holds one write unit per value, which fires when the source gate and that source value are both on, and one keep unit per value, which the block gate cuts; and the readout sums keep and write back into the block, so one gate fans a rule out over the values of a block as a demultiplexer and clears the old value in the same step. The decoder is the same cell run backwards, sharing one state and one set of layers with the encoder: reading rules consume a token and write slots, printing rules read slots and emit a token, and the control block holds the reading positions and then the printing positions, so the end of reading is the first printing position. Every mechanism of the encoder has a mirror in the decoder — the shift unit that advances the read head mirrors an emit unit that advances the print head, a write gate that fans a rule out over the values of a block mirrors a read gate that selects a block onto the output, the USE lookup that rewrites the verb from its noun mirrors a form lookup that selects the verb's stem form, the suffix following as a constant the flags choose, a reading step that consumes no token mirrors a printing step that emits none, and a copy with no source mirrors a read with nothing in the block or a form the verb has not, all of which refuse. The output is one morpheme per step, a one-hot over slot values, verb and adjective stem forms, particles, suffixes and punctuation, summed by the readout from the read and lookup units, and an output lexical layer maps each unit to its string the way the input lexical layer maps each string to its readings. Inner clauses need no second bank of slots: a that-clause is read in the one frame — at its boundary the print head emits the outer prefix and the outer predicate moves into a few tag slots, every other slot being cleared by the gate that writes it; the clause is read into the cleared slots, printed when it closes and cleared again, and the tags move back for the outer predicate; a clause that finds the tags in use is refused rather than mis-parsed, so the cell holds one that-clause deep. The printing rules are written in the normal form of the noun-phrase module — per position, an ordered decision over slot predicates whose terminals emit one token: a constant morpheme, a slot's value, or a form of the verb or the adjective from the morphology table — and the `printer` action checks them against `generate()` on every sentence the rule parser parses and on a sweep of {{f:printer.sweep.frames|,}} frames over the predicates they test, a that-clause under a reporting verb and around a fixed inner clause among them, so the printer is one more chain of the lattice, agreeing with, refusing or misfiring against the rule generator on every sentence. The compiled cell runs unchanged as a PyTorch module whose {{f:cell.layers}} layers are sparse matrices held as fixed buffers: the `torch` action steps the scipy runner and the torch runner side by side on every dev and test sentence, reading and then printing, and their state vectors agree in single precision at every one of the {{f:torch.steps|,}} steps compared, the read head and the print head being the only operations outside the matrices.

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
│   ├── architecture.svg              the architecture diagram, as rendered
│   ├── cell.puml                     the cell as a recurrent network, with one slot write drawn out, source
│   ├── cell.svg                      the cell diagram, as rendered
│   └── cell_neurons.svg              the cell unit by unit, drawn by harness/render_neurons.py
├── harness/
│   ├── adjudications.json            human verdicts on outputs: right, wrong, or ungrammatical input
│   ├── baseline.json                 every item of the lattice at the last accepted commit, metrics as views
│   ├── check_diagram.py              the diagram files against their lock
│   ├── check_readme.py               README.md against the template, the facts and the appendix
│   ├── check_schemas.py              tables, corpus lock and baseline against their JSON Schemas
│   ├── check_todo.py                 TODO.yaml: schema, pins, acceptance tests run, stubs exercised
│   ├── check_tree.py                 this file tree against the tracked files
│   ├── corpus.lock.json              the pinned corpus: commit, sha256 and line count per file
│   ├── eval_matrix.py                action: hybrid cell compiled on dev+test, cross-tab vs the rule parser
│   ├── eval_negatives.py             action: published ungrammatical sentences refused, their controls accepted
│   ├── eval_np.py                    action: np_module against parse_np over all windows
│   ├── eval_printer.py               action: the printing rules against generate() on every parsed sentence and on a sweep
│   ├── eval_rules.py                 action: coverage and char-BLEU of the transducer
│   ├── eval_suites.py                action: the acceptance suites, each case against its expectation
│   ├── eval_torch.py                 action: the scipy and torch runners stepped side by side on dev+test, states compared
│   ├── facts.py                      the facts the README may cite
│   ├── lattice.py                    the item order: chains, levels, comparison, frontier
│   ├── negatives.json                the ungrammatical sentences, their controls, phenomena and sources
│   ├── probe_sandbox.py              a probe action for the sandbox test
│   ├── render_diagram.py             renders docs/*.puml with the pinned PlantUML jar
│   ├── render_neurons.py             draws docs/cell_neurons.svg with nothing but Python
│   ├── run.py                        the action cache, sandbox, baseline comparison, corpus fetch
│   ├── suites/
│   │   └── *.json                    one acceptance suite per residual class
│   ├── test_acceptance.py            one acceptance test per item of TODO.yaml, failing until the item is done
│   ├── test_claims.py                the tests the README's claims are pinned to
│   └── trace_rules.py                step the hybrid rule set through one sentence
├── schemas/
│   ├── tables/
│   │   ├── *.schema.json             one JSON Schema per table's row
│   │   └── integrity.json            uniqueness and cross-table references
│   ├── adjudications.schema.json
│   ├── baseline.schema.json
│   ├── corpus-lock.schema.json
│   ├── diagram-lock.schema.json
│   ├── integrity.schema.json         the format of integrity.json
│   ├── negatives.schema.json
│   ├── suite.schema.json
│   └── todo.schema.json
├── tables/
│   └── *.tbl                         the tables, each a small DSL file with its own header
├── .gitattributes
├── .gitignore
├── .ignore                           keeps search tools out of archive/
├── .python-version
├── CLAUDE.md                         the rules for working here: read first, pin every claim, keep everything reproducible
├── cell_torch.py                     the compiled cell as a PyTorch module of sparse fixed buffers
├── gen_module.py                     generate() translated by hand into printing rules: per position, a decision over slot predicates
├── LICENSE
├── README.md                         rendered from docs/README.tmpl.md — do not edit by hand
├── np_module.py                      parse_np translated by hand: derived predicates + rules, complete guards
├── parser.py                         the parser interface and its two implementations
├── parser_matrix.py                  the compiler to matrices and the MatrixParser
├── parser_rules.py                   the rule parser
├── pp_module.py                      tail_pp_adv translated by hand: the PP machine as ordered decisions with complete guards
├── predicate_module.py               predicate and copula_predicate translated by hand, the same way
├── pyproject.toml
├── rules_ja.py                       lexical layer, frame, generator, translate()
├── run_rules.py                      evaluation on dev/test with samples
├── stubs.py                          the interfaces of work not done yet, for the acceptance tests to be written against
├── tables.py                         table loader and record view
├── TODO.yaml                         the open items: context pinned like this README, stubs, an acceptance test each
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

Every commit runs the harness on the staged tree through the pre-commit hook, and the commit is refused if any item of the lattice moves down or an unjudged output changes without a verdict. Each action declares its inputs and runs in a sandbox that holds only those inputs, cached under `.cache/actions/` by the hash of the inputs, `uv.lock` and the interpreter. Every dependency is pinned with its sha256 in `uv.lock` ({{f:deps.count}} packages). The corpus is pinned to commit {{f:corpus.commit}} with a sha256 and line count per file, and is not vendored. Every table row, the corpus lock and the baseline are validated against JSON Schemas (draft {{f:schema.draft}}), with uniqueness and cross-table references declared in `schemas/tables/integrity.json`, and every sentence the baseline cites is verified against the pinned corpus line. The file tree above is compared with the tracked files. The architecture and cell diagrams are rendered from `docs/architecture.puml` and `docs/cell.puml` by PlantUML {{f:plantuml.version}}, pinned by the jar's sha256, the neuron drawing by `harness/render_neurons.py` with nothing but Python, and each committed SVG must match the hash recorded at render time and, for the drawing, be exactly what the script emits. Every number in this README is rendered from a named fact, and every sentence is either pinned to a test or declared a hypothesis in the appendix below. Retired files leave the tree, and what is worth reading later goes into `archive/` as a compressed blob that no text search sees into. Every open item of the plan is in `TODO.yaml`: its context is statements pinned like this README's, it names an acceptance test in `harness/test_acceptance.py` that must fail while the item is open and pass once it is done, and every stub those tests are written against is exercised by them. `CLAUDE.md` states the rules every agent and human follows here, and every file and check it names exists.

The `rules` action measures coverage and char-BLEU of the transducer on dev and test and records every output. The `matrix` action compiles the hybrid cell on dev and test, runs it on every sentence — reading, then printing — and cross-tabulates its output against the rule transducer's. The `np` action checks `np_module` against `parse_np` over every window of at most {{f:np.max_len}} tokens, or {{f:np.full_len}} with `--full`. The `negatives` action runs the {{f:negatives.count}} ungrammatical sentences of `harness/negatives.json`, taken or adapted from the syntax literature, asserts that the rule parser refuses every one and accepts every grammatical control, and counts what the matrix does with them. The `torch` action compiles the same cell, runs it under PyTorch as `cell_torch.py` builds it, and counts the sentences on which the torch runner's state ever differs from the scipy runner's. The `printer` action steps the printing rules of `gen_module.py` on the frame of every parsed sentence and on the sweep, and records what they print.

## The lattice

The state of the project is a product of small chains, one per item, and a change is an improvement exactly when no item moves down. A corpus sentence is WRONG, REFUSED or RIGHT, where RIGHT is the reference or an adjudicated acceptable output and an input adjudicated ungrammatical has refusal as its top; an output nobody has judged is UNKNOWN, below RIGHT but incomparable with the rest, so that a change away from it demands a verdict in `harness/adjudications.json`. The matrix parser on a sentence is MISFIRE, GAP or AGREE against the rule parser, and so are the printing rules against the generator; a negative is ACCEPTED or REFUSED, a control is REFUSED or ACCEPTED, the exhaustive NP check and the printer's sweep are MISMATCHES or MATCH, and the torch run is DIVERGING or EXACT. Coverage and quality are views of this state, not gates. The frontier is the set of items below the top of their chain, grouped by what blocks them, and `uv run python harness/run.py frontier` prints it from the committed baseline.

Today the {{f:lattice.items|,}} items stand at: sentences RIGHT {{f:lattice.sentence.right}}, UNKNOWN {{f:lattice.sentence.unknown}}, REFUSED {{f:lattice.sentence.refused}}, WRONG {{f:lattice.sentence.wrong}}; matrix AGREE {{f:lattice.matrix.agree}}, GAP {{f:lattice.matrix.gap}}, MISFIRE {{f:lattice.matrix.misfire}}; negatives REFUSED {{f:lattice.negative.refused}} of {{f:negatives.count}} and controls ACCEPTED {{f:lattice.control.accepted}}; {{f:lattice.adjudications}} adjudications so far. The frontier holds {{f:lattice.frontier.items|,}} items in {{f:lattice.frontier.groups}} groups.

## Acceptance suites

Each residual class of the second-round review has an acceptance suite under `harness/suites/`: sentences with the exact Japanese the transducer must produce, every case an item of the lattice, and the class is closed when every case is RIGHT. The suites hold {{f:suites.cases}} cases in {{f:suites.classes}} classes, {{f:suites.right}} right, {{f:suites.closed}} classes closed.

| class | what it covers | right / cases | status |
| --- | --- | --- | --- |
| B | {{f:suite.B.title}} | {{f:suite.B.right}} / {{f:suite.B.cases}} | {{f:suite.B.status}} |
| C | {{f:suite.C.title}} | {{f:suite.C.right}} / {{f:suite.C.cases}} | {{f:suite.C.status}} |
| D | {{f:suite.D.title}} | {{f:suite.D.right}} / {{f:suite.D.cases}} | {{f:suite.D.status}} |
| E | {{f:suite.E.title}} | {{f:suite.E.right}} / {{f:suite.E.cases}} | {{f:suite.E.status}} |
| F | {{f:suite.F.title}} | {{f:suite.F.right}} / {{f:suite.F.cases}} | {{f:suite.F.status}} |
| G | {{f:suite.G.title}} | {{f:suite.G.right}} / {{f:suite.G.cases}} | {{f:suite.G.status}} |
| H | {{f:suite.H.title}} | {{f:suite.H.right}} / {{f:suite.H.cases}} | {{f:suite.H.status}} |
| I | {{f:suite.I.title}} | {{f:suite.I.right}} / {{f:suite.I.cases}} | {{f:suite.I.status}} |
| K | {{f:suite.K.title}} | {{f:suite.K.right}} / {{f:suite.K.cases}} | {{f:suite.K.status}} |
| L | {{f:suite.L.title}} | {{f:suite.L.right}} / {{f:suite.L.cases}} | {{f:suite.L.status}} |
| M | {{f:suite.M.title}} | {{f:suite.M.right}} / {{f:suite.M.cases}} | {{f:suite.M.status}} |

## Where it stands

The rule transducer translates {{f:rules.test.covered}} of {{f:rules.test.n}} test sentences ({{f:rules.test.coverage_pct}}%) and {{f:rules.dev.covered}} of {{f:rules.dev.n}} dev sentences ({{f:rules.dev.coverage_pct}}%), with char-BLEU {{f:rules.test.bleu_covered|.1f}} and {{f:rules.dev.bleu_covered|.1f}} inside coverage. On the {{f:matrix.sentences}} dev and test sentences, the cell translates {{f:matrix.identical}} exactly as the rule transducer does; it misfires silently on {{f:matrix.silent_misfires}} ({{f:matrix.wrong_frame}} different translations and {{f:matrix.matrix_parses_refused}} translations of sentences the rules refuse) and refuses {{f:matrix.refused_parseable}} sentences the rules translate. The compiled cell has {{f:cell.rules}} rules ({{f:cell.rules.outer}} extracted, {{f:cell.rules.entry}} entry, {{f:cell.rules.exit}} exit, {{f:cell.rules.fail}} refusal, {{f:cell.rules.module}} from the noun-phrase module, {{f:cell.rules.clause}} from the clause modules and their hand-overs, {{f:cell.rules.print}} printing, {{f:cell.rules.flush}} of the clause flush), {{f:cell.state_units|,}} state units, {{f:cell.input_units|,}} input units and {{f:cell.nnz|,}} nonzeros. `np_module` agrees with `parse_np` on all {{f:np.cases|,}} windows of at most {{f:np.max_len}} tokens over {{f:np.alphabet}} token classes. Its two adjective slots refuse an adverb followed by two adjectives, where `parse_np` concatenates. An of-phrase is the module run once more on the role's of-phrase slots, printed before the head with の, and a second of-phrase inside it is refused; a frame whose string came from an inner clause is marked folded and stays outside the cell's scope. The rule parser refuses all {{f:negatives.count}} negatives and accepts all {{f:negatives.controls_accepted}} controls; the matrix refuses {{f:negatives.matrix_refused}} of the negatives and accepts {{f:negatives.controls_matrix}} of the controls, the rest falling to its extracted clause-level rules. The {{f:printer.rules|,}} printing rules at {{f:printer.positions}} positions print {{f:printer.agree}} of the {{f:printer.parsed}} parsed sentences exactly as `generate()` does and refuse the other {{f:printer.gap}}, whose frames hold a register the cell has no block for; on the sweep they print {{f:printer.sweep.match|,}} frames as `generate()` does, refuse {{f:printer.sweep.refused}}, and mismatch on {{f:printer.sweep.mismatches}}. The PP machine, `predicate` and `copula_predicate` are translated by hand in `pp_module.py` and `predicate_module.py` — per position an ordered decision whose last rule is the refusal — and spliced into the cell at every activation of those methods, so no rule extracted from traces is left for them; what is still extracted is the start of the sentence, the question paths, the verb phrase, its objects and the end. A that-clause after a verb is read in the one frame and printed as the rules print it; one after a copula (I am sure that he will come) opens inside the copula module, which refuses it. Hypothesis: the gaps that remain are a that-clause after a copula or inside another, folded frames and the paths still extracted, and translating those paths by hand would leave the cell no induced guard.

## License

The code is MIT-licensed, and the corpus is the Tanaka Corpus via small_parallel_enja under its own terms.

## Appendix: claims and their pins

Every sentence above is quoted here with what pins it: a test in `harness/test_claims.py`, an action or a check in `harness/run.py` (all run by the pre-commit hook), or `Hypothesis`. `harness/check_readme.py` verifies the quotes, the pins and the coverage.

- "RNNTranslator is an English-to-Japanese translator written as a circuit, not trained: a hand-written rule transducer whose parser is compiled, module by module, into sparse ReLU matrices that run one token at a time.": Pinned by `test_cell_weights_are_named_integers`
- "Hypothesis: a language is a small circuit of tables and bounded control, and such a circuit can be written down, verified exhaustively, and compiled into weights whose every nonzero has a name.": Hypothesis
- "It is evaluated on the Tanaka-corpus dev ({{f:rules.dev.n}} sentences) and test ({{f:rules.test.n}} sentences) splits of small_parallel_enja, pinned by commit and hash.": Pinned by `test_corpus_pinned`
- "The lexical layer merges multiword tokens and gives each token its readings from the {{f:tables.count}} tables, and the semantic layer generates the Japanese from the frame.": Pinned by `test_layers_roundtrip`
- "The parser layer has one interface and two implementations, `RuleParser` and `MatrixParser`, and the matrix implementation is meant to agree with the rule implementation wherever the matrix can represent the parse.": Pinned by `test_parser_interface_two_implementations`
- "`RuleParser` is a bounded transducer: a {{f:parser.window}}-token window, a frame stack of depth at most {{f:parser.max_depth}}, at most {{f:parser.width}} live readings resolved within {{f:parser.hold}} tokens, and no rewinds.": Pinned by `test_rule_parser_bounds`
- "The rule parser refuses an object-form or reflexive pronoun as a subject and a finite verb that disagrees in number with a pronoun or plural subject.": Pinned by `test_subject_case_and_agreement`
- "`MatrixParser` runs the same machine as {{f:cell.layers}} sparse ReLU layers per step, with the noun-phrase module, the PP machine, the predicate and the copula translated by hand and the rest of the clause-level control extracted from traces of the rule parser.": Pinned by `test_matrix_cell_shape`
- "Two feedback edges cross the layers: the noun's use-verb rewrites the verb (she took a rest → 彼女は休んだ), and the parser's state decides which reading of a token is used (i rest → 休む, i need a rest → 休息).": Pinned by `test_feedback_edges`
- "The second edge is the lexical layer's part inside the cell: the table rows that join the frame's verb with the window — the idiom tails, the particle after a verb, the use nouns — are AND units over the verb block and the window's table keys, ORed into the hit predicates the rules read, so the runner hands the cell the window alone.": Pinned by `test_verb_hits_are_layer_zero`
- "The compiled cell is a recurrent network: the state is a register file of one-hot blocks (the control point, the slots of the frame, the pending preposition, the shift and emit units, and the output morpheme), the input is one-hot over the readings of the window, and one step is {{f:cell.layers}} ReLU layers followed by a readout of zeros and ones into the next state.": Pinned by `test_matrix_cell_shape`
- "Every weight is plus or minus one, except a large negative weight that silences a rule whose literal the cell cannot see, and every bias is an integer threshold.": Pinned by `test_cell_weights_are_named_integers`
- "The second picture is the same cell unit by unit, in the textbook style: circles are units, columns are the layers, arrows are the weights, one write into the object block is traced in colour with its mirror in the decoder — a printing position, the emit unit, one slot read and one form lookup into the output — in orange, and the grey tag slots hold the outer predicate while a that-clause is read in the same slots.": Pinned by `check_diagram`
- "A slot write is a gated copy: the rules that write a block are ORed into a block gate and, per source, a source gate; the next layer holds one write unit per value, which fires when the source gate and that source value are both on, and one keep unit per value, which the block gate cuts; and the readout sums keep and write back into the block, so one gate fans a rule out over the values of a block as a demultiplexer and clears the old value in the same step.": Pinned by `test_slot_write_circuit`
- "The decoder is the same cell run backwards, sharing one state and one set of layers with the encoder: reading rules consume a token and write slots, printing rules read slots and emit a token, and the control block holds the reading positions and then the printing positions, so the end of reading is the first printing position.": Pinned by `test_decoder_is_the_cell_run_backwards`
- "Every mechanism of the encoder has a mirror in the decoder — the shift unit that advances the read head mirrors an emit unit that advances the print head, a write gate that fans a rule out over the values of a block mirrors a read gate that selects a block onto the output, the USE lookup that rewrites the verb from its noun mirrors a form lookup that selects the verb's stem form, the suffix following as a constant the flags choose, a reading step that consumes no token mirrors a printing step that emits none, and a copy with no source mirrors a read with nothing in the block or a form the verb has not, all of which refuse.": Pinned by `test_decoder_mirrors_encoder`
- "The output is one morpheme per step, a one-hot over slot values, verb and adjective stem forms, particles, suffixes and punctuation, summed by the readout from the read and lookup units, and an output lexical layer maps each unit to its string the way the input lexical layer maps each string to its readings.": Pinned by `test_output_is_one_hot_morphemes`
- "Inner clauses need no second bank of slots: a that-clause is read in the one frame — at its boundary the print head emits the outer prefix and the outer predicate moves into a few tag slots, every other slot being cleared by the gate that writes it; the clause is read into the cleared slots, printed when it closes and cleared again, and the tags move back for the outer predicate; a clause that finds the tags in use is refused rather than mis-parsed, so the cell holds one that-clause deep.": Pinned by `test_inner_clause_in_one_frame`
- "The printing rules are written in the normal form of the noun-phrase module — per position, an ordered decision over slot predicates whose terminals emit one token: a constant morpheme, a slot's value, or a form of the verb or the adjective from the morphology table — and the `printer` action checks them against `generate()` on every sentence the rule parser parses and on a sweep of {{f:printer.sweep.frames|,}} frames over the predicates they test, a that-clause under a reporting verb and around a fixed inner clause among them, so the printer is one more chain of the lattice, agreeing with, refusing or misfiring against the rule generator on every sentence.": Pinned by `printer`
- "The compiled cell runs unchanged as a PyTorch module whose {{f:cell.layers}} layers are sparse matrices held as fixed buffers: the `torch` action steps the scipy runner and the torch runner side by side on every dev and test sentence, reading and then printing, and their state vectors agree in single precision at every one of the {{f:torch.steps|,}} steps compared, the read head and the print head being the only operations outside the matrices.": Pinned by `torch`
- "Every commit runs the harness on the staged tree through the pre-commit hook, and the commit is refused if any item of the lattice moves down or an unjudged output changes without a verdict.": Pinned by `test_precommit_hook_runs_check`
- "Each action declares its inputs and runs in a sandbox that holds only those inputs, cached under `.cache/actions/` by the hash of the inputs, `uv.lock` and the interpreter.": Pinned by `test_sandbox_holds_only_declared_inputs`
- "Every dependency is pinned with its sha256 in `uv.lock` ({{f:deps.count}} packages).": Pinned by `test_dependencies_hash_pinned`
- "The corpus is pinned to commit {{f:corpus.commit}} with a sha256 and line count per file, and is not vendored.": Pinned by `test_corpus_pinned`
- "Every table row, the corpus lock and the baseline are validated against JSON Schemas (draft {{f:schema.draft}}), with uniqueness and cross-table references declared in `schemas/tables/integrity.json`, and every sentence the baseline cites is verified against the pinned corpus line.": Pinned by `check_schemas`
- "The file tree above is compared with the tracked files.": Pinned by `check_tree`
- "The architecture and cell diagrams are rendered from `docs/architecture.puml` and `docs/cell.puml` by PlantUML {{f:plantuml.version}}, pinned by the jar's sha256, the neuron drawing by `harness/render_neurons.py` with nothing but Python, and each committed SVG must match the hash recorded at render time and, for the drawing, be exactly what the script emits.": Pinned by `check_diagram`
- "Every number in this README is rendered from a named fact, and every sentence is either pinned to a test or declared a hypothesis in the appendix below.": Pinned by `check_readme`
- "Retired files leave the tree, and what is worth reading later goes into `archive/` as a compressed blob that no text search sees into.": Pinned by `test_archive_is_opaque`
- "Every open item of the plan is in `TODO.yaml`: its context is statements pinned like this README's, it names an acceptance test in `harness/test_acceptance.py` that must fail while the item is open and pass once it is done, and every stub those tests are written against is exercised by them.": Pinned by `check_todo`
- "`CLAUDE.md` states the rules every agent and human follows here, and every file and check it names exists.": Pinned by `test_claude_md_references_exist`
- "The `rules` action measures coverage and char-BLEU of the transducer on dev and test and records every output.": Pinned by `rules`
- "The `matrix` action compiles the hybrid cell on dev and test, runs it on every sentence — reading, then printing — and cross-tabulates its output against the rule transducer's.": Pinned by `matrix`
- "The `np` action checks `np_module` against `parse_np` over every window of at most {{f:np.max_len}} tokens, or {{f:np.full_len}} with `--full`.": Pinned by `np`
- "The `negatives` action runs the {{f:negatives.count}} ungrammatical sentences of `harness/negatives.json`, taken or adapted from the syntax literature, asserts that the rule parser refuses every one and accepts every grammatical control, and counts what the matrix does with them.": Pinned by `negatives`
- "The `torch` action compiles the same cell, runs it under PyTorch as `cell_torch.py` builds it, and counts the sentences on which the torch runner's state ever differs from the scipy runner's.": Pinned by `torch`
- "The `printer` action steps the printing rules of `gen_module.py` on the frame of every parsed sentence and on the sweep, and records what they print.": Pinned by `printer`
- "The rule transducer translates {{f:rules.test.covered}} of {{f:rules.test.n}} test sentences ({{f:rules.test.coverage_pct}}%) and {{f:rules.dev.covered}} of {{f:rules.dev.n}} dev sentences ({{f:rules.dev.coverage_pct}}%), with char-BLEU {{f:rules.test.bleu_covered|.1f}} and {{f:rules.dev.bleu_covered|.1f}} inside coverage.": Pinned by `rules`
- "On the {{f:matrix.sentences}} dev and test sentences, the cell translates {{f:matrix.identical}} exactly as the rule transducer does; it misfires silently on {{f:matrix.silent_misfires}} ({{f:matrix.wrong_frame}} different translations and {{f:matrix.matrix_parses_refused}} translations of sentences the rules refuse) and refuses {{f:matrix.refused_parseable}} sentences the rules translate.": Pinned by `matrix`
- "The compiled cell has {{f:cell.rules}} rules ({{f:cell.rules.outer}} extracted, {{f:cell.rules.entry}} entry, {{f:cell.rules.exit}} exit, {{f:cell.rules.fail}} refusal, {{f:cell.rules.module}} from the noun-phrase module, {{f:cell.rules.clause}} from the clause modules and their hand-overs, {{f:cell.rules.print}} printing, {{f:cell.rules.flush}} of the clause flush), {{f:cell.state_units|,}} state units, {{f:cell.input_units|,}} input units and {{f:cell.nnz|,}} nonzeros.": Pinned by `matrix`
- "`np_module` agrees with `parse_np` on all {{f:np.cases|,}} windows of at most {{f:np.max_len}} tokens over {{f:np.alphabet}} token classes.": Pinned by `np`
- "Its two adjective slots refuse an adverb followed by two adjectives, where `parse_np` concatenates.": Pinned by `test_np_two_adjective_slots`
- "An of-phrase is the module run once more on the role's of-phrase slots, printed before the head with の, and a second of-phrase inside it is refused; a frame whose string came from an inner clause is marked folded and stays outside the cell's scope.": Pinned by `test_np_of_phrase`
- "The rule parser refuses all {{f:negatives.count}} negatives and accepts all {{f:negatives.controls_accepted}} controls; the matrix refuses {{f:negatives.matrix_refused}} of the negatives and accepts {{f:negatives.controls_matrix}} of the controls, the rest falling to its extracted clause-level rules.": Pinned by `negatives`
- "The {{f:printer.rules|,}} printing rules at {{f:printer.positions}} positions print {{f:printer.agree}} of the {{f:printer.parsed}} parsed sentences exactly as `generate()` does and refuse the other {{f:printer.gap}}, whose frames hold a register the cell has no block for; on the sweep they print {{f:printer.sweep.match|,}} frames as `generate()` does, refuse {{f:printer.sweep.refused}}, and mismatch on {{f:printer.sweep.mismatches}}.": Pinned by `printer`
- "The PP machine, `predicate` and `copula_predicate` are translated by hand in `pp_module.py` and `predicate_module.py` — per position an ordered decision whose last rule is the refusal — and spliced into the cell at every activation of those methods, so no rule extracted from traces is left for them; what is still extracted is the start of the sentence, the question paths, the verb phrase, its objects and the end.": Pinned by `test_clause_modules_replace_extraction`
- "A that-clause after a verb is read in the one frame and printed as the rules print it; one after a copula (I am sure that he will come) opens inside the copula module, which refuses it.": Pinned by `test_inner_clause_in_one_frame`
- "Hypothesis: the gaps that remain are a that-clause after a copula or inside another, folded frames and the paths still extracted, and translating those paths by hand would leave the cell no induced guard.": Hypothesis
- "The state of the project is a product of small chains, one per item, and a change is an improvement exactly when no item moves down.": Pinned by `test_lattice_order`
- "A corpus sentence is WRONG, REFUSED or RIGHT, where RIGHT is the reference or an adjudicated acceptable output and an input adjudicated ungrammatical has refusal as its top; an output nobody has judged is UNKNOWN, below RIGHT but incomparable with the rest, so that a change away from it demands a verdict in `harness/adjudications.json`.": Pinned by `test_lattice_order`
- "The matrix parser on a sentence is MISFIRE, GAP or AGREE against the rule parser, and so are the printing rules against the generator; a negative is ACCEPTED or REFUSED, a control is REFUSED or ACCEPTED, the exhaustive NP check and the printer's sweep are MISMATCHES or MATCH, and the torch run is DIVERGING or EXACT.": Pinned by `test_lattice_order`
- "Coverage and quality are views of this state, not gates.": Pinned by `test_precommit_hook_runs_check`
- "The frontier is the set of items below the top of their chain, grouped by what blocks them, and `uv run python harness/run.py frontier` prints it from the committed baseline.": Pinned by `test_frontier_command`
- "Today the {{f:lattice.items|,}} items stand at: sentences RIGHT {{f:lattice.sentence.right}}, UNKNOWN {{f:lattice.sentence.unknown}}, REFUSED {{f:lattice.sentence.refused}}, WRONG {{f:lattice.sentence.wrong}}; matrix AGREE {{f:lattice.matrix.agree}}, GAP {{f:lattice.matrix.gap}}, MISFIRE {{f:lattice.matrix.misfire}}; negatives REFUSED {{f:lattice.negative.refused}} of {{f:negatives.count}} and controls ACCEPTED {{f:lattice.control.accepted}}; {{f:lattice.adjudications}} adjudications so far.": Pinned by `rules`
- "The frontier holds {{f:lattice.frontier.items|,}} items in {{f:lattice.frontier.groups}} groups.": Pinned by `rules`
- "Each residual class of the second-round review has an acceptance suite under `harness/suites/`: sentences with the exact Japanese the transducer must produce, every case an item of the lattice, and the class is closed when every case is RIGHT.": Pinned by `test_suites_are_items`
- "The suites hold {{f:suites.cases}} cases in {{f:suites.classes}} classes, {{f:suites.right}} right, {{f:suites.closed}} classes closed.": Pinned by `suites`
- "The code is MIT-licensed, and the corpus is the Tanaka Corpus via small_parallel_enja under its own terms.": Pinned by `test_license_is_mit`
