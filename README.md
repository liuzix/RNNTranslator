# RNNTranslator

An English → Japanese translator written as a circuit, not trained: a hand-written rule transducer whose
parser is being compiled, module by module, into sparse ReLU matrices that run one token at a time.

The thesis under test: a trained dense network is the wrong representation to inspect; a language is a
small circuit of tables and bounded control, and such a circuit can be written down, verified exhaustively,
and then *compiled* into weights whose every nonzero has a name.

Evaluated on the Tanaka-corpus dev (496) and test (499) splits of
[small_parallel_enja](https://github.com/odashi/small_parallel_enja).

## Layout

| Path | What it is |
| --- | --- |
| `rules_ja.py` | The translator: lexical layer (`merge_multiword`, `classify`), the frame (`Clause`), the generator (`generate`, `predicate_form`, conjugation paradigms). `translate(tokens)` = merge → parse → generate. |
| `tables.py`, `tables/*.tbl` | The 16 tables, each a small DSL file: lexicon (1,148 rows), idioms, verb classes, complements, use-verbs, prepositions, pronouns, determiners, wh-words, modals, numbers, contractions, connectives, duration, adjectives, godan stems. |
| `parser_rules.py` | The rule parser: a bounded transducer over a 4-token window with a frame stack of depth 2, an ambiguity budget (2 live readings resolved within 4 tokens, else `Overflow`), no rewinds. |
| `parser.py` | The parser interface (`ParserInterface.parse(tokens) -> Frame`) and its two implementations: `RuleParser` (the reference) and `build_matrix_parser` (the compiled cell). |
| `parser_matrix.py` | The compiler to matrices: traces the rule parser, induces the clause-level transition rules, adds the hand-translated modules, and emits a five-layer sparse ReLU cell (`MatrixParser`). |
| `np_module.py` | The first hand-translated parser module: `parse_np` as derived predicates + 65 rules over 8 states, with complete guards and a FAIL exit for everything out of scope. |
| `run_rules.py` | Evaluation harness: coverage and char-BLEU on dev/test, failures by cause. |
| `checks/` | The verification scripts: regression against a checkpoint, table round-trip, phase enumeration, the exhaustive NP check, the hybrid cross-tab. |
| `history/` | Earlier stages: the Dyck-2 RNN, the hand-written fragment RNN and its sparse compile, the noun-centric fragment, the data-only basis experiment, and the pre-hybrid compiler. |

## Setup

```
pip install -r requirements.txt
git clone https://github.com/odashi/small_parallel_enja data/small_parallel_enja
python run_rules.py            # coverage + BLEU of the rule transducer on dev
python checks/crosstab.py      # compile the matrix parser and cross-tabulate it against the rule parser
python checks/np_exhaustive.py # the exhaustive check of np_module against parse_np (~4 min)
```

## Where it stands (2026-11-08)

Rule transducer: test coverage 80% (400/499), dev 62.5% (310/496); char-BLEU inside coverage ≈ 35.

Matrix parser (dev+test, 995 sentences, reference = the bounded rule parser):

| | extraction only | hybrid: `parse_np` as a module |
| --- | --- | --- |
| both parse, identical | 461 | 496 |
| both parse, different frame | 72 | 15 |
| rules refuse, matrix parses | 44 | 10 |
| rules parse, matrix refuses | 177 | 199 |
| both refuse | 241 | 275 |
| rules in the cell | 946 | 1,458 |
| state / input units | 6,443 / 1,518 | 12,592 / 1,577 |
| nonzeros | 52,069 | 180,065 |

The cell: state = one-hot control states + one-hot slot blocks + the pending preposition + a `shift` unit
(the parser tells the window when to advance); input = window predicates for 4 slots + copy sources for
slot 0. Five layers per step: register predicates and derived conjunctions (L1), counts and negations (L2),
rules as AND units (L3), next control, shift and write gates (L4), keep/write/crash per slot value (L5), then
the new state as a sum. Weights are in {−2, −1, +1}.

`np_module.py` was checked exhaustively against `Parser.parse_np`: 2,315,036 window cases (83 token
classes, sequences of 1–3 plus the end, four call contexts), 0 mismatches; 1,827,904 length-4 cases over 26
NP-internal tokens found one divergence, all of one shape (adverb + adjective + adjective + noun), where the
module's two adjective slots are full and it refuses loudly where `parse_np` concatenates — the measured
capacity, not a misfire. The remaining silent misfires
(25) are all in the extracted clause-level rules, whose guards are only as complete as the corpus; the next
modules to translate by hand are the PP machine, `predicate` and `copula_predicate`.

## License

MIT (see `LICENSE`). The corpus is the Tanaka Corpus via small_parallel_enja, under its own terms; it is
not vendored here.
