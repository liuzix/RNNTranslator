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
| `harness/` | The hermetic checks and their runner (`run.py`: content-addressed action cache, sandboxed runs, the committed baseline, the pinned corpus), plus `trace_rules.py`, the rule stepper for debugging a disagreement. |
| `archive/` | Retired code as compressed blobs that no text search sees into, each with a manifest; `archive/README.md` says what is in them and why. |

## Setup

Dependencies are resolved by [uv](https://docs.astral.sh/uv/) into `uv.lock` with a sha256 for every artifact;
`uv sync --locked` refuses a stale lock. The corpus is pinned to a commit and to file hashes in `harness/corpus.lock`.

```
uv sync --locked                                   # the environment, exactly as locked
uv run python harness/run.py fetch-corpus          # the pinned corpus commit into .cache/corpus/, hashes verified
uv run python harness/run.py install-hooks         # git runs the checks on every commit
uv run python run_rules.py                         # coverage + BLEU of the rule transducer on dev, with samples
uv run python harness/run.py check                 # the three checks, cached by content, compared with the baseline
```

## The harness

Three checks, each an *action* with declared inputs: `rules` (coverage and char-BLEU of the transducer on dev and
test, every output recorded), `matrix` (the hybrid cell compiled on dev+test and cross-tabulated against the rule
parser), `np` (`np_module` against `parse_np` over every window up to length 2; `--full` goes to length 3, ~4 min).
An action's key is the hash of its inputs, the lock file and the interpreter; it runs in a sandbox holding only
those inputs, with a fixed hash seed and a minimal environment, and its result is cached under `.cache/actions/`.
Unchanged inputs cost nothing; nothing undeclared can leak in, so a cache hit is sound. This is a Bazel-style action
cache in 200 lines, with no Bazel.

`harness/baseline.json` holds the metrics and every rule output at the last accepted commit. The pre-commit hook
(`.githooks/pre-commit`) runs `check --staged` on the index, not the working tree, and refuses the commit if
coverage, BLEU-over-all, matrix agreement, silent misfires or NP mismatches regress against the baseline *as
staged*. An intended change is accepted with `uv run python harness/run.py baseline --staged` and staging the new
baseline with it; the hook also prints which sentences changed output.

Retired files leave the repo (git keeps them recoverable); what is worth reading later goes into `archive/` as a
compressed blob with a manifest, so searches of the live tree stay quiet.

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

(The table is the hand-measured cross-tab of Nov 8; `harness/baseline.json` carries the harness's own numbers, which
differ by a few sentences because the tree induction now breaks ties deterministically.)

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
