# Archive

Retired code, kept for the record and nothing else. Each blob is a compressed tarball, so no text search
(`grep -r`, `rg`, an IDE index) sees its contents; `.ignore` and `.gitattributes` mark the blobs as binary as well.
Files are retired when they are no longer reachable from the current iteration frontier and unlikely to be reused.
Everything is also recoverable from git history; the blob exists so the history can be read without checking out
old commits.

To look inside: `tar -tJf archive/<blob>.tar.xz` lists, `tar -xJf archive/<blob>.tar.xz -C /tmp` extracts. Each blob
carries a `MANIFEST.txt` describing its files.

## retired-2026-11-08.tar.xz

The fragment era, before the Tanaka-corpus transducer: the Dyck-2 RNN written weight by weight; the hand-written
RNN for a controlled English→Japanese fragment and its compile into a three-layer sparse ReLU cell with a lower
bound on nonzeros; the noun-centric fragment (nouns carry their use-verb) and its compile; the 1-minimality witness
search; a weight-space edit made from the raw dump alone; and the opposite experiment, weights written from 1,000
sentence pairs with no linguistics at all. These established the method (unit equations → matrices, verified by
enumeration) that the transducer and its compiler now follow at corpus scale.

Also the transducer-era files the harness superseded: the pre-layered-parser checkpoint and its zero-diff regression
script (replaced by `harness/baseline.json`), the extraction-only compiler and its reports (replaced by the hybrid
compiler with hand-translated modules), the phase enumeration and own-phase colouring measurements (their results are
in the compile tracker doc), and the first versions of the exhaustive NP check, the cross-tab and the rule stepper,
which live on as `harness/eval_np.py`, `harness/eval_matrix.py` and `harness/trace_rules.py`.
