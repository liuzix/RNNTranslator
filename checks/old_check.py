import os, sys, collections, importlib.util; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("parser_matrix_old", "history/parser_matrix.before_hybrid.py")
PM = importlib.util.module_from_spec(spec); sys.modules["parser_matrix_old"] = PM; spec.loader.exec_module(PM)
import rules_ja as R, run_rules as H
from parser_rules import Parser, ParseFail
import parser as PI; rp = PI.RuleParser()
pairs = H.load("dev") + H.load("test")
sents = [R.merge_multiword(en if isinstance(en, list) else en.split()) for en, ja in pairs]
rows = []
for toks in sents:
    try: Parser(list(toks), []).parse(); r, _ = PM.event_rows_pred(list(toks), []); rows += r
    except (ParseFail, PM.OutOfScope):
        try: rows += PM.fail_rows(list(toks))
        except Exception: pass
    except Exception: pass
mp = PM.MatrixParser(PM.compile_matrices(PM.induce(rows)))
print(f"OLD extraction-only: state {mp.nS} input {mp.nX} nnz {mp.nnz()} rules {len(mp.c['rules'])}")
cnt = collections.Counter()
for toks in sents:
    try: want = rp.translate(toks)
    except ParseFail: want = None
    try: got = R.generate(mp.parse(list(toks)))
    except Exception as e: got = ("FAIL",)
    if want is None: key = "rules refuse / matrix refuses" if isinstance(got, tuple) else "rules refuse / matrix parses"
    else: key = "both parse, identical" if got == want else ("rules parse / matrix refuses" if isinstance(got, tuple) else "both parse, different")
    cnt[key] += 1
print(dict(cnt.most_common()))
