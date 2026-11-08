import os, sys, time, collections, importlib.util; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rules_ja as R, run_rules as H
spec = importlib.util.spec_from_file_location("old", "history/rules_ja.checkpoint1.py")
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
def outputs(M, pairs):
    res = []
    for en, ja in pairs:
        toks = en if isinstance(en, list) else en.split()
        try: res.append(("ok", M.translate(toks)))
        except M.ParseFail as e: res.append(("fail", str(e)))
        except RuntimeError as e: res.append(("BUFFER", str(e)))
    return res
for split in ("test", "dev"):
    pairs = H.load(split)
    refs = [ja if isinstance(ja, str) else "".join(ja) for en, ja in pairs]
    t0 = time.time(); new = outputs(R, pairs); t1 = time.time(); old = outputs(OLD, pairs)
    ok_new = [i for i, (s, _) in enumerate(new) if s == "ok"]; ok_old = [i for i, (s, _) in enumerate(old) if s == "ok"]
    lost = [i for i in ok_old if new[i][0] != "ok"]; gained = [i for i in ok_new if old[i][0] != "ok"]
    changed = [i for i in ok_new if old[i][0] == "ok" and old[i][1] != new[i][1]]
    overflow = collections.Counter(new[i][1].split(" at ")[0] for i in range(len(new)) if new[i][0] == "fail" and new[i][1].startswith("OVERFLOW"))
    buf = [i for i in range(len(new)) if new[i][0] == "BUFFER"]
    bleu_new = H.char_bleu([new[i][1] for i in ok_new], [refs[i] for i in ok_new]); bleu_old = H.char_bleu([old[i][1] for i in ok_old], [refs[i] for i in ok_old])
    bn = bleu_new[0] if isinstance(bleu_new, tuple) else bleu_new; bo = bleu_old[0] if isinstance(bleu_old, tuple) else bleu_old
    print(f"== {split}: checkpoint {len(ok_old)}/{len(pairs)} -> layered {len(ok_new)}/{len(pairs)}   BLEU in coverage {bo:.1f} -> {bn:.1f}   ({t1-t0:.0f}s)")
    print(f"   lost {len(lost)}, gained {len(gained)}, same-parse-different-output {len(changed)}, overflow {dict(overflow)}, buffer violations {len(buf)}")
    for i in lost[:12]: print("   LOST  ", " ".join(pairs[i][0]) if isinstance(pairs[i][0], list) else pairs[i][0], "->", new[i][1])
    for i in gained[:6]: print("   GAINED", " ".join(pairs[i][0]) if isinstance(pairs[i][0], list) else pairs[i][0], "->", new[i][1])
    for i in changed[:6]: print("   CHANGED", " ".join(pairs[i][0]) if isinstance(pairs[i][0], list) else pairs[i][0], "|", old[i][1], "->", new[i][1])
    for i in buf[:5]: print("   BUFFER", " ".join(pairs[i][0]) if isinstance(pairs[i][0], list) else pairs[i][0], new[i][1])
