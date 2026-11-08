"""A probe action: lists what the sandbox holds (used by test_sandbox_holds_only_declared_inputs)."""
import os, json, sys
files = sorted(os.path.relpath(os.path.join(d, f), ".") for d, _, fs in os.walk(".") for f in fs if f != "result.json")
res = {"files": files, "env_keys": sorted(os.environ), "summary": f"probe: {len(files)} files"}
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "result.json"
json.dump(res, open(out, "w"))
