"""The README's file tree must match the repository: every tracked file appears exactly once (by name, or covered by one
glob entry such as `*.tbl`), every entry in the tree exists, directories are shown with a trailing slash.  Trailing
comments after two or more spaces are free text and are not compared.

    python harness/check_tree.py [--root DIR] [--staged]      compare (the index with --staged, else index + untracked files)
    python harness/check_tree.py --write                      regenerate the block from the tracked files, keeping the comments

The block in README.md sits between <!-- tree:begin --> and <!-- tree:end -->, as a fenced code block.
"""
import sys, os, re, subprocess, argparse, fnmatch, collections

BEGIN, END = "<!-- tree:begin -->", "<!-- tree:end -->"
PREFIX = re.compile(r"^[│├└─ ]*")

def tracked_files(repo, staged):
    args = ["git", "ls-files", "--cached"] + ([] if staged else ["--others", "--exclude-standard"])
    out = subprocess.run(args, cwd=repo, capture_output=True, text=True, check=True).stdout.split("\n")
    return sorted(set(p for p in out if p))

def read_block(readme):
    text = open(readme, encoding="utf-8").read()
    if BEGIN not in text or END not in text: sys.exit("README.md: no <!-- tree:begin --> … <!-- tree:end --> block")
    inner = text.split(BEGIN, 1)[1].split(END, 1)[0]
    lines = [l for l in inner.split("\n") if l.strip() and not l.strip().startswith("```")]
    return text, lines

def parse_tree(lines):
    """-> list of (path, is_dir, comment); the first line is the root"""
    entries = []; stack = []
    for raw in lines:
        m = PREFIX.match(raw); prefix = m.group(0); rest = raw[m.end():]
        depth = len(prefix) // 4
        parts = re.split(r" {2,}", rest, maxsplit=1)
        name = parts[0].strip(); comment = parts[1].strip() if len(parts) > 1 else ""
        if not entries: entries.append((".", True, comment)); stack = []; continue
        stack = stack[:depth - 1]
        is_dir = name.endswith("/")
        path = "/".join(stack + [name.rstrip("/")])
        entries.append((path, is_dir, comment))
        if is_dir: stack.append(name.rstrip("/"))
    return entries

def check(files, entries):
    problems = []
    covered = collections.Counter()
    for path, is_dir, _ in entries[1:]:
        if is_dir:
            if not any(f.startswith(path + "/") for f in files): problems.append(f"tree lists directory {path}/ but it holds no tracked file")
        elif "*" in path:
            hits = [f for f in files if fnmatch.fnmatch(f, path) and os.path.dirname(f) == os.path.dirname(path)]
            if not hits: problems.append(f"tree entry {path} matches no tracked file")
            for f in hits: covered[f] += 1
        else:
            if path not in files: problems.append(f"tree lists {path} but it is not tracked")
            else: covered[path] += 1
    for f in files:
        if covered[f] == 0: problems.append(f"tracked file {f} is not in the tree")
        elif covered[f] > 1: problems.append(f"tracked file {f} is matched by {covered[f]} tree entries")
    return problems

def render(files, old_entries):
    """a canonical tree over the tracked files, keeping the old block's glob entries and comments where they still apply"""
    comments = {p: c for p, d, c in old_entries}
    globs = [p for p, d, c in old_entries if "*" in p]
    shown = {}
    for f in files:
        g = next((g for g in globs if fnmatch.fnmatch(f, g) and os.path.dirname(f) == os.path.dirname(g)), None)
        shown[g or f] = False
    tree = {}
    for p in shown:
        node = tree
        for part in p.split("/")[:-1]: node = node.setdefault(part + "/", {})
        node[p.split("/")[-1]] = None
    lines = [".  " + comments.get(".", "") if comments.get(".") else "."]
    def walk(node, prefix, path):
        items = sorted(node.items(), key=lambda kv: (kv[1] is None, kv[0].lower()))
        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            full = (path + "/" if path else "") + name.rstrip("/")
            c = comments.get(full, "")
            lines.append(f"{prefix}{'└── ' if last else '├── '}{name}" + (f"  {c}" if c else ""))
            if child is not None: walk(child, prefix + ("    " if last else "│   "), full)
    walk(tree, "", "")
    width = max(len(l.split("  ")[0]) for l in lines) + 2
    out = []
    for l in lines:
        if "  " in l:
            head, c = l.split("  ", 1); out.append(head.ljust(width) + c.strip())
        else: out.append(l)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--repo", help="the git repository whose tracked files are compared (default: --root, or the repository containing the cwd)")
    ap.add_argument("--staged", action="store_true"); ap.add_argument("--write", action="store_true")
    a = ap.parse_args(); root = os.path.abspath(a.root)
    repo = a.repo or (root if os.path.isdir(os.path.join(root, ".git")) else subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())
    files = tracked_files(repo, a.staged)
    text, lines = read_block(os.path.join(root, "README.md"))
    entries = parse_tree(lines)
    if a.write:
        block = BEGIN + "\n```\n" + "\n".join(render(files, entries)) + "\n```\n" + END
        new = text.split(BEGIN, 1)[0] + block + text.split(END, 1)[1]
        open(os.path.join(root, "README.md"), "w", encoding="utf-8").write(new); print("README.md: file tree rewritten"); return
    problems = check(files, entries)
    for p in problems: print("error:", p)
    print(f"file tree: {len(files)} tracked files, {len(entries) - 1} tree entries — {len(problems)} problem(s)")
    sys.exit(1 if problems else 0)

if __name__ == "__main__":
    main()
