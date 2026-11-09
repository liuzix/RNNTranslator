"""Draw docs/cell_neurons.svg: the compiled cell as a textbook recurrent network — circles are units, columns are the
five layers, arrows are the ±1 weights — with one slot write (the obj noun block) traced in colour through
L3 -> L4 -> L5 -> W6, its mirror, one slot read into the output y(t) (orange), and everything else as grey context.
The picture is a drawing of the circuits that test_slot_write_circuit and test_decoder_mirrors_encoder pin, not a
rendering of a compiled cell: its unit counts are illustrative.  The dashed grey tag slots are proposed, not built.

The script needs nothing but Python and is deterministic, so it is pinned the way the PlantUML diagrams are:
docs/architecture.lock.json records the sha256 of this file and of the SVG it wrote, and harness/check_diagram.py
re-runs render() and fails when the committed SVG differs from what the script emits.

    uv run python harness/render_neurons.py        write docs/cell_neurons.svg and update the lock
"""
import os, sys, json, math, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "docs", "architecture.lock.json")
SOURCE, OUTPUT = "harness/render_neurons.py", "docs/cell_neurons.svg"

W, H = 1900, 1130
R = 7                      # neuron radius
P = 18                     # vertical pitch inside a block
C_WRITE = "#1f5fbf"; C_KEEP = "#2a8a5c"; C_NEG = "#c0392b"; C_CTRL = "#333333"; C_OTHER = "#c9c9c9"; C_TEXT = "#222"; C_READ = "#d4791c"
C_STATE = "#dcefd8"; C_IN = "#f8dcdc"; C_NEXT = "#fde7c8"; C_OUT = "#fff3e2"

class Canvas:
    def __init__(self): self.out = []
    def E(self, s): self.out.append(s)

    def neuron(self, x, y, fill=False, color="#333", dashed=False, lw=1.3, fillcolor=None):
        f = (fillcolor or color) if fill else "#fff"
        dash = ' stroke-dasharray="3,3"' if dashed else ""
        self.E(f'<circle cx="{x}" cy="{y}" r="{R}" fill="{f}" stroke="{color}" stroke-width="{lw}"{dash}/>')

    def edge(self, x1, y1, x2, y2, color=C_OTHER, lw=1.0, dashed=False, head=True, shrink=R):
        dx, dy = x2 - x1, y2 - y1; d = math.hypot(dx, dy)
        if d == 0: return
        ux, uy = dx / d, dy / d
        x1, y1 = x1 + ux * shrink, y1 + uy * shrink; x2, y2 = x2 - ux * (shrink + (4 if head else 0)), y2 - uy * (shrink + (4 if head else 0))
        dash = ' stroke-dasharray="4,3"' if dashed else ""
        marker = f' marker-end="url(#arr-{color.strip("#")})"' if head else ""
        self.E(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{lw}"{dash}{marker}/>')

    def curve(self, x1, y1, x2, y2, bend, color=C_OTHER, lw=1.0, dashed=False, head=True):
        """a cubic from the right side of one neuron to the left side of another, bowing by `bend` (positive = down)"""
        x1, x2 = x1 + R, x2 - R - 4
        mx = (x1 + x2) / 2
        dash = ' stroke-dasharray="4,3"' if dashed else ""
        marker = f' marker-end="url(#arr-{color.strip("#")})"' if head else ""
        self.E(f'<path d="M{x1:.1f},{y1:.1f} C{mx:.1f},{y1 + bend:.1f} {mx:.1f},{y2 + bend:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="{color}" stroke-width="{lw}"{dash}{marker}/>')

    def text(self, x, y, s, size=13, anchor="start", color=C_TEXT, weight="normal", italic=False):
        st = f'font-size="{size}" font-family="Helvetica, Arial, sans-serif" fill="{color}" text-anchor="{anchor}" font-weight="{weight}"' + (' font-style="italic"' if italic else "")
        self.E(f'<text x="{x}" y="{y}" {st}>{s}</text>')

    def panel(self, x, y, w, h, fill, dashed=False, stroke="#999"):
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        self.E(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>')

    def block(self, x, y0, n, filled=None, color="#333", dashed=False, label=None, lw=1.3):
        ys = [y0 + i * P for i in range(n)]
        for i, y in enumerate(ys): self.neuron(x, y, fill=(i == filled), color=color, dashed=dashed, lw=lw)
        if label: self.text(x - 22, y0 + (n - 1) * P / 2 + 4, label, size=12, anchor="end", color="#444")
        return ys

    def vdots(self, x, y, color="#666"):
        for i in range(3): self.E(f'<circle cx="{x}" cy="{y + i * 6}" r="1.3" fill="{color}"/>')

    def thresh(self, x, y, s, color="#555"):
        self.text(x + R + 4, y + 4, s, size=10, color=color)

def render():
    c = Canvas(); E, neuron, edge, curve, text, panel, block, vdots, thresh = c.E, c.neuron, c.edge, c.curve, c.text, c.panel, c.block, c.vdots, c.thresh
    X0, X1, X2, X3, X4, X5, X6, X7 = 215, 470, 625, 790, 975, 1180, 1430, 1700       # the columns: inputs, L1..L5, s(t+1), y(t)

    E(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')
    E('<defs>')
    for col in (C_WRITE, C_KEEP, C_NEG, C_CTRL, C_OTHER, C_READ, "#777777", "#555555"):
        E(f'<marker id="arr-{col.strip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
    E('</defs>')
    E(f'<rect width="{W}" height="{H}" fill="#fff"/>')

    # ---- panels: the state, the input, the next state
    panel(70, 60, 220, 690, C_STATE); panel(70, 775, 220, 250, C_IN); panel(X6 - 110, 60, 220, 690, C_NEXT)
    panel(X7 - 90, 450, 180, 320, C_OUT, stroke="#999")
    text(180, 84, "s(t)", 15, "middle", weight="bold"); text(180, 102, "one-hot blocks", 11, "middle", color="#555")
    text(180, 799, "x(t)", 15, "middle", weight="bold"); text(180, 817, "readings of the window", 11, "middle", color="#555")
    text(X6, 84, "s(t+1)", 15, "middle", weight="bold")
    text(X7, 474, "y(t)", 15, "middle", weight="bold"); text(X7, 492, "one morpheme, one-hot", 11, "middle", color="#555")

    # ---- s(t): one-hot blocks
    y = 120
    ctrl = block(X0, y, 6, filled=2, label="ctrl: reading"); y += 6 * P + 6
    pctrl = block(X0, y, 3, label="ctrl: printing", color=C_READ); y += 3 * P + 10
    subj = block(X0, y, 4, filled=0, label="subj"); y += 4 * P + 10
    obj = block(X0, y, 4, filled=1, label="obj noun", color=C_KEEP, lw=1.6); y += 4 * P + 10
    verb = block(X0, y, 4, filled=3, label="verb"); y += 4 * P + 2
    vdots(X0, y); y += 24
    block(X0, y, 3, label="tag slots", dashed=True, color="#777"); text(X0 - 22, y + 2 * P + 14, "(proposed)", 10, "end", "#777"); y += 3 * P + 18
    block(X0, y, 3, label="pend"); y += 3 * P + 10
    block(X0, y, 1, label="shift"); y += P
    neuron(X0, y, color=C_READ); text(X0 - 22, y + 4, "emit", 12, "end", "#444")

    # ---- x(t): the window's readings and the copy sources
    y = 838
    win = block(X0, y, 6, filled=0, label="w0..w3"); neuron(X0, win[3], fill=True); y += 6 * P + 10
    src = block(X0, y, 4, filled=2, label="noun source", color=C_WRITE, lw=1.6)

    # ---- s(t+1): the same blocks
    y = 120
    ctrl2 = block(X6, y, 6, filled=4); y += 6 * P + 6
    block(X6, y, 3, color=C_READ); y += 3 * P + 10
    subj2 = block(X6, y, 4, filled=0); y += 4 * P + 10
    obj2 = block(X6, y, 4, filled=2, color="#333", lw=1.6); y += 4 * P + 10
    verb2 = block(X6, y, 4, filled=3); y += 4 * P + 2
    vdots(X6, y); y += 24
    block(X6, y, 3, dashed=True, color="#777"); y += 3 * P + 18
    block(X6, y, 3); y += 3 * P + 10
    shift2 = block(X6, y, 1, filled=0); y += P
    emit2 = y; neuron(X6, y, color=C_READ)
    text(X6 + 22, obj2[0] - 12, "obj noun", 12, "start", "#444")

    # ---- y(t): one-hot token blocks
    ytok = block(X7, 520, 4, label="obj noun tokens", color=C_READ)
    ystem = block(X7, 610, 3, label="verb stem forms", color=C_READ)
    ypart = block(X7, 682, 3, label="particles", color=C_READ)
    block(X7, 742, 1, label="。", color=C_READ)

    # ---- L1: predicates — an OR over a block, an AND of two window literals, a USE pair
    l1 = [300 + i * P for i in range(11)]
    for yy in l1: neuron(X1, yy, color="#777")
    L1_OR, L1_AND, L1_USE = l1[3], l1[6], l1[9]
    for u in (L1_OR, L1_AND, L1_USE): neuron(X1, u, color="#555")
    for yy in obj: edge(X0, yy, X1, L1_OR, C_OTHER)
    thresh(X1, L1_OR, "0")
    edge(X0, win[0], X1, L1_AND, "#555555"); edge(X0, win[3], X1, L1_AND, "#555555"); thresh(X1, L1_AND, "−1")
    edge(X0, src[2], X1, L1_USE, C_OTHER); edge(X0, verb[3], X1, L1_USE, C_OTHER); thresh(X1, L1_USE, "−1")
    for i in (0, 1, 2, 4, 5, 7, 8, 10):
        edge(X0, subj[i % 4], X1, l1[i], C_OTHER, 0.6); edge(X0, win[(i * 5) % 6], X1, l1[i], C_OTHER, 0.6)
    vdots(X1, l1[-1] + 18)

    # ---- L2: NOT and counts
    l2 = [360 + i * P for i in range(5)]
    for yy in l2: neuron(X2, yy, color="#777")
    edge(X1, L1_OR, X2, l2[1], C_NEG, 1.2); thresh(X2, l2[1], "+1")
    for i in (0, 2, 3, 4): edge(X1, l1[(i * 3) % 11], X2, l2[i], C_OTHER, 0.6); edge(X1, l1[(i * 3 + 5) % 11], X2, l2[i], C_OTHER, 0.6)

    # ---- L3: rules — AND of the control one-hot and the literals; exactly one fires
    l3 = [250 + i * P for i in range(13)]
    RULE = l3[6]; PRULE = l3[12]
    for yy in l3: neuron(X3, yy, color="#777")
    neuron(X3, RULE, fill=True, color=C_CTRL, lw=1.6)
    neuron(X3, PRULE, color=C_READ, lw=1.4); curve(X0, pctrl[1], X3, PRULE, 40, C_READ, 1.0)   # a printing position and its rule
    curve(X0, ctrl[2], X3, RULE, -60, C_CTRL, 1.4); edge(X1, L1_AND, X3, RULE, C_CTRL, 1.4); edge(X2, l2[1], X3, RULE, C_CTRL, 1.4)
    thresh(X3, RULE, "−2", C_CTRL)
    for i, yy in enumerate(l3):
        if yy in (RULE, PRULE): continue
        edge(X1, l1[(i * 7) % 11], X3, yy, C_OTHER, 0.6); edge(X2, l2[(i * 2) % 5], X3, yy, C_OTHER, 0.6)
        edge(X0 + 2, ctrl[i % 6], X3, yy, C_OTHER, 0.5, shrink=R)
    vdots(X3, l3[-1] + 18)

    # ---- L4: next control, shift, the gates (each an OR of the rules that write a block)
    nxt = block(X4, 150, 6, filled=4, color="#777"); neuron(X4, nxt[4], fill=True, color=C_CTRL, lw=1.6)
    text(X4, 136, "next ctrl", 11, "middle", "#555")
    SH = 300; neuron(X4, SH, fill=True, color=C_CTRL, lw=1.6); text(X4, SH - 14, "shift", 11, "middle", "#555")
    EM = 330; neuron(X4, EM, color=C_READ, lw=1.4); text(X4 + 14, EM + 4, "emit", 11, "start", C_READ)
    GB = 390; neuron(X4, GB, fill=True, color=C_NEG, lw=1.6); text(X4 + 14, GB - 10, "block gate", 11, "start", C_NEG)
    GS = 450; neuron(X4, GS, fill=True, color=C_WRITE, lw=1.6); text(X4 + 14, GS + 20, "source gate", 11, "start", C_WRITE)
    RG = 510; neuron(X4, RG, color=C_READ, lw=1.4); text(X4 + 14, RG - 8, "read gate", 11, "start", C_READ)
    FG = 570; neuron(X4, FG, color=C_READ, lw=1.4); text(X4 + 14, FG + 16, "form gate", 11, "start", C_READ)
    other_g = block(X4, 640, 4, color="#777"); text(X4, 626, "other gates", 11, "middle", "#555")
    GU = 760; neuron(X4, GU, color="#777"); text(X4, GU - 14, "USE gate", 11, "middle", "#555")
    edge(X3, PRULE, X4, RG, C_READ, 1.2); edge(X3, PRULE, X4, EM, C_READ, 1.2); edge(X3, PRULE, X4, FG, C_READ, 1.2)
    edge(X3, RULE, X4, nxt[4], C_CTRL, 1.4); edge(X3, RULE, X4, SH, C_CTRL, 1.4)
    edge(X3, RULE, X4, GB, C_NEG, 1.4); edge(X3, RULE, X4, GS, C_WRITE, 1.4)
    for yy in (l3[2], l3[10]): edge(X3, yy, X4, GB, C_OTHER, 0.8); edge(X3, yy, X4, GS, C_OTHER, 0.8)
    for i, yy in enumerate(l3):
        if yy == RULE: continue
        edge(X3, yy, X4, nxt[i % 6], C_OTHER, 0.5); edge(X3, yy, X4, other_g[i % 4], C_OTHER, 0.5)
    edge(X1, L1_USE, X4, GU, "#777777", 1.0); edge(X3, l3[11], X4, GU, "#777777", 1.0)

    # ---- L5: write units (gate AND source value) and keep units (old value minus gate) for the obj block
    wr = block(X5, 330, 4, color=C_WRITE, lw=1.6); text(X5, 316, "write units", 11, "middle", C_WRITE)
    kp = block(X5, 450, 4, color=C_KEEP, lw=1.6); text(X5, 436, "keep units", 11, "middle", C_KEEP)
    neuron(X5, wr[2], fill=True, color=C_WRITE, lw=1.6)
    rd = block(X5, 570, 4, color=C_READ, lw=1.4); text(X5, 556, "read units", 11, "middle", C_READ)
    lk = block(X5, 670, 3, color=C_READ, lw=1.4); text(X5, 656, "lookup units", 11, "middle", C_READ)
    rest = block(X5, 760, 6, color="#777"); text(X5, 746, "other blocks", 11, "middle", "#555")
    CR = 910; neuron(X5, CR, color=C_NEG); text(X5 + 14, CR + 4, "no source → FAIL", 11, "start", C_NEG)
    for yy in rd: edge(X4, RG, X5, yy, C_READ, 1.1); thresh(X5, yy, "−1", C_READ)             # the gate selects: a mux
    for yy in lk: edge(X4, FG, X5, yy, C_READ, 1.1); thresh(X5, yy, "−1", C_READ)             # conjugation as a table
    for i, yy in enumerate(rd): curve(X0, obj[i], X5, yy, 140, C_READ, 0.9)                  # + the block value, read from s(t)
    for i, yy in enumerate(lk): curve(X0, verb[i + 1], X5, yy, 170, C_READ, 0.9)             # + the verb, read from s(t)
    for yy in wr: edge(X4, GS, X5, yy, C_WRITE, 1.3); thresh(X5, yy, "−1", C_WRITE)
    for yy in kp: edge(X4, GB, X5, yy, C_NEG, 1.3); thresh(X5, yy, "0", C_KEEP)
    for i, yy in enumerate(wr): curve(X0, src[i], X5, yy, 120, C_WRITE, 1.0)
    for i, yy in enumerate(kp): curve(X0, obj[i], X5, yy, -150, C_KEEP, 1.0)
    edge(X4, GS, X5, CR, C_OTHER, 0.8)
    for i, yy in enumerate(rest[1:]): edge(X4, other_g[i % 4], X5, yy, C_OTHER, 0.6); curve(X0, subj[i % 4], X5, yy, 60, C_OTHER, 0.5)
    edge(X4, GU, X5, rest[0], "#777777", 1.0); edge(X1, L1_USE, X5, rest[0], "#777777", 1.0); thresh(X5, rest[0], "−1", "#777")
    vdots(X5, rest[-1] + 18)

    # ---- W6: keep + write into the next state's block; next control; shift; the crash into FAIL
    edge(X4, nxt[4], X6, ctrl2[4], C_CTRL, 1.4); edge(X4, SH, X6, shift2[0], C_CTRL, 1.4)
    for i in range(4): edge(X5, wr[i], X6, obj2[i], C_WRITE, 1.2); edge(X5, kp[i], X6, obj2[i], C_KEEP, 1.2)
    for i, yy in enumerate(rest[1:]): edge(X5, yy, X6, (subj2 + verb2)[i % 8], C_OTHER, 0.6)
    edge(X5, rest[0], X6, verb2[1], "#777777", 1.0)
    edge(X5, CR, X6, ctrl2[5], C_NEG, 0.9, dashed=True); text(X6 + 22, ctrl2[5] + 4, "FAIL", 10, "start", C_NEG)
    edge(X4, EM, X6, emit2, C_READ, 1.1)
    for i in range(4): curve(X5, rd[i], X7, ytok[i], 0, C_READ, 1.0)                         # W6 into y(t): the block's token
    for i in range(3): curve(X5, lk[i], X7, ystem[i], 0, C_READ, 1.0)                        # W6 into y(t): the stem form
    curve(X3, PRULE, X7, ypart[1], 120, C_READ, 0.9)                                           # a constant particle, straight from the rule

    # ---- the recurrence, over the top
    E(f'<path d="M{X6 - 110},90 H{X6 - 125} V30 H{330} V90 H{292}" fill="none" stroke="{C_CTRL}" stroke-width="1.6" marker-end="url(#arr-333333)"/>')
    text((X6 - 125 + 330) / 2, 22, "next word: s(t) ← s(t+1)", 12, "middle", C_CTRL)

    # ---- outputs: the output lexical layer and the print head
    E(f'<rect x="{X7 - 90}" y="800" width="180" height="54" rx="8" fill="#eeeeee" stroke="#999"/>')
    text(X7, 823, "output lexical layer", 12, "middle", weight="bold"); text(X7, 841, "unit → string", 11, "middle", "#555")
    E(f'<path d="M{X7},774 V796" fill="none" stroke="#777777" stroke-width="1" marker-end="url(#arr-777777)"/>')
    E(f'<rect x="{X7 - 90}" y="890" width="180" height="54" rx="8" fill="#eeeeee" stroke="#999"/>')
    text(X7, 913, "print head", 12, "middle", weight="bold"); text(X7, 931, "appends when emit = 1", 11, "middle", "#555")
    E(f'<path d="M{X7},858 V886" fill="none" stroke="#777777" stroke-width="1" marker-end="url(#arr-777777)"/>')
    curve(X6, emit2, X7 - 90 - R, 917, 60, C_READ, 0.9)
    E(f'<rect x="{70}" y="{1035}" width="220" height="40" rx="8" fill="#eeeeee" stroke="#999"/>')
    text(180, 1052, "read head", 12, "middle", weight="bold"); text(180, 1067, "advances when shift = 1", 10, "middle", "#555")

    # ---- layer and weight labels
    yl = H - 30
    for x, s in ((X1, "L1 predicates"), (X2, "L2 NOT"), (X3, "L3 rules"), (X4, "L4 gates"), (X5, "L5 keep · write")):
        text(x, yl, s, 13, "middle", weight="bold")
    text(X3, yl + 16, "exactly one fires", 11, "middle", "#555"); text(X5, yl + 16, "ReLU, weights ±1", 11, "middle", "#555")
    for x, s in (((X0 + X1) / 2, "W1"), ((X1 + X2) / 2, "W2"), ((X2 + X3) / 2, "W3"), ((X3 + X4) / 2, "W4"), ((X4 + X5) / 2, "W5"), ((X5 + X6) / 2 - 50, "W6")):
        text(x, 108, s, 12, "middle", "#555", italic=True)

    # ---- legend
    lx, ly = 1300, 990
    panel(lx - 20, ly - 24, 580, 140, "#fafafa", stroke="#ccc")
    text(lx, ly, "one slot write, and its mirror, one slot read", 13, weight="bold")
    neuron(lx + 8, ly + 24, fill=True); text(lx + 24, ly + 28, "unit = 1", 12)
    neuron(lx + 98, ly + 24); text(lx + 114, ly + 28, "unit = 0", 12)
    neuron(lx + 188, ly + 24, dashed=True, color="#777"); text(lx + 204, ly + 28, "proposed (tag slots)", 12)
    text(lx + 290, ly + 28, "−1  threshold beside a unit", 12)
    entries = ((C_CTRL, 1.4, False, "control: one-hot → rule → next"), (C_WRITE, 1.4, False, "write: gate ∧ source → slot"),
               (C_KEEP, 1.4, False, "keep: old value unless gated"), (C_NEG, 1.4, False, "−1: the gate cuts"),
               ("#777777", 1.0, False, "USE: the noun rewrites the verb"), (C_OTHER, 1.0, False, "other units, sparse"),
               (C_READ, 1.1, False, "read: read gate ∧ slot → y(t)"), (C_READ, 1.1, False, "lookup: form gate ∧ verb → stem"))
    for k, (col, lw, dsh, s) in enumerate(entries):
        cx, cy = lx + 290 * (k // 4), ly + 50 + 22 * (k % 4)
        edge(cx, cy, cx + 36, cy, col, lw, shrink=0, dashed=dsh); text(cx + 46, cy + 4, s, 12)

    E('</svg>')
    return "\n".join(c.out) + "\n"

def sha256_bytes(b): return hashlib.sha256(b).hexdigest()

def main():
    svg = render().encode("utf-8")
    out = os.path.join(ROOT, OUTPUT); open(out, "wb").write(svg)
    lock = json.load(open(LOCK, encoding="utf-8"))
    entry = {"source": SOURCE, "output": OUTPUT, "source_sha256": sha256_bytes(open(os.path.abspath(__file__), "rb").read()), "output_sha256": sha256_bytes(svg)}
    drawings = [d for d in lock.get("drawings", []) if d["output"] != OUTPUT] + [entry]
    lock["drawings"] = drawings
    json.dump(lock, open(LOCK, "w", encoding="utf-8"), indent=1)
    print(f"drew {OUTPUT} ({len(svg)} bytes); updated {os.path.relpath(LOCK, ROOT)}")

if __name__ == "__main__":
    main()
