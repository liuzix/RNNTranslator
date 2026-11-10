"""Draw docs/cell_neurons.svg: the compiled machine as two textbook recurrent networks — circles are units, columns are the
five layers, arrows are the ±1 weights.  Above, the encoder cell, with one slot write (the obj noun block) traced in colour
through L3 -> L4 -> L5 -> W6 and the flush unit that hands the slot block to the context; below, the decoder cell, whose
input is the frame under the attention head, with the mirror of the write, one slot read and one form lookup into the output
y(t), in orange, and the down and up units that move the attention.  Between them the context: one frame per clause.
The picture is a drawing of the circuits that test_slot_write_circuit and test_decoder_mirrors_encoder pin, not a rendering
of a compiled cell: its unit counts are illustrative.

The script needs nothing but Python and is deterministic, so it is pinned the way the PlantUML diagrams are:
docs/architecture.lock.json records the sha256 of this file and of the SVG it wrote, and harness/check_diagram.py
re-runs render() and fails when the committed SVG differs from what the script emits.

    uv run python harness/render_neurons.py        write docs/cell_neurons.svg and update the lock
"""
import os, sys, json, math, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "docs", "architecture.lock.json")
SOURCE, OUTPUT = "harness/render_neurons.py", "docs/cell_neurons.svg"

W, H = 1900, 1740
R = 7                      # neuron radius
P = 18                     # vertical pitch inside a block
C_WRITE = "#1f5fbf"; C_KEEP = "#2a8a5c"; C_NEG = "#c0392b"; C_CTRL = "#333333"; C_OTHER = "#c9c9c9"; C_TEXT = "#222"; C_READ = "#d4791c"
C_STATE = "#dcefd8"; C_IN = "#f8dcdc"; C_NEXT = "#fde7c8"; C_OUT = "#fff3e2"; C_CTX = "#9a7b12"; C_FRAME = "#fff6d6"

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
    X0, X1, X2, X3, X4, X5, X6, X7 = 215, 470, 625, 790, 975, 1180, 1430, 1700       # the columns: state and input, L1..L5, s(t+1), context / y(t)
    DY = 900                                                                          # the decoder's row begins here

    E(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')
    E('<defs>')
    for col in (C_WRITE, C_KEEP, C_NEG, C_CTRL, C_OTHER, C_READ, C_CTX, "#777777", "#555555"):
        E(f'<marker id="arr-{col.strip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
    E('</defs>')
    E(f'<rect width="{W}" height="{H}" fill="#fff"/>')
    text(40, 34, "encoder cell", 17, weight="bold"); text(40, DY + 34, "decoder cell", 17, weight="bold")
    E(f'<line x1="30" y1="{DY - 4}" x2="{W - 30}" y2="{DY - 4}" stroke="#ccc" stroke-width="1" stroke-dasharray="2,5"/>')

    # ================================================================ the encoder
    panel(70, 60, 220, 545, C_STATE); panel(70, 625, 220, 215, C_IN); panel(X6 - 110, 60, 220, 545, C_NEXT)
    text(180, 84, "s(t)", 15, "middle", weight="bold"); text(180, 102, "one-hot blocks", 11, "middle", color="#555")
    text(180, 649, "x(t)", 15, "middle", weight="bold"); text(180, 667, "readings of the window", 11, "middle", color="#555")
    text(X6, 84, "s(t+1)", 15, "middle", weight="bold")

    # ---- s(t): one-hot blocks
    y = 120
    ctrl = block(X0, y, 6, filled=2, label="ctrl: reading"); y += 6 * P + 10
    subj = block(X0, y, 4, filled=0, label="subj"); y += 4 * P + 10
    obj = block(X0, y, 4, filled=1, label="obj noun", color=C_KEEP, lw=1.6); y += 4 * P + 10
    verb = block(X0, y, 4, filled=3, label="verb"); y += 4 * P + 2
    vdots(X0, y); y += 24
    block(X0, y, 3, label="pend"); y += 3 * P + 10
    block(X0, y, 1, label="shift"); y += P + 6
    neuron(X0, y, color=C_CTX); text(X0 - 22, y + 4, "flush", 12, "end", "#444")

    # ---- x(t): the window's readings and the copy sources
    y = 690
    win = block(X0, y, 5, filled=0, label="w0..w3"); neuron(X0, win[3], fill=True); y += 5 * P + 10
    src = block(X0, y, 3, filled=2, label="noun source", color=C_WRITE, lw=1.6)

    # ---- s(t+1): the same blocks
    y = 120
    ctrl2 = block(X6, y, 6, filled=4); y += 6 * P + 10
    subj2 = block(X6, y, 4, filled=0); y += 4 * P + 10
    obj2 = block(X6, y, 4, filled=2, color="#333", lw=1.6); y += 4 * P + 10
    verb2 = block(X6, y, 4, filled=3); y += 4 * P + 2
    vdots(X6, y); y += 24
    block(X6, y, 3); y += 3 * P + 10
    shift2 = block(X6, y, 1, filled=0); y += P + 6
    flush2 = y; neuron(X6, y, fill=True, color=C_CTX)
    text(X6 + 22, obj2[0] - 12, "obj noun", 12, "start", "#444"); text(X6 + 22, shift2[0] + 4, "shift", 11, "start", "#555"); text(X6 + 22, flush2 + 4, "flush", 11, "start", C_CTX)

    # ---- L1: predicates — an OR over a block, an AND of two window literals, a USE pair
    l1 = [270 + i * P for i in range(11)]
    for yy in l1: neuron(X1, yy, color="#777")
    L1_OR, L1_AND, L1_USE = l1[3], l1[6], l1[9]
    for u in (L1_OR, L1_AND, L1_USE): neuron(X1, u, color="#555")
    for yy in obj: edge(X0, yy, X1, L1_OR, C_OTHER)
    thresh(X1, L1_OR, "0")
    edge(X0, win[0], X1, L1_AND, "#555555"); edge(X0, win[3], X1, L1_AND, "#555555"); thresh(X1, L1_AND, "−1")
    edge(X0, src[2], X1, L1_USE, C_OTHER); edge(X0, verb[3], X1, L1_USE, C_OTHER); thresh(X1, L1_USE, "−1")
    for i in (0, 1, 2, 4, 5, 7, 8, 10):
        edge(X0, subj[i % 4], X1, l1[i], C_OTHER, 0.6); edge(X0, win[(i * 3) % 5], X1, l1[i], C_OTHER, 0.6)
    vdots(X1, l1[-1] + 18)

    # ---- L2: NOT and counts
    l2 = [330 + i * P for i in range(5)]
    for yy in l2: neuron(X2, yy, color="#777")
    edge(X1, L1_OR, X2, l2[1], C_NEG, 1.2); thresh(X2, l2[1], "+1")
    for i in (0, 2, 3, 4): edge(X1, l1[(i * 3) % 11], X2, l2[i], C_OTHER, 0.6); edge(X1, l1[(i * 3 + 5) % 11], X2, l2[i], C_OTHER, 0.6)

    # ---- L3: rules — AND of the control one-hot and the literals; exactly one fires
    l3 = [230 + i * P for i in range(13)]
    RULE = l3[6]; FRULE = l3[11]
    for yy in l3: neuron(X3, yy, color="#777")
    neuron(X3, RULE, fill=True, color=C_CTRL, lw=1.6)
    curve(X0, ctrl[2], X3, RULE, -60, C_CTRL, 1.4); edge(X1, L1_AND, X3, RULE, C_CTRL, 1.4); edge(X2, l2[1], X3, RULE, C_CTRL, 1.4)
    thresh(X3, RULE, "−2", C_CTRL)
    for i, yy in enumerate(l3):
        if yy == RULE: continue
        edge(X1, l1[(i * 7) % 11], X3, yy, C_OTHER, 0.6); edge(X2, l2[(i * 2) % 5], X3, yy, C_OTHER, 0.6)
        edge(X0 + 2, ctrl[i % 6], X3, yy, C_OTHER, 0.5, shrink=R)
    vdots(X3, l3[-1] + 18)

    # ---- L4: next control, shift, flush, the gates (each an OR of the rules that write a block)
    nxt = block(X4, 130, 6, filled=4, color="#777"); neuron(X4, nxt[4], fill=True, color=C_CTRL, lw=1.6)
    text(X4, 116, "next ctrl", 11, "middle", "#555")
    SH = 270; neuron(X4, SH, fill=True, color=C_CTRL, lw=1.6); text(X4 + 14, SH + 4, "shift", 11, "start", "#555")
    FL = 310; neuron(X4, FL, color=C_CTX, lw=1.4); text(X4 + 14, FL + 4, "flush", 11, "start", C_CTX)
    GB = 370; neuron(X4, GB, fill=True, color=C_NEG, lw=1.6); text(X4 + 14, GB - 10, "block gate", 11, "start", C_NEG)
    GS = 430; neuron(X4, GS, fill=True, color=C_WRITE, lw=1.6); text(X4 + 14, GS + 20, "source gate", 11, "start", C_WRITE)
    other_g = block(X4, 510, 4, color="#777"); text(X4, 496, "other gates", 11, "middle", "#555")
    GU = 630; neuron(X4, GU, color="#777"); text(X4, GU - 14, "USE gate", 11, "middle", "#555")
    edge(X3, RULE, X4, nxt[4], C_CTRL, 1.4); edge(X3, RULE, X4, SH, C_CTRL, 1.4)
    edge(X3, RULE, X4, GB, C_NEG, 1.4); edge(X3, RULE, X4, GS, C_WRITE, 1.4)
    edge(X3, FRULE, X4, FL, C_CTX, 1.1); edge(X3, l3[12], X4, FL, C_CTX, 1.1)                 # the rules at a clause boundary and at the end
    for yy in (l3[2], l3[10]): edge(X3, yy, X4, GB, C_OTHER, 0.8); edge(X3, yy, X4, GS, C_OTHER, 0.8)
    for i, yy in enumerate(l3):
        if yy == RULE: continue
        edge(X3, yy, X4, nxt[i % 6], C_OTHER, 0.5); edge(X3, yy, X4, other_g[i % 4], C_OTHER, 0.5)
    edge(X1, L1_USE, X4, GU, "#777777", 1.0); edge(X3, l3[9], X4, GU, "#777777", 1.0)

    # ---- L5: write units (gate AND source value) and keep units (old value minus gate) for the obj block
    wr = block(X5, 310, 4, color=C_WRITE, lw=1.6); text(X5, 296, "write units", 11, "middle", C_WRITE)
    kp = block(X5, 430, 4, color=C_KEEP, lw=1.6); text(X5, 416, "keep units", 11, "middle", C_KEEP)
    neuron(X5, wr[2], fill=True, color=C_WRITE, lw=1.6)
    rest = block(X5, 550, 6, color="#777"); text(X5, 536, "other blocks", 11, "middle", "#555")
    CR = 700; neuron(X5, CR, color=C_NEG); text(X5 + 14, CR + 4, "no source → FAIL", 11, "start", C_NEG)
    for yy in wr: edge(X4, GS, X5, yy, C_WRITE, 1.3); thresh(X5, yy, "−1", C_WRITE)
    for yy in kp: edge(X4, GB, X5, yy, C_NEG, 1.3); thresh(X5, yy, "0", C_KEEP)
    for i, yy in enumerate(wr[:3]): curve(X0, src[i], X5, yy, 110, C_WRITE, 1.0)
    for i, yy in enumerate(kp): curve(X0, obj[i], X5, yy, -150, C_KEEP, 1.0)
    edge(X4, GS, X5, CR, C_OTHER, 0.8)
    for i, yy in enumerate(rest[1:]): edge(X4, other_g[i % 4], X5, yy, C_OTHER, 0.6); curve(X0, subj[i % 4], X5, yy, 60, C_OTHER, 0.5)
    edge(X4, GU, X5, rest[0], "#777777", 1.0); edge(X1, L1_USE, X5, rest[0], "#777777", 1.0); thresh(X5, rest[0], "−1", "#777")
    vdots(X5, rest[-1] + 18)

    # ---- W6: keep + write into the next state's block; next control; shift; flush; the crash into FAIL
    edge(X4, nxt[4], X6, ctrl2[4], C_CTRL, 1.4); edge(X4, SH, X6, shift2[0], C_CTRL, 1.4); edge(X4, FL, X6, flush2, C_CTX, 1.1)
    for i in range(4): edge(X5, wr[i], X6, obj2[i], C_WRITE, 1.2); edge(X5, kp[i], X6, obj2[i], C_KEEP, 1.2)
    for i, yy in enumerate(rest[1:]): edge(X5, yy, X6, (subj2 + verb2)[i % 8], C_OTHER, 0.6)
    edge(X5, rest[0], X6, verb2[1], "#777777", 1.0)
    edge(X5, CR, X6, ctrl2[5], C_NEG, 0.9, dashed=True); text(X6 + 22, ctrl2[5] + 4, "FAIL", 10, "start", C_NEG)

    # ---- the recurrence, over the top; the read head
    E(f'<path d="M{X6 - 110},90 H{X6 - 125} V30 H{330} V90 H{292}" fill="none" stroke="{C_CTRL}" stroke-width="1.6" marker-end="url(#arr-333333)"/>')
    text((X6 - 125 + 330) / 2, 22, "next word: s(t) ← s(t+1)", 12, "middle", C_CTRL)
    E(f'<rect x="70" y="850" width="220" height="38" rx="8" fill="#eeeeee" stroke="#999"/>')
    text(180, 866, "read head", 12, "middle", weight="bold"); text(180, 881, "advances when shift = 1", 10, "middle", "#555")

    # ---- the context: one frame per clause; the slot block is appended when flush = 1
    panel(X7 - 100, 60, 200, 545, C_FRAME, stroke=C_CTX)
    text(X7, 84, "context", 15, "middle", weight="bold"); text(X7, 102, "one frame per clause", 11, "middle", color="#555")
    FX1, FX2 = X7 - 45, X7 + 45
    f1 = [block(FX1, yy, 4, filled=k, color="#333") for yy, k in ((subj2[0], 0), (obj2[0], 2), (verb2[0], 3))]
    f2 = [block(FX2, yy, 4, color="#999", dashed=True) for yy in (subj2[0], obj2[0], verb2[0])]
    text(FX1, subj2[0] - 16, "frame 1", 11, "middle", "#444"); text(FX2, subj2[0] - 16, "frame 2", 11, "middle", "#888")
    for b2, fb in zip((subj2, obj2, verb2), f1): curve(X6, b2[1], FX1, fb[1], 0, C_CTX, 1.2)
    text(X7, verb2[-1] + 44, "appended when flush = 1:", 11, "middle", C_CTX); text(X7, verb2[-1] + 60, "the slot block, whole", 11, "middle", C_CTX)
    neuron(FX1, 566, color="#333"); text(FX1 + 16, 570, "sentence block", 11, "start", "#444")
    text(X7, 592, "(of the final state)", 10, "middle", "#777")

    # ---- encoder: layer and weight labels, legend
    yl = 800
    for x, s in ((X1, "L1 predicates"), (X2, "L2 NOT"), (X3, "L3 rules"), (X4, "L4 gates"), (X5, "L5 keep · write")):
        text(x, yl, s, 13, "middle", weight="bold")
    text(X3, yl + 16, "exactly one fires", 11, "middle", "#555"); text(X5, yl + 16, "ReLU, weights ±1", 11, "middle", "#555")
    for x, s in (((X0 + X1) / 2, "W1"), ((X1 + X2) / 2, "W2"), ((X2 + X3) / 2, "W3"), ((X3 + X4) / 2, "W4"), ((X4 + X5) / 2, "W5"), ((X5 + X6) / 2 - 50, "W6")):
        text(x, 108, s, 12, "middle", "#555", italic=True)
    lx, ly = 1340, 664
    panel(lx - 20, ly - 24, 520, 150, "#fafafa", stroke="#ccc")
    text(lx, ly, "the encoder: one slot write", 13, weight="bold")
    neuron(lx + 8, ly + 24, fill=True); text(lx + 24, ly + 28, "unit = 1", 12)
    neuron(lx + 98, ly + 24); text(lx + 114, ly + 28, "unit = 0", 12)
    text(lx + 200, ly + 28, "−1  threshold beside a unit", 12)
    entries = ((C_CTRL, 1.4, "control: one-hot → rule → next"), (C_WRITE, 1.4, "write: gate ∧ source → slot"), (C_KEEP, 1.4, "keep: old value unless gated"), (C_NEG, 1.4, "−1: the gate cuts"),
               ("#777777", 1.0, "USE: the noun rewrites the verb"), (C_CTX, 1.1, "flush: the frame → context"), (C_OTHER, 1.0, "other units, sparse"))
    for k, (col, lw, s) in enumerate(entries):
        cx, cy = lx + 262 * (k // 4), ly + 52 + 22 * (k % 4)
        edge(cx, cy, cx + 36, cy, col, lw, shrink=0); text(cx + 46, cy + 4, s, 12)

    # ================================================================ the decoder
    D = DY
    panel(70, D + 60, 220, 300, C_FRAME, stroke=C_CTX); panel(70, D + 380, 220, 270, C_STATE)
    panel(X6 - 110, D + 60, 220, 330, C_NEXT); panel(X7 - 90, D + 60, 180, 350, C_OUT, stroke="#999")
    text(180, D + 84, "x(t)", 15, "middle", weight="bold"); text(180, D + 102, "the frame under the attention", 11, "middle", color="#555")
    text(180, D + 404, "s(t)", 15, "middle", weight="bold"); text(180, D + 422, "one-hot blocks", 11, "middle", color="#555")
    text(X6, D + 84, "s(t+1)", 15, "middle", weight="bold")
    text(X7, D + 84, "y(t)", 15, "middle", weight="bold"); text(X7, D + 102, "one morpheme, one-hot", 11, "middle", color="#555")

    # ---- x(t): the frame, block for block the encoder's slots
    y = D + 124
    dsubj = block(X0, y, 4, filled=0, label="subj"); y += 4 * P + 10
    dobj = block(X0, y, 4, filled=2, label="obj noun", color=C_READ, lw=1.6); y += 4 * P + 10
    dverb = block(X0, y, 4, filled=3, label="verb", color=C_READ, lw=1.3)
    # ---- s(t): the printing position, the sentence block, the head units
    y = D + 444
    pos = block(X0, y, 5, filled=1, label="ctrl: printing", color=C_READ); y += 5 * P + 12
    sent = y; neuron(X0, y); text(X0 - 22, y + 4, "sentence block", 12, "end", "#444"); y += P + 12
    for k, name in enumerate(("emit", "down", "up")): neuron(X0, y + k * P, color=C_READ if k == 0 else C_CTX); text(X0 - 22, y + k * P + 4, name, 12, "end", "#444")

    # ---- L1 and L2: conjunctions over the frame and their NOTs
    d1 = [D + 190 + i * P for i in range(7)]
    for yy in d1: neuron(X1, yy, color="#777")
    D1_AND = d1[3]; neuron(X1, D1_AND, color="#555")
    edge(X0, dverb[3], X1, D1_AND, "#555555"); edge(X0, sent, X1, D1_AND, "#555555"); thresh(X1, D1_AND, "−1")
    for i in (0, 1, 2, 4, 5, 6): edge(X0, dsubj[i % 4], X1, d1[i], C_OTHER, 0.6); edge(X0, dverb[(i * 3) % 4], X1, d1[i], C_OTHER, 0.6)
    d2 = [D + 226 + i * P for i in range(3)]
    for yy in d2: neuron(X2, yy, color="#777")
    edge(X1, D1_AND, X2, d2[1], C_NEG, 1.2); thresh(X2, d2[1], "+1")
    for i in (0, 2): edge(X1, d1[i * 3], X2, d2[i], C_OTHER, 0.6); edge(X1, d1[i * 2 + 1], X2, d2[i], C_OTHER, 0.6)

    # ---- L3: the printing rules — AND of the printing position and the literals over the frame
    d3 = [D + 170 + i * P for i in range(9)]
    PRULE = d3[4]; MRULE = d3[7]
    for yy in d3: neuron(X3, yy, color="#777")
    neuron(X3, PRULE, fill=True, color=C_READ, lw=1.6)
    curve(X0, pos[1], X3, PRULE, 60, C_READ, 1.4); edge(X2, d2[1], X3, PRULE, C_READ, 1.2); thresh(X3, PRULE, "−1", C_READ)
    for i, yy in enumerate(d3):
        if yy == PRULE: continue
        edge(X1, d1[(i * 5) % 7], X3, yy, C_OTHER, 0.6); edge(X2, d2[i % 3], X3, yy, C_OTHER, 0.6); curve(X0, pos[i % 5], X3, yy, 40, C_OTHER, 0.5)
    vdots(X3, d3[-1] + 18)

    # ---- L4: the next position, emit, down, up, the read gate and the form gate
    dn = block(X4, D + 130, 5, filled=2, color="#777"); neuron(X4, dn[2], fill=True, color=C_READ, lw=1.6)
    text(X4, D + 116, "next position", 11, "middle", "#555")
    EM = D + 250; neuron(X4, EM, fill=True, color=C_READ, lw=1.6); text(X4 + 14, EM + 4, "emit", 11, "start", C_READ)
    DN = D + 290; neuron(X4, DN, color=C_CTX, lw=1.4); text(X4 + 14, DN + 4, "down", 11, "start", C_CTX)
    UP = D + 320; neuron(X4, UP, color=C_CTX, lw=1.4); text(X4 + 14, UP + 4, "up", 11, "start", C_CTX)
    RG = D + 380; neuron(X4, RG, fill=True, color=C_READ, lw=1.6); text(X4 + 14, RG - 8, "read gate", 11, "start", C_READ)
    FG = D + 440; neuron(X4, FG, color=C_READ, lw=1.4); text(X4 + 14, FG + 18, "form gate", 11, "start", C_READ)
    edge(X3, PRULE, X4, dn[2], C_READ, 1.4); edge(X3, PRULE, X4, EM, C_READ, 1.4); edge(X3, PRULE, X4, RG, C_READ, 1.4)
    edge(X3, d3[2], X4, FG, C_READ, 0.9); edge(X3, d3[2], X4, EM, C_READ, 0.9)
    edge(X3, MRULE, X4, DN, C_CTX, 1.1); edge(X3, d3[8], X4, UP, C_CTX, 1.1)                    # the rules at a link and at the end of an inner clause
    for i, yy in enumerate(d3):
        if yy == PRULE: continue
        edge(X3, yy, X4, dn[i % 5], C_OTHER, 0.5)

    # ---- L5: read units (read gate AND the frame's value), lookup units (form gate AND the verb), the crash, the sentence block kept
    rd = block(X5, D + 330, 4, color=C_READ, lw=1.6); text(X5, D + 316, "read units", 11, "middle", C_READ)
    neuron(X5, rd[2], fill=True, color=C_READ, lw=1.6)
    lk = block(X5, D + 440, 3, color=C_READ, lw=1.4); text(X5, D + 426, "lookup units", 11, "middle", C_READ)
    CR2 = D + 520; neuron(X5, CR2, color=C_NEG); text(X5 + 14, CR2 + 4, "nothing to read → FAIL", 11, "start", C_NEG)
    KS = D + 560; neuron(X5, KS, color="#777"); text(X5 + 14, KS + 4, "keep: sentence block", 11, "start", "#555")
    for yy in rd: edge(X4, RG, X5, yy, C_READ, 1.2); thresh(X5, yy, "−1", C_READ)               # the gate selects: a mux
    for yy in lk: edge(X4, FG, X5, yy, C_READ, 1.0); thresh(X5, yy, "−1", C_READ)               # conjugation as a table
    for i, yy in enumerate(rd): curve(X0, dobj[i], X5, yy, 150, C_READ, 1.0)                    # + the block's value, read from the frame
    for i, yy in enumerate(lk): curve(X0, dverb[i + 1], X5, yy, 190, C_READ, 0.8)               # + the verb, read from the frame
    edge(X4, RG, X5, CR2, C_OTHER, 0.8); curve(X0, sent, X5, KS, 90, C_OTHER, 0.7)

    # ---- s(t+1) and y(t)
    y = D + 130
    pos2 = block(X6, y, 5, filled=2, color=C_READ); y += 5 * P + 12
    sent2 = y; neuron(X6, y); y += P + 12
    em2, dn2, up2 = y, y + P, y + 2 * P
    neuron(X6, em2, fill=True, color=C_READ); neuron(X6, dn2, color=C_CTX); neuron(X6, up2, color=C_CTX)
    for yy, s_ in ((sent2, "sentence block"), (em2, "emit"), (dn2, "down"), (up2, "up")): text(X6 + 22, yy + 4, s_, 11, "start", "#555")
    text(X6 + 22, pos2[0] - 12, "printing position", 12, "start", "#444")
    ytok = block(X7, D + 140, 4, label="obj noun tokens", color=C_READ); neuron(X7, ytok[2], fill=True, color=C_READ)
    ystem = block(X7, D + 232, 3, label="verb stem forms", color=C_READ)
    ypart = block(X7, D + 306, 3, label="particles", color=C_READ)
    block(X7, D + 380, 1, label="。", color=C_READ)
    edge(X4, dn[2], X6, pos2[2], C_READ, 1.4); edge(X4, EM, X6, em2, C_READ, 1.2); edge(X4, DN, X6, dn2, C_CTX, 1.1); edge(X4, UP, X6, up2, C_CTX, 1.1)
    edge(X5, KS, X6, sent2, C_OTHER, 0.8); edge(X5, CR2, X6, pos2[4], C_NEG, 0.9, dashed=True); text(X6 - 24, pos2[4] + 16, "FAIL", 10, "end", C_NEG)
    for i in range(4): curve(X5, rd[i], X7, ytok[i], -40, C_READ, 1.0)                           # W6 into y(t): the block's token
    for i in range(3): curve(X5, lk[i], X7, ystem[i], -30, C_READ, 0.9)                          # W6 into y(t): the stem form
    curve(X3, d3[1], X7, ypart[1], -90, C_READ, 0.8)                                             # a constant particle, straight from its rule

    # ---- the recurrence; the output lexical layer and the print head; the attention head
    E(f'<path d="M{X6 - 110},{D + 90} H{X6 - 125} V{D + 30} H{330} V{D + 410} H{292}" fill="none" stroke="{C_CTRL}" stroke-width="1.6" marker-end="url(#arr-333333)"/>')
    text((X6 - 125 + 330) / 2, D + 22, "next morpheme: s(t) ← s(t+1)", 12, "middle", C_CTRL)
    E(f'<rect x="{X7 - 90}" y="{D + 440}" width="180" height="54" rx="8" fill="#eeeeee" stroke="#999"/>')
    text(X7, D + 463, "output lexical layer", 12, "middle", weight="bold"); text(X7, D + 481, "unit → string", 11, "middle", "#555")
    E(f'<path d="M{X7},{D + 412} V{D + 436}" fill="none" stroke="#777777" stroke-width="1" marker-end="url(#arr-777777)"/>')
    E(f'<rect x="{X7 - 90}" y="{D + 530}" width="180" height="54" rx="8" fill="#eeeeee" stroke="#999"/>')
    text(X7, D + 553, "print head", 12, "middle", weight="bold"); text(X7, D + 571, "appends when emit = 1", 11, "middle", "#555")
    E(f'<path d="M{X7},{D + 498} V{D + 526}" fill="none" stroke="#777777" stroke-width="1" marker-end="url(#arr-777777)"/>')
    curve(X6, em2, X7 - 90 - R, D + 557, 120, C_READ, 0.9)
    E(f'<rect x="70" y="{D + 664}" width="220" height="54" rx="8" fill="#eeeeee" stroke="{C_CTX}"/>')
    text(180, D + 686, "attention head", 12, "middle", weight="bold"); text(180, D + 704, "on one frame; moves on down / up", 10, "middle", "#555")
    E(f'<path d="M{X6},{up2 + R + 2} V{D + 600} H{1290} V{D + 691} H{296}" fill="none" stroke="{C_CTX}" stroke-width="1.1" stroke-dasharray="4,3" marker-end="url(#arr-{C_CTX.strip("#")})"/>')
    # the context reaches the decoder through the attention head: from the context panel down the right edge, under the decoder, to the head
    E(f'<path d="M{X7 + 100},400 H{W - 24} V{D + 760} H{40} V{D + 691} H{66}" fill="none" stroke="{C_CTX}" stroke-width="1.4" marker-end="url(#arr-{C_CTX.strip("#")})"/>')
    text(W / 2, D + 752, "the context reaches the decoder through the attention head: one frame at a time", 12, "middle", C_CTX)
    E(f'<path d="M{50},{D + 664} V{D + 210} H{66}" fill="none" stroke="{C_CTX}" stroke-width="1.4" marker-end="url(#arr-{C_CTX.strip("#")})"/>')
    # the sentence block of the encoder's final state is the decoder's at its first step
    text(X0 + 20, sent + 4, "← from the encoder's final state", 10, "start", "#777")

    # ---- decoder: layer labels, legend
    yl = D + 800
    for x, s in ((X1, "L1 conjunctions"), (X2, "L2 NOT"), (X3, "L3 printing rules"), (X4, "L4 gates"), (X5, "L5 read · lookup")):
        text(x, yl, s, 13, "middle", weight="bold")
    text(X3, yl + 16, "exactly one fires", 11, "middle", "#555"); text(X5, yl + 16, "ReLU, weights ±1", 11, "middle", "#555")
    lx, ly = 1340, D + 650
    panel(lx - 20, ly - 24, 520, 110, "#fafafa", stroke="#ccc")
    text(lx, ly, "the decoder: the mirror, one slot read", 13, weight="bold")
    entries = ((C_READ, 1.4, "printing: position → rule → next"), (C_READ, 1.1, "read: read gate ∧ frame → y(t)"), (C_READ, 1.1, "lookup: form gate ∧ verb → stem"),
               (C_CTX, 1.1, "down / up: the attention moves"), (C_NEG, 1.0, "nothing to read: FAIL"), (C_OTHER, 1.0, "other units, sparse"))
    for k, (col, lw, s) in enumerate(entries):
        cx, cy = lx + 262 * (k // 3), ly + 30 + 22 * (k % 3)
        edge(cx, cy, cx + 36, cy, col, lw, shrink=0); text(cx + 46, cy + 4, s, 12)

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
