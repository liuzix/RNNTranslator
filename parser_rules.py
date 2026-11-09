"""The parser layer, rule-based implementation.

The parser is a map from a token sequence (after the lexical layer has merged multiword tokens) to a frame: a Clause
whose registers hold the sentence's constituents.  This file is the reference implementation: a left-to-right machine
over a 4-token buffer, with no rewinds, a frame stack of bounded depth, and an ambiguity budget enforced by
parse_bounded.  parser_matrix.py is the second implementation of the same interface, compiled to sparse matrices.

The grammar's tables and the semantic layer (generate, Pred) live in rules_ja; this module is imported from the end
of rules_ja, after everything it needs is defined, so a star-import gives it those names.
"""
from rules_ja import *     # tables, lexical layer, Pred/generate (semantic layer); resolved at import time from rules_ja's tail

def agrees(form, subj_num):
    """a finite verb form fits the subject's number: 3sg needs a 3sg subject, base needs a non-3sg one, past fits any"""
    return not ((form == "3sg" and subj_num == "non3sg") or (form == "base" and subj_num == "3sg"))

class ParseFail(Exception):
    """the parser layer's crash: a reading died at token `pos`"""
    def __init__(self, msg, pos=None):
        super().__init__(msg); self.pos = pos

class Overflow(ParseFail):
    """the budget crash: frame stack too deep, too many live readings, or a reading held too long"""
    pass

BUFFER = 4        # lexical layer: the parser sees tokens t .. t+3 and nothing further

MAX_DEPTH = 2     # frame stack: main clause + two nested frames

WIDTH = 2         # ambiguity budget: at most two live readings

HOLD = 4          # ambiguity budget: a fork must be resolved within this many tokens

class Clause:
    def __init__(self):
        self.subj = None; self.subj_cls = "t"; self.subj_poss = ""; self.subj_adj = ""; self.subj_det = ""
        self.subj_num = None       # "3sg" / "non3sg" when the subject's head decides verb agreement, else None
        self.obj = None; self.obj_poss = ""; self.obj_adj = ""; self.obj_det = ""
        self.iobj = None
        self.pp = []               # list of (prep_entry, np string)
        self.time = ""; self.adv = ""; self.front = ""      # front: sentence adverbs before the subject (in fact , -> 実は、), the softener of a request
        self.verb = None           # (lemma, ja, conj)
        self.past = False; self.neg = False; self.modal = None; self.prog = False; self.perf = False
        self.copula = None         # ("adj", (ja, kind)) | ("np", string, cls) | ("pp", prep, np)
        self.there = False; self.question = False; self.wh = None; self.imp = False; self.please = False
        self.lets = False; self.want = None; self.degree = ""; self.causative = False
        self.purpose = None; self.purpose_clause = None; self.control = None; self.relative = False
        self.obj_particle = None; self.infinitive = None; self.pre_clause = ""; self.howabout = None; self.exclaim = None; self.wh_cls = "t"
        self.embedded = None       # another Clause (that-clause / infinitive complement)
        self.inner_of = None       # for an inner clause: what it is to its outer clause — "embedded" for a that-clause (set where it is created, so a trace can tell it from a control or purpose clause)
        self.compare = None        # ("as"|"than"|"noless", np) degree construction
        self.noless = False        # "no less ADJ than": the comparison reads に劣らず
        self.perf_had = False      # the perfect auxiliary was "had": a negative perfect then reads ていなかった, else ていない
        self.manner = None         # V-ing clause: stood looking at ... -> 〜ながら
        self.tag = False           # tag question: ..., isn't it ?
        self.sub = False           # a subordinate clause being parsed: stop at a comma or a conjunction
        self.coord = False         # a coordinated clause: keeps は on its subject
        self.suffix = ""           # sentence-final hearsay: ..., i hear -> と聞いている
        self.fixed_out = None      # a fixed sentence (thank you for NP) that bypasses the template
        self.so_that = None        # so ADJ that S: S becomes the main clause, the adjective clause its ので-reason
        self.from_seen = False     # a from-PP was parsed: the next to-PP is まで
        self.folded = False        # a string of this frame was generated from an inner clause (a relative clause, an infinitive, a that-clause folded into a noun phrase or the predicate): the frame cannot hold the clause itself

QUANTIFIERS = ("such", "all", "both")                                   # determiners that may precede an article (mirrored in np_module)
INDEFINITES = ("something", "anything", "nothing", "everything")        # pronouns an adjective may follow (mirrored in np_module)

class Parser:
    """parser layer: a left-to-right machine over the token buffer; every decision it cannot make from the
    buffer and its own state is a fork handed to the oracle (`choose`), and the driver above enforces the budget"""
    def __init__(self, tokens, choices=()):
        self.toks = tokens; self.i = 0
        self.main = Clause(); self.main.depth = 0; self.c = self.main
        self.phase = "PRE"
        self.np = None             # NP under construction: dict(det, poss, adj, num, noun, cls)
        self.np_role = None
        self.obj_cls = "t"; self.obj_reflexive = False    # the class of the last object noun phrase and whether it was a reflexive pronoun
        self.choices = list(choices); self.forks = []     # (position, name) of every fork met, in order

    @property
    def c(self): return self._c
    @c.setter
    def c(self, clause):
        if not hasattr(clause, "depth"):
            clause.depth = self._c.depth + 1
            if clause.depth > MAX_DEPTH: raise Overflow(f"OVERFLOW: frame stack depth {clause.depth} at '{self.peek()}'", self.i)
        self._c = clause

    def choose(self, name):
        """a two-way decision the buffer cannot settle: the oracle's answer if given, else option 0 (recorded as a fork)"""
        k = len(self.forks); self.forks.append((self.i, name))
        return self.choices[k] if k < len(self.choices) else 0

    def peek(self, k=0):
        if k >= BUFFER: raise RuntimeError(f"look-ahead {k} beyond the {BUFFER}-token buffer")
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None
    def readings(self, tok):
        return classify(tok) if tok is not None else []
    def has(self, tok, cls):
        return any(c == cls for c, _ in self.readings(tok))
    def take(self, cls):
        """consume the current token in class cls, return its payload"""
        for c, p in self.readings(self.peek()):
            if c == cls:
                self.i += 1; return p
        raise ParseFail(f"{self.phase}: expected {cls} at '{self.peek()}'", self.i)
    def fail(self, why):
        raise ParseFail(f"{self.phase}: {why} at '{self.peek()}'", self.i)

    # --- noun phrase: [det|poss|num] [adj]* noun | pron  -> (japanese, class) ---
    def parse_np(self, role=None):
        """role="subj": a subject — an object-only or reflexive pronoun is refused (subject case), and the head's
        grammatical number is left in self.np_number for the agreement check in the predicate"""
        self.np_number = None
        tok = self.peek(); r = dict(self.readings(tok))
        nxt = dict(self.readings(self.peek(1))) if self.peek(1) else {}
        nxt2 = dict(self.readings(self.peek(2))) if self.peek(2) else {}
        next_is_nominal = "noun" in nxt or "num" in nxt or ("adj" in nxt and ("noun" in nxt2 or "adj" in nxt2)) or ("adv" in nxt and "adj" in nxt2 and self.peek(3) and self.has(self.peek(3), "noun"))   # her sad face, her very old car; not "made her sad"
        det = poss = num = adjs = ""
        if tok == "a lot" and self.peek(1) == "of": det = "たくさんの"; self.i += 2
        elif tok == "a" and self.peek(1) == "lot" and self.peek(2) == "of": det = "たくさんの"; self.i += 3
        elif tok == "lots" and self.peek(1) == "of": det = "たくさんの"; self.i += 2
        elif "pron" in r and "p" in r["pron"][1] and next_is_nominal:
            poss = r["pron"][0] if r["pron"][1] == "p" else r["pron"][0] + "の"; self.i += 1
        elif "det" in r and (next_is_nominal or "pron" not in r):
            det = r["det"]; self.i += 1
            if tok == "no": self.c.neg = True                                   # no obligation / no money -> ...ない
            if tok in QUANTIFIERS and self.peek() in ("a", "an", "the"): self.i += 1          # such a thing / all the windows: the article is dropped (suite F)
        elif "pron" in r:
            self.i += 1
            ja, prole = r["pron"]
            if prole == "p": self.fail("possessive pronoun without noun")
            if role == "subj" and prole in ("o", "r", "op"): self.fail("subject case")      # *me went, *himself saw john
            self.last_head = None; self.np_number = number_of(tok)
            if tok in INDEFINITES and self.peek() and self.has(self.peek(), "adj") and not self.has(self.peek(), "noun") and not self.has(self.peek(), "verb"):   # something new -> 何か新しいもの (suite F)
                aja, ak = dict(self.readings(self.peek()))["adj"]; self.i += 1
                return ja + aja + ("な" if ak == "na" and not aja.endswith("な") else "") + "もの", "t"
            if self.peek() == "'s" and tok not in ("it", "he", "she", "i", "you", "we", "they", "this", "that", "who", "what", "somebody", "someone", "everybody", "everyone", "nobody", "anybody", "anyone", "one", "there", "here"):
                if self.peek(1) and (self.has(self.peek(1), "noun") or self.has(self.peek(1), "adj")):
                    self.i += 1; return self.parse_np_rest_after_poss(ja + "の")
                self.i += 1; return ja + "のもの", "t"
            return ja, "h"
        tok = self.peek(); r = dict(self.readings(tok))
        if "num" in r and not (tok == "one" and (det or poss) and not (self.peek(1) and self.has(self.peek(1), "noun"))):
            num = r["num"]; self.i += 1; tok = self.peek(); r = dict(self.readings(tok))
        while True:
            if "adv" in r and self.peek(1) and "adj" in dict(self.readings(self.peek(1))):
                adjs += r["adv"]; self.i += 1; tok = self.peek(); r = dict(self.readings(tok)); continue
            if "adj" in r and self.peek(1) is not None and ("noun" in dict(self.readings(self.peek(1))) or "adj" in dict(self.readings(self.peek(1)))):
                ja, kind = r["adj"]
                adjs += ja + ("な" if kind == "na" and not ja.endswith("な") else ""); self.i += 1
                tok = self.peek(); r = dict(self.readings(tok)); continue
            break
        if "noun" not in r:
            if num and not adjs and not det and not poss:          # a bare number as a noun phrase: "before ten"
                self.last_head = None
                return num + "時" if self.np_role == "pp_time" else num, "t"
            self.fail("noun phrase")
        ja, cls = r["noun"]; self.i += 1
        self.last_head = tok; self.np_number = None if num else number_of(tok)     # a measure phrase (ten years is a long time) takes either number
        # noun-noun compound: english homework, bus stop, road map
        nr2 = dict(self.readings(self.peek())) if self.peek() else {}
        aux_next = self.peek(1) and any(self.has(self.peek(1), k) for k in ("aux_be", "aux_do", "aux_have", "modal"))
        if "noun" in nr2 and not any(k in nr2 for k in ("aux_be", "aux_do", "aux_have", "modal", "prep", "adv", "det", "pron", "adj", "time", "conj", "num")) and ("verb" not in nr2 or aux_next):
            ja2, cls2 = nr2["noun"]; self.i += 1
            self.last_head = self.toks[self.i - 1]; self.np_number = number_of(self.last_head)
            ja, cls = ja + "の" + ja2, cls2
        # relative clause on the last NP of the sentence: the book (that) i bought / the man who came
        nxt = self.peek(); nr = dict(self.readings(nxt)) if nxt else {}
        rel_pron = nxt in ("who", "which", "that") and self.peek(1) and (self.has(self.peek(1), "verb") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "aux_have") or self.has(self.peek(1), "modal"))
        rel_subj = ("pron" in nr and "s" in nr["pron"][1] and self.peek(1) and (self.has(self.peek(1), "verb") or self.has(self.peek(1), "aux_have") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "modal")))
        if (rel_pron or rel_subj) and self.rel_ok():
            inner = Clause(); inner.relative = True
            if rel_pron:
                self.i += 1; inner.subj = None; inner.subj_cls = cls
            else:
                sja, scls = self.parse_np(role="subj"); inner.subj, inner.subj_cls, inner.subj_num = sja, scls, self.np_number
            saved = self.c; self.c = inner; head_number = self.np_number
            self.phase = "VP0"; self.predicate(); self.c = saved; self.np_number = head_number
            self.c.folded = True
            return generate(inner, polite=False, embedded=True)[:-1] + det + poss + num + adjs + ja, cls
        # possessive 's: John 's car
        if self.peek() == "'s" and (self.peek(1) is None or self.has(self.peek(1), "end") or self.has(self.peek(1), "prep") or self.peek(1) == ","):   # tom 's -> トムのもの
            self.i += 1; return det + poss + num + adjs + ja + "のもの", "t"
        if tok == "fact" and self.peek() == "that" and self.peek(1) and self.starts_np(self.peek(1)) and not (self.c.subj == "それ" and self.c.verb is None and self.c.copula is None):     # the fact that S -> Sという事実
            self.i += 1; inner = Clause(); saved = self.c; self.c = inner
            sja, scls = self.parse_np(role="subj"); inner.subj, inner.subj_cls, inner.subj_num = sja, scls, self.np_number
            self.phase = "VP0"; self.predicate(); self.c = saved
            self.c.folded = True
            return generate(inner, polite=False, embedded=True)[:-1] + "という" + ja, cls
        if self.peek() == "'s" and self.peek(1) and ("noun" in dict(self.readings(self.peek(1))) or "adj" in dict(self.readings(self.peek(1)))):
            self.i += 1
            inner_ja, cls2 = self.parse_np_rest_after_poss(ja + "の")
            return inner_ja, cls2
        # of-phrase: the top of the mountain -> 山の頂上
        if self.peek() == "of" and self.peek(1) and self.peek(1) not in ("course",):
            self.i += 1
            of_ja, _ = self.parse_np()
            self.last_head = tok; self.np_number = number_of(tok)
            return det + poss + of_ja + "の" + num + adjs + ja, cls
        if self.peek() == "and" and self.peek(1) and self.starts_np(self.peek(1)) and not self.has(self.peek(1), "verb") and self.choose("and after a noun phrase: clause | noun phrase") == 1:
            self.i += 1; ja2, cls2 = self.parse_np()                            # name and address -> 名前と住所
            self.np_number = "non3sg"
            return det + poss + num + adjs + ja + "と" + ja2, cls
        return det + poss + num + adjs + ja, cls
    def use_after_np(self, lemma, ja):
        """semantic feedback after an NP: if its head noun carries a use-verb for lemma, the NP is the object and the verb is replaced"""
        c = self.c; head = self.last_head
        if head in USE and lemma in USE[head] and not c.want:
            uja, uconj, upart, stative = USE[head][lemma]
            c.obj = ja; c.verb = (lemma + "+" + head, uja, uconj); c.obj_particle = upart
            if upart == "-": c.obj = None; c.obj_particle = None
            if stative: c.prog = True
            return True
        return False

    def rel_ok(self):
        """only attach a relative clause when this NP is the last constituent (no main predicate follows)"""
        return self.c.verb is not None or self.c.copula is not None or self.np_role == "last" or self.phase == "VP0" and self.c.subj is not None

    def parse_np_rest_after_poss(self, prefix):
        ja, cls = self.parse_np()
        return prefix + ja, cls

    def parse(self):
        self.pre()
        if self.phase != "DONE":
            self.fail("unparsed tail")
        return self.main

    def pre(self):
        c = self.c
        while True:
            tok = self.peek(); r = dict(self.readings(tok))
            if tok is None: self.fail("empty")
            if tok == "thank" and self.peek(1) == "you" or tok == "thank you":          # thank you (very much) for NP
                self.i += 2 if tok == "thank" else 1
                while self.peek() and self.has(self.peek(), "adv") and not self.has(self.peek(), "prep"): self.i += 1
                if self.peek() == "for" and self.peek(1) and self.starts_np(self.peek(1)):
                    self.i += 1; ja, cls = self.parse_np(); c.fixed_out = ja + "をありがとう。"
                else: c.fixed_out = "ありがとう。"
                self.tail_pp_adv(); return self.end()
            if "int" in r or "yesno" in r:
                self.i += 1
                if self.peek() == ",": self.i += 1
                continue
            if "time" in r: c.time += r["time"]; self.i += 1; continue
            if "adv" in r and self.peek(1) == "," and not any(k in r for k in ("verb", "aux_be", "noun", "adj", "prep", "pron", "det")):   # in fact , S / of course , S (suite C)
                c.front += r["adv"] + "、"; self.i += 2; continue
            if tok in SOFTENER and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base" and not self.starts_np(self.peek(1)):   # just wait (suite C)
                c.front += SOFTENER[tok]; self.i += 1; continue
            if "please" in r: c.please = True; self.i += 1; continue
            if "aux_be" in r and self.peek(1) == "there":                                 # is there NP ?
                c.past = r["aux_be"] == "past"; self.i += 2; c.there = True; c.question = True
                ja, cls = self.parse_np(role="subj"); c.subj, c.subj_cls, c.subj_num = ja, cls, self.np_number
                head0 = self.last_head
                if head0 in ("need", "reason", "time") and self.peek() == "for" and self.peek(1) and self.starts_np(self.peek(1)):
                    self.i += 1; xja, xcls = self.parse_np()
                    if self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb"):
                        self.i += 1; inner = Clause(); inner.subj = xja; inner.subj_cls = xcls
                        saved = self.c; self.c = inner; self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                        self.c.folded = True
                        c.subj = generate(inner, polite=False, embedded=True)[:-1] + {"need": "必要", "reason": "理由", "time": "時間"}[head0]
                    else:
                        c.pp.append((PREPS["for"], xja))
                self.tail_pp_adv(); return self.end()
            if tok == "the" and self.peek(1) and (self.peek(1) == "more" or self.has(self.peek(1), "adj") and dict(self.readings(self.peek(1)))["adj"][0].startswith("より")) and self.peek(2) and self.starts_np(self.peek(2)):   # the more S1 , the more S2
                self.i += 2
                inner = Clause(); inner.sub = True; saved = self.c; self.c = inner
                ja, cls = self.parse_np(role="subj"); inner.subj, inner.subj_cls, inner.subj_num = ja, cls, self.np_number
                self.phase = "VP0"; self.predicate(); self.c = saved
                if self.phase != "DONE": self.phase = "PRE"
                if self.peek() == ",": self.i += 1
                self.c.folded = True
                c.pre_clause += generate(inner, polite=False, embedded=True)[:-1] + "ほど"
                if self.peek() == "the" and self.peek(1) and (self.peek(1) == "more" or self.has(self.peek(1), "adj")):
                    self.i += 2; c.adv += "ますます"
                continue
            if "conj" in r and tok in SUBORD and c is self.main and not (tok == "when" and self.peek(1) and any(self.has(self.peek(1), k) for k in ("aux_be", "aux_do", "aux_have", "modal"))):
                c.pre_clause += self.sub_clause(tok); continue
            if tok == "being" and self.peek(1) and "adj" in dict(self.readings(self.peek(1))) and self.peek(2) == ",":
                ja, kind = dict(self.readings(self.peek(1)))["adj"]; self.i += 3
                c.pre_clause += adj_pred(ja, kind).finite(False, False, False) + "ので"; continue
            if "wh" in r:
                c.question = True; c.wh = r["wh"]; self.i += 1
                nxt = self.peek(); nr = dict(self.readings(nxt)) if nxt else {}
                if tok == "how" and nxt == "about":                                  # how about V-ing / NP ?
                    self.i += 1
                    if self.verb_form(self.peek()) == "ing":
                        inner = Clause(); self.c = inner
                        self.verb_phrase(expect_ing=True, from_copula=True); self.c = c
                        self.c.folded = True
                        c.howabout = generate(inner, polite=False, embedded=True)[:-1] + "の"
                    else:
                        ja, cls = self.parse_np(); c.howabout = ja
                    return self.end()
                if tok == "what" and nxt == "a" or tok == "how" and "adj" in nr and nxt not in ("many", "much", "far", "long", "often", "old") and self.peek(1) and self.starts_np(self.peek(1)):
                    # exclamation: what a bad boy he is ! / how beautiful the sun is !
                    c.question = False
                    if tok == "what":
                        ja, cls = self.parse_np(); c.exclaim = ja
                    else:
                        c.exclaim = nr["adj"][0]; self.i += 1
                    if self.peek() and self.starts_np(self.peek()):
                        ja, cls = self.parse_np(role="subj"); c.subj, c.subj_cls, c.subj_num = ja, cls, self.np_number
                        if self.has(self.peek(), "aux_be"): self.i += 1
                    return self.end()
                if tok == "what" and nxt == "time": c.wh = ("何時", "t"); self.i += 1
                elif tok in ("what", "which") and ("noun" in nr or "adj" in nr):          # wh-determiner
                    ja, cls = self.parse_np()
                    c.wh = (("どんな" if tok == "what" else "どの") + ja, "o")
                elif tok == "how" and nxt in ("many", "much") and self.peek(1) and self.starts_np(self.peek(1)):
                    self.i += 1; ja, cls = self.parse_np()
                    c.wh = (ja, "count"); c.wh_cls = cls
                elif tok == "how" and ("adj" in nr or "adv" in nr):                       # how long / how old
                    c.wh = ("何歳" if nxt == "old" else "どのくらい", "adv"); self.i += 1
                if self.peek() == "many" or self.peek() == "much": c.wh = ("いくつ" if self.peek() == "many" else "いくら", "o"); self.i += 1
                return self.after_wh()
            if "there" in r and self.peek(1) and (self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "aux_be3")):
                self.i += 1; c.there = True
                if self.has(self.peek(), "aux_be3"): self.i += 1; tense = "pres"
                else: tense = self.take("aux_be")
                c.past = tense == "past"
                if self.has(self.peek(), "not"): self.i += 1; c.neg = True
                ja, cls = self.parse_np(role="subj"); c.subj, c.subj_cls, c.subj_num = ja, cls, self.np_number
                head0 = self.last_head
                if head0 in ("need", "reason", "time") and self.peek() == "for" and self.peek(1) and self.starts_np(self.peek(1)):   # is there any need for me to go ?
                    self.i += 1; xja, xcls = self.parse_np()
                    if self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb"):
                        self.i += 1; inner = Clause(); inner.subj = xja; inner.subj_cls = xcls
                        saved = self.c; self.c = inner; self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                        self.c.folded = True
                        c.subj = generate(inner, polite=False, embedded=True)[:-1] + {"need": "必要", "reason": "理由", "time": "時間"}[head0]
                    else:
                        c.pp.append((PREPS["for"], xja))
                self.tail_pp_adv()
                return self.end()
            if "let" in r and self.peek(1) == "'s":
                self.i += 2; c.lets = True
                return self.verb_phrase(expect_base=True)
            if "let" in r and self.peek(1) in ("me", "us") and self.peek(2) and self.has(self.peek(2), "verb"):
                self.i += 2; c.imp = True; c.please = True; c.causative = True
                return self.verb_phrase(expect_base=True)
            if ("aux_do" in r or "aux_be" in r or "aux_have" in r or "modal" in r) and self.peek(1) and self.starts_np(self.peek(1)):
                return self.question_inversion()
            if "aux_do" in r and self.peek(1) and self.has(self.peek(1), "not"):      # don't V (imperative)
                self.i += 2; c.imp = True; c.neg = True
                return self.verb_phrase(expect_base=True)
            if "verb" in r and not self.starts_np(tok) and any(f == "base" for _, f, _, _ in [p for cc, p in self.readings(tok) if cc == "verb"]) and (self.peek(1) is None or not self.has(self.peek(1), "aux_be")):
                c.imp = True
                return self.verb_phrase(expect_base=True)
            break
        # subject
        ja, cls = self.parse_np(role="subj"); c.subj, c.subj_cls, c.subj_num = ja, cls, self.np_number
        self.phase = "VP0"
        return self.predicate()

    def object_starts(self, tok):
        """does tok begin an object noun phrase of the verb? (not a PP, not an adverb unless it modifies a noun, not a bare
        adjective after a copular verb)"""
        if tok is None: return False
        c = self.c; r = dict(self.readings(tok))
        return self.starts_np(tok) and "prep" not in r and ("adv" not in r or self.peek(1) and self.has(self.peek(1), "noun") and not (self.peek(2) and self.starts_np(self.peek(2)))) and not ("adj" in r and "noun" not in r and "det" not in r and c.verb and c.verb[0] in VC["copular"])

    def starts_np(self, tok):
        r = dict(self.readings(tok))
        if "pron" in r or "det" in r or "noun" in r or "num" in r: return True
        if "adj" in r and not ("verb" in r or "adv" in r):                      # bare adjective + noun: sad movies, hot food
            for k in range(BUFFER - 1):
                if self.peek(k) == tok:
                    return self.peek(k + 1) is not None and self.has(self.peek(k + 1), "noun") and not self.has(self.peek(k + 1), "verb")
        return False

    def after_wh(self):
        c = self.c; tok = self.peek(); r = dict(self.readings(tok))
        if c.wh[1] == "count" and "aux_be" in r and self.peek(1) == "there":        # how many N are there ?
            c.past = r["aux_be"] == "past"; self.i += 2; c.there = True
            c.subj, c.subj_cls = c.wh[0], c.wh_cls; c.wh = ("いくつ", "count")
            self.tail_pp_adv(); return self.end()
        if c.wh[1] in ("o", "count") and c.wh[0] not in ("何", "いくつ", "いくら") and not ("aux_do" in r or "modal" in r or "aux_have" in r) and ("verb" in r or "aux_be" in r and not (self.peek(1) and self.starts_np(self.peek(1)))):   # which bus goes there ?
            c.subj, c.subj_cls = c.wh[0], "t"; c.wh = (c.wh[0], "s")
            self.phase = "VP0"; return self.predicate()
        if c.wh[1] == "s" and not ("aux_do" in r or "modal" in r or "aux_have" in r) and ("verb" in r and "aux_be" not in r or "aux_be" in r and not (self.peek(1) and self.starts_np(self.peek(1)))):              # who came ? / who is in the room ?
            c.subj, c.subj_cls = None, "h"
            self.phase = "VP0"; return self.predicate()
        if ("aux_be" in r or "aux_be3" in r) and self.peek(1) and (self.starts_np(self.peek(1)) or self.has(self.peek(1), "wh")):
            tense = r.get("aux_be", "pres"); self.i += 1; c.past = tense == "past"
            ja, cls = self.parse_np(role="subj"); c.subj, c.subj_cls, c.subj_num = ja, cls, self.np_number
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            while self.peek() and self.has(self.peek(), "time"): c.time += dict(self.readings(self.peek()))["time"]; self.i += 1
            self.copula_predicate(allow_empty=True)
            if c.copula is None: c.copula = ("wh",)
            return self.end()
        if "aux_do" in r or "modal" in r or "aux_have" in r or "aux_be" in r:
            return self.question_inversion()
        self.fail("wh question")

    def question_inversion(self):
        c = self.c; tok = self.peek(); r = dict(self.readings(tok))
        c.question = True
        if "aux_do" in r:
            c.past = r["aux_do"] == "past"; self.i += 1
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            ja, cls = self.parse_np(role="subj"); c.subj, c.subj_cls, c.subj_num = ja, cls, self.np_number
            return self.verb_phrase(expect_base=True)
        if "modal" in r:
            c.modal, _ = r["modal"]; self.i += 1
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            ja, cls = self.parse_np(role="subj"); c.subj, c.subj_cls, c.subj_num = ja, cls, self.np_number
            while self.peek() and self.has(self.peek(), "adv") and not self.has(self.peek(), "verb") and not self.has(self.peek(), "aux_be"):
                c.adv += dict(self.readings(self.peek()))["adv"]; self.i += 1
            if tok == "would" and self.peek() == "mind" and self.peek(1) and self.verb_form(self.peek(1)) == "ing":   # would you mind V-ing ?
                self.i += 1; c.want = "mind_request"; c.modal = None
                return self.verb_phrase(expect_ing=True)
            if tok in ("would", "will", "could", "can") and ja == "あなた" and self.peek() == "please" and self.peek(1) and self.verb_form(self.peek(1)) == "base":   # would you please open the door ? -> 開けてくださいませんか (suite H)
                self.i += 1; c.want = "request"; c.modal = None
                return self.verb_phrase(expect_base=True)
            return self.verb_phrase(expect_base=True)
        if "aux_have" in r:
            c.past = r["aux_have"] == "past"; c.perf = True; c.perf_had = r["aux_have"] == "past"; self.i += 1
            ja, cls = self.parse_np(role="subj"); c.subj, c.subj_cls, c.subj_num = ja, cls, self.np_number
            return self.verb_phrase(expect_pp=True)
        if "aux_be" in r:
            c.past = r["aux_be"] == "past"; self.i += 1
            ja, cls = self.parse_np(role="subj"); c.subj, c.subj_cls, c.subj_num = ja, cls, self.np_number
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            self.copula_predicate()
            return self.end()
        self.fail("inversion")

    def predicate(self):
        """after the subject"""
        c = self.c; tok = self.peek(); r = dict(self.readings(tok))
        while "adv" in r and "verb" not in r and "aux_be" not in r:
            c.adv += r["adv"]; self.i += 1; tok = self.peek(); r = dict(self.readings(tok))
        if "time" in r: c.time += r["time"]; self.i += 1; tok = self.peek(); r = dict(self.readings(tok))
        if "aux_be" in r and c.subj_num and not agrees({"is": "3sg", "are": "base"}.get(tok, "past"), c.subj_num): self.fail("agreement")   # *they is, *he are
        if "aux_be" in r or "aux_be3" in r:
            c.past = (r.get("aux_be") == "past"); self.i += 1
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            self.copula_predicate()
            return self.end()
        if "aux_do" in r and self.peek(1) and self.has(self.peek(1), "not"):
            c.past = r["aux_do"] == "past"; self.i += 2; c.neg = True
            while self.peek() and self.has(self.peek(), "adv") and not self.has(self.peek(), "verb"):
                c.adv += dict(self.readings(self.peek()))["adv"]; self.i += 1
            return self.verb_phrase(expect_base=True)
        if "aux_do" in r and self.peek(1) and self.verb_form(self.peek(1)) == "base" and not self.has(self.peek(1), "noun"):   # emphatic: did come
            c.past = r["aux_do"] == "past"; self.i += 1; c.adv += "確かに"
            return self.verb_phrase(expect_base=True)
        if tok == "may" and self.peek(1) == "as" and self.peek(2) == "well":   # may as well V
            self.i += 3; c.want = "better"
            return self.verb_phrase(expect_base=True)
        if tok in ("had", "'d") and self.peek(1) == "better":                    # had better V
            self.i += 2; c.want = "better"
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            return self.verb_phrase(expect_base=True)
        if tok == "need" and self.peek(1) and self.has(self.peek(1), "not"):   # need not V
            self.i += 2; c.want = "neednot"
            return self.verb_phrase(expect_base=True)
        if "modal" in r:
            c.modal, c.past = r["modal"]; self.i += 1
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            if self.peek() == "to": self.i += 1
            while self.peek() and self.has(self.peek(), "adv") and not self.has(self.peek(), "verb") and not self.has(self.peek(), "aux_be"):
                c.adv += dict(self.readings(self.peek()))["adv"]; self.i += 1
            if c.modal == "would" and self.peek() == "mind" and self.peek(1) and self.verb_form(self.peek(1)) == "ing":   # would you mind V-ing ?
                self.i += 1; c.want = "mind_request"; c.modal = None
                return self.verb_phrase(expect_ing=True)
            if self.peek() in ("have",) and self.peek(1) and "pp" in self.verb_forms(self.peek(1)):
                self.i += 1; c.past = True; c.perf = True
                if c.modal == "can" and c.neg: c.want = "canthave"; c.neg = False; c.modal = None
                return self.verb_phrase(expect_pp=True)
            if self.has(self.peek(), "aux_be") and self.peek(1) and not self.has(self.peek(1), "verb"):   # will be late
                self.i += 1
                self.copula_predicate(); return self.end()
            return self.verb_phrase(expect_base=True)
        if ("aux_have" in r or tok == "'ve") and self.peek(1) and (self.has(self.peek(1), "not") and self.peek(2) and self.has(self.peek(2), "verb") or "pp" in self.verb_forms(self.peek(1))) and self.peek(1) != "to":
            c.past = True; c.perf = True; c.perf_had = tok == "had"; self.i += 1
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            return self.verb_phrase(expect_pp=True)
        if ("aux_have" in r or tok == "'ve") and self.peek(1) and self.has(self.peek(1), "adv") and not self.has(self.peek(1), "verb"):   # have just now finished
            j = 1
            while self.peek(j) and self.has(self.peek(j), "adv") and not self.has(self.peek(j), "verb"): j += 1
            if "pp" in self.verb_forms(self.peek(j)):
                c.past = True; c.perf = True; c.perf_had = tok == "had"; self.i += 1
                while self.peek() and self.has(self.peek(), "adv") and not self.has(self.peek(), "verb"):
                    c.adv += dict(self.readings(self.peek()))["adv"]; self.i += 1
                return self.verb_phrase(expect_pp=True)
        if tok in ("have", "has", "had") and self.peek(1) == "to":          # have to
            c.past = tok == "had"; c.modal = "must"; self.i += 2
            return self.verb_phrase(expect_base=True)
        if "verb" in r:
            return self.verb_phrase()
        self.fail("predicate")

    def verb_forms(self, tok):
        return {f for cc, (l, f, ja, k) in [(cc, p) for cc, p in self.readings(tok) if cc == "verb"]}
    def verb_form(self, tok):
        fs = [f for cc, (l, f, ja, k) in [(cc, p) for cc, p in self.readings(tok) if cc == "verb"]]
        return fs[0] if fs else None

    def copula_predicate(self, allow_empty=False):
        c = self.c; tok = self.peek(); r = dict(self.readings(tok))
        if c.imp and ("adj" in r or (tok, self.peek(1)) in ADJ_PREP) and not self.starts_np(tok):       # be quiet / don't be afraid: the imperative of an adjective (〜にして / 〜がらないで) is not rendered
            self.fail("imperative adjective")
        while "time" in r and "adv" not in r: c.time += r["time"]; self.i += 1; tok = self.peek(); r = dict(self.readings(tok))
        while "adv" in r and (r["adv"] in ("とても", "本当に", "かなり", "も", "まだ", "もう", "いつも") or self.peek(1) and (self.has(self.peek(1), "adj") or self.starts_np(self.peek(1)) and "adj" not in r) and not self.has(self.peek(1), "noun")):
            c.degree = "すぎ" if tok == "too" else c.degree + r["adv"]; self.i += 1; tok = self.peek(); r = dict(self.readings(tok))
        if tok == "the" and self.peek(1) and self.has(self.peek(1), "adj") and not (self.peek(2) and (self.has(self.peek(2), "noun") or self.has(self.peek(2), "adj"))):   # she is the happiest: a superlative without its noun (suite E)
            self.i += 1; tok = self.peek(); r = dict(self.readings(tok))
        if tok == "no" and self.peek(1) == "less" and self.peek(2) and self.has(self.peek(2), "adj") and self.peek(3) == "than":         # no less kind than NP -> NPに劣らず優しい (suite E)
            c.noless = True; self.i += 2; tok = self.peek(); r = dict(self.readings(tok))      # the adjective and the than-phrase follow as usual
        if tok == "going" and self.peek(1) == "to" and self.peek(2) and self.has(self.peek(2), "verb"):
            self.i += 2; c.want = "intend"
            return self.verb_phrase(expect_base=True, from_copula=True)
        if tok == "to" and self.peek(1) and self.verb_form(self.peek(1)) == "base":        # he is to arrive tomorrow
            self.i += 1; c.want = "be_to"
            return self.verb_phrase(expect_base=True, from_copula=True)
        if tok == "about" and self.peek(1) == "to" and self.peek(2) and self.verb_form(self.peek(2)) == "base":   # i 'm about to tell you -> 話すところだ (suite G)
            self.i += 2; c.want = "about_to"
            return self.verb_phrase(expect_base=True, from_copula=True)
        if tok == "as" and self.peek(1) and self.has(self.peek(1), "adj") and self.peek(2) == "as" and self.peek(3) and self.starts_np(self.peek(3)):
            adj = dict(self.readings(self.peek(1)))["adj"]; self.i += 3
            ja, cls = self.parse_np(); c.compare = ("as", ja); c.copula = ("adj", adj); self.tail_pp_adv(); return
        if tok == "more" and self.peek(1) and self.has(self.peek(1), "adj"):
            ja, k = dict(self.readings(self.peek(1)))["adj"]; self.i += 1
            self.toks[self.i] = self.toks[self.i]        # keep token; reading below sees the plain adjective
            c.degree += "より"
        if "verb" in r and self.verb_form(tok) == "ing":                     # progressive
            c.prog = True
            return self.verb_phrase(expect_ing=True, from_copula=True)
        if (tok, self.peek(1)) in ADJ_PREP and self.peek(2) and (self.starts_np(self.peek(2)) or self.verb_form(self.peek(2)) == "ing"):
            ja, kind, part = ADJ_PREP[(tok, self.peek(1))]; self.i += 2
            if self.verb_form(self.peek()) == "ing" and not self.starts_np(self.peek()):    # afraid of making mistakes -> 間違いをすることを (suite D)
                inner = Clause(); saved = self.c; self.c = inner
                self.verb_phrase(expect_ing=True, from_copula=True); self.c = saved
                self.c.folded = True
                c.obj = generate(inner, polite=False, embedded=True)[:-1] + "こと"
            else:
                ja2, cls2 = self.parse_np(); c.obj = ja2
            c.obj_particle = part; c.copula = ("idiom", ja, kind); self.tail_pp_adv(); return
        if "verb" in r and self.verb_form(tok) == "pp" and not self.starts_np(tok):   # passive
            self.fail("passive")
        if (tok, self.peek(1)) in ADJ_PREP and self.peek(2) and (self.starts_np(self.peek(2)) or self.verb_form(self.peek(2)) == "ing"):
            ja, kind, part = ADJ_PREP[(tok, self.peek(1))]; self.i += 2
            if self.verb_form(self.peek()) == "ing":
                inner = Clause(); saved = self.c; self.c = inner
                self.verb_phrase(expect_ing=True, from_copula=True); self.c = saved
                self.c.folded = True
                c.obj = generate(inner, polite=False, embedded=True)[:-1] + "の"
            else:
                ja2, cls2 = self.parse_np(); c.obj = ja2
            c.obj_particle = part; c.copula = ("idiom", ja, kind); self.tail_pp_adv(); return
        if "adj" in r and not (self.peek(1) and "noun" in dict(self.readings(self.peek(1)))):
            c.copula = ("adj", r["adj"]); self.i += 1
            if self.peek() == "than" and self.peek(1) and (self.starts_np(self.peek(1)) or self.has(self.peek(1), "time")):
                self.i += 1
                if self.has(self.peek(), "time"): ja = dict(self.readings(self.peek()))["time"]; self.i += 1
                else: ja, cls = self.parse_np()
                c.compare = ("noless" if c.noless else "than", ja); self.tail_pp_adv(); return
            if self.peek() == "enough" and self.peek(1) == "to" and self.peek(2) and self.has(self.peek(2), "verb"):
                self.i += 2; inner = Clause(); saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                c.infinitive = ("enough", inner); self.tail_pp_adv(); return
            if c.degree == "すぎ" and self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb"):
                self.i += 1; inner = Clause(); saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                c.infinitive = ("too", inner); self.tail_pp_adv(); return
            if self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb"):   # glad to see / hard to say
                self.i += 1; inner = Clause(); saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                c.infinitive = ("extraposed" if c.subj in ("それ", None) else "adj_to", inner); self.tail_pp_adv(); return
            if self.peek() == "of" and self.peek(1) and self.starts_np(self.peek(1)) and c.subj == "それ":       # it was careless of you to V
                self.i += 1; xja, xcls = self.parse_np()
                if not (self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb")): self.fail("ADJ of X to V")
                self.i += 1; inner = Clause(); inner.subj = xja; inner.subj_cls = xcls
                saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                c.infinitive = ("of_to", inner); self.tail_pp_adv(); return
            if self.peek() == "for" and self.peek(1) and self.starts_np(self.peek(1)) and c.subj == "それ":      # it is hard for X to V: dummy subject decides
                self.i += 1
                xja, xcls = self.parse_np()
                if not (self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb")): self.fail("for X to V")
                self.i += 1; inner = Clause(); inner.subj = xja; inner.subj_cls = xcls
                saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                c.infinitive = ("extraposed", inner); self.tail_pp_adv(); return
            if self.peek() == "that" or (self.peek() and self.starts_np(self.peek()) and self.peek(1) and (self.has(self.peek(1), "verb") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "modal") or self.has(self.peek(1), "aux_do"))):
                so_that = self.peek() == "that" and c.degree.endswith("とても") and self.toks[self.i - 2] == "so"
                if self.peek() == "that": self.i += 1
                inner = Clause(); inner.inner_of = "embedded"; self.c = inner
                ja, cls = self.parse_np(role="subj"); inner.subj, inner.subj_cls, inner.subj_num = ja, cls, self.np_number
                self.phase = "VP0"; self.predicate(); self.c = c
                if so_that: c.so_that = inner                                   # so hard that S -> とても難しいので S
                elif c.subj == "それ": c.infinitive = ("extraposed", inner)     # it is true that S -> Sのは本当だ
                else: c.embedded = inner
                return
            self.tail_pp_adv(); return
        if "prep" in r:
            p = r["prep"]; self.i += 1
            self.np_role = "pp_time" if tok in ("before", "after", "at", "until", "till", "around", "by") else None
            ja, cls = self.parse_np(); self.np_role = None
            c.copula = ("pp", p, ja); self.tail_pp_adv(); return
        if "loc" in r:                                                       # is here: the place with the existence particle
            c.copula = ("pp", PREPS["at"], r["loc"]); self.i += 1; self.tail_pp_adv(); return
        if "adv" in r:                                                       # is out
            c.copula = ("adv", r["adv"]); self.i += 1; self.tail_pp_adv(); return
        if "end" in r and allow_empty:
            return
        if tok == "the" and self.peek(1) in ("first", "last", "only") and self.peek(2) == "to" and self.peek(3) and self.has(self.peek(3), "verb"):   # the first to V
            w = self.peek(1); self.i += 3
            inner = Clause(); saved = self.c; self.c = inner
            self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
            inner.past = c.past
            self.c.folded = True
            c.copula = ("np", {"first": "最初に", "last": "最後に", "only": "唯一"}[w] + generate(inner, polite=False, embedded=True)[:-1] + "人", "h")
            self.tail_pp_adv(); return
        if self.starts_np(tok):
            ja, cls = self.parse_np()
            c.copula = ("np", ja, cls)
            if c.neg and self.peek() == "but" and self.peek(1) and self.starts_np(self.peek(1)):                      # not a lawyer but a doctor
                self.i += 1; ja2, cls2 = self.parse_np(); c.copula = ("np", ja + "ではなく" + ja2, cls2); c.neg = False
                self.tail_pp_adv(); return
            if self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base":   # (it is) a long time to wait -> 待つのには
                if c.subj_cls == "h" and c.subj not in ("それ", None): self.fail("copula infinitive")   # you must be a fool to do that: a judgment (〜するとは), not rendered
                self.i += 1; inner = Clause(); saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                self.c.folded = True
                c.copula = ("np", generate(inner, polite=False, embedded=True)[:-1] + "のには" + ja, cls)
                self.tail_pp_adv(); return
            if c.subj == "それ" and self.peek() == "that" and self.peek(1) and self.starts_np(self.peek(1)):          # it is a pity that S
                self.i += 1; inner = Clause(); self.c = inner
                ja2, cls2 = self.parse_np(role="subj"); inner.subj, inner.subj_cls, inner.subj_num = ja2, cls2, self.np_number
                self.phase = "VP0"; self.predicate(); self.c = c
                c.infinitive = ("extraposed", inner); return
            self.tail_pp_adv(); return
        self.fail("copula predicate")

    def verb_phrase(self, expect_base=False, expect_pp=False, expect_ing=False, from_copula=False):
        c = self.c; tok = self.peek(); r = dict(self.readings(tok))
        if "verb" not in r: self.fail("verb")
        if expect_pp and tok == "been" and self.peek(1) == "to" and self.peek(2) and self.starts_np(self.peek(2)): self.fail("have been to")   # the experiential perfect (行ったことがある) is not rendered
        readings = [p for cc, p in self.readings(tok) if cc == "verb"]
        lemma, form, ja, conj = readings[0]
        if not (expect_base or expect_pp or expect_ing) and c.subj_num and not any(agrees(f, c.subj_num) for _, f, _, _ in readings):
            self.fail("agreement")                                                 # *he have a car, *she go to school
        if lemma == "be" and not from_copula:
            self.i += 1
            self.copula_predicate(); return self.end()
        if lemma == "be" and from_copula:
            self.i += 1
            self.copula_predicate(); return
        self.i += 1
        if not expect_base and not expect_pp and not expect_ing:
            c.past = form == "past"
        c.verb = (lemma, ja, conj)
        for k in (3, 2, 1):                       # multiword verb idioms: "get in touch with", "look forward to", "wait for"
            key = (lemma, tuple(self.toks[self.i:self.i + k]))
            if key in VERB_IDIOM and self.i + k < len(self.toks) and (self.starts_np(self.toks[self.i + k]) or self.verb_form(self.toks[self.i + k]) == "ing" or self.has(self.toks[self.i + k], "wh")):
                ja, conj, part = VERB_IDIOM[key]; self.i += k
                c.verb = (lemma + " " + " ".join(key[1]), ja, conj); c.obj_particle = part
                if self.has(self.peek(), "wh"):                    # account for why S
                    c.obj = self.embedded_question(); c.obj_particle = ""
                elif self.verb_form(self.peek()) == "ing":          # look forward to seeing you
                    inner = Clause(); saved = self.c; self.c = inner
                    self.verb_phrase(expect_ing=True, from_copula=True); self.c = saved
                    self.c.folded = True
                    c.obj = generate(inner, polite=False, embedded=True)[:-1] + "の"
                else:
                    ja2, cls2 = self.parse_np(); c.obj = ja2
                self.tail_pp_adv()
                if from_copula: return
                return self.end()
        if (lemma, self.peek()) in PHRASAL and not (self.peek() in PREPS and self.peek(1) and self.starts_np(self.peek(1)) and self.peek() in ("for", "at", "to", "from", "with", "on") and (lemma, self.peek()) not in PHRASAL_NP):
            ja, conj = PHRASAL[(lemma, self.peek())]; self.i += 1
            c.verb = (lemma + " " + self.toks[self.i - 1], ja, conj)
        if " " in c.verb[0] and self.peek() == "that" and self.peek(1) and self.starts_np(self.peek(1)):      # made out that S
            self.i += 1; return self.embedded_after(c) if not from_copula else (self.embedded_after(c), None)[1]
        if lemma == "make" and self.peek() == "it" and self.peek(1) in ("rule", "a"):                           # make it (a) rule (never) to V
            self.i += 2
            if self.peek() == "rule": self.i += 1
            neg = False
            if self.peek() == "never": neg = True; self.i += 1
            if self.peek() != "to": self.fail("make it a rule")
            self.i += 1; inner = Clause(); inner.neg = neg; saved = self.c; self.c = inner
            self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
            self.c.folded = True
            c.verb = ("make it a rule", generate(inner, polite=False, embedded=True)[:-1] + "ことにしている", "fixed")
            self.tail_pp_adv(); return self.end() if not from_copula else None
        if lemma == "go" and self.peek() in USE and "go" in USE[self.peek()]:
            uja, uconj, upart, _ = USE[self.peek()][lemma]
            c.obj = LEX[self.peek()][0][1]; c.obj_particle = upart; self.i += 1
            c.verb = ("go+" + self.toks[self.i - 1], uja, uconj)
            self.tail_pp_adv(); return self.end()
        if lemma == "let" and self.peek() in ("me", "us") and self.peek(1) and self.has(self.peek(1), "verb"):
            self.i += 1; c.imp = True; c.please = True; c.causative = True
            return self.verb_phrase(expect_base=True)
        # complements that change the predicate
        if lemma == "want" and self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb"):
            self.i += 1; c.want = "tai"
            return self.verb_phrase(expect_base=True)
        if lemma == "would like" or (lemma == "like" and c.modal == "will" and self.peek() == "to"):
            self.i += 1; c.want = "tai"; c.modal = None
            return self.verb_phrase(expect_base=True)
        if lemma in VC["to_comp"] and self.peek() in ("to",) and self.peek(1) and self.has(self.peek(1), "verb"):
            self.i += 1; c.want = lemma
            return self.verb_phrase(expect_base=True)
        if lemma in VC["not_to"] and self.has(self.peek(), "not") and self.peek(1) == "to" and self.peek(2) and self.has(self.peek(2), "verb"):   # promised not to tell
            self.i += 2; c.want = lemma; c.neg_inner = True
            return self.verb_phrase(expect_base=True)
        if lemma in VC["ing_comp"] and self.peek() and self.verb_form(self.peek()) == "ing":
            c.want = lemma
            return self.verb_phrase(expect_ing=True)
        if c.verb[0] in VC["ing_comp"] and self.peek() and self.verb_form(self.peek()) == "ing" and not self.starts_np(self.peek()):   # give up going abroad -> 行くことを諦めた (suite D); give up smoking keeps the noun 喫煙
            c.want = c.verb[0]
            return self.verb_phrase(expect_ing=True)
        bare_clause = (self.peek() and self.starts_np(self.peek()) and lemma not in VC["that_only"] and not (self.has(self.peek(), "pron") and PRONOUNS[self.peek()][1] in ("o", "r", "op"))
                       and not (lemma in VC["perception"] and self.peek(1) and self.verb_form(self.peek(1)) == "ing")
                       and self.peek(1) and (self.has(self.peek(1), "verb") and not self.has(self.peek(1), "noun") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "aux_do") or self.has(self.peek(1), "modal") or self.has(self.peek(1), "aux_have")))
        if lemma in VC["that_comp"] and (self.peek() == "that" or bare_clause):               # a that-less clause: its subject is not an object pronoun (heard him sing) and its verb is not also a noun (heard a sound)
            if self.peek() == "that": self.i += 1
            inner = Clause(); inner.inner_of = "embedded"; c.embedded = inner
            self.c = inner
            ja, cls = self.parse_np(role="subj"); inner.subj, inner.subj_cls, inner.subj_num = ja, cls, self.np_number
            self.phase = "VP0"
            self.predicate_noend()
            self.c = c
            return self.end()
        if lemma in ("tell", "ask") and self.peek() and self.starts_np(self.peek()) and not (lemma == "ask" and self.has(self.peek(), "wh")):
            ja, cls = self.parse_np()
            if self.use_after_np(lemma, ja):                    # tell a lie / ask a question: the noun carries the verb
                self.tail_pp_adv(); return self.end() if not from_copula else None
            c.iobj = ja
            if self.peek() == "that": self.i += 1; return self.embedded_after(c)
            if self.peek() and (self.has(self.peek(), "wh") or self.peek() in ("if", "whether")):
                c.obj = self.embedded_question(); c.obj_particle = ""; self.tail_pp_adv()
                return self.end() if not from_copula else None
            if self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base":   # told me to go
                self.i += 1
                inner = Clause(); inner.subj = ja; inner.subj_cls = cls
                saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True)
                self.c = saved; c.iobj = None; c.control = inner
                self.tail_pp_adv(); return self.end() if not from_copula else None
            if self.peek() and self.starts_np(self.peek()):                    # asked me a question / told him the truth
                ja2, cls2 = self.parse_np()
                if not self.use_after_np(lemma, ja2): c.obj = ja2
            self.tail_pp_adv(); return self.end() if not from_copula else None
        if lemma in VC["wh_comp"] and self.peek() and (self.has(self.peek(), "wh") or self.peek() in ("if", "whether")) and not (self.peek() == "if" and lemma not in VC["if_comp"]):
            if lemma == "ask" and self.peek() and self.starts_np(self.peek()): ja, cls = self.parse_np(); c.iobj = ja
            c.obj = self.embedded_question(); c.obj_particle = ""
            if lemma == "wonder": c.verb = ("wonder", "と思う", "fixed"); c.obj_particle = ""
            self.tail_pp_adv()
            return self.end() if not from_copula else None
        self.tail()
        if from_copula:
            return
        return self.end()

    def sub_clause(self, conj):
        """'conj S V ...' -> Japanese subordinate clause string; the reading dies (ParseFail) if the clause does not parse"""
        save = (self.i, self.c, self.phase, self.np_role)
        self.i += 1
        inner = Clause(); inner.sub = True; self.c = inner
        if self.verb_form(self.peek()) == "ing" and conj in ("while", "before", "after", "when", "since", "once"):     # while walking
            inner.subj = None
            self.verb_phrase(expect_ing=True, from_copula=True)
        elif self.peek() == "there" and self.peek(1) and self.has(self.peek(1), "aux_be"):
            self.i += 1; r = dict(self.readings(self.peek())); inner.past = r["aux_be"] == "past"; self.i += 1; inner.there = True
            ja, cls = self.parse_np(role="subj"); inner.subj, inner.subj_cls, inner.subj_num = ja, cls, self.np_number; self.tail_pp_adv()
        else:
            ja, cls = self.parse_np(role="subj"); inner.subj, inner.subj_cls, inner.subj_num = ja, cls, self.np_number
            self.phase = "VP0"; self.predicate()
        self.c, self.np_role = save[1], save[3]
        if self.phase != "DONE": self.phase = save[2]
        if conj == "whether" and self.peek() == "or" and self.peek(1) == "not": self.i += 2
        if self.peek() == ",": self.i += 1
        if self.peek() == "then": self.i += 1
        return subord_form(inner, conj)

    def coord(self, conj):
        """main clause + conj + clause: the first clause becomes a te-form (and) / が (but) / ので (so) link"""
        c = self.c
        save = (self.i, self.phase)
        self.i += 1
        second = Clause(); second.coord = True
        self.c = second
        if self.peek() and self.starts_np(self.peek()) and self.peek(1) and (self.has(self.peek(1), "verb") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "aux_do") or self.has(self.peek(1), "aux_have") or self.has(self.peek(1), "modal")):
            ja, cls = self.parse_np(role="subj"); second.subj, second.subj_cls, second.subj_num = ja, cls, self.np_number
        else:
            second.subj = None; second.subj_cls = c.subj_cls
        if self.has(self.peek(), "not") and self.peek(1) and self.has(self.peek(1), "verb"): self.i += 1; second.neg = True
        self.phase = "VP0"; self.predicate()
        self.c = c
        if self.phase != "DONE": self.phase = save[1]
        first = Clause(); first.__dict__.update(c.__dict__); first.pre_clause = ""; first.suffix = ""
        del second.depth
        link = COORD[conj]
        if link == "":
            head = clause_te(first)
        else:
            head = generate(first, polite=False)[:-1] + link
        pre = c.pre_clause
        c.__dict__.update(second.__dict__)
        c.imp = first.imp or second.imp; c.please = first.please or second.please; c.lets = first.lets
        c.pre_clause = pre + head
        return True

    def embedded_question(self):
        """where the bus stop is / which you want / if he will come -> ...か / ...かどうか"""
        tok = self.peek(); r = dict(self.readings(tok))
        inner = Clause(); inner.sub = True; saved = self.c; self.c = inner
        if tok in ("if", "whether"):
            self.i += 1; ja, cls = self.parse_np(role="subj"); inner.subj, inner.subj_cls, inner.subj_num = ja, cls, self.np_number
            self.phase = "VP0"; self.predicate(); self.c = saved
            if self.peek() == "or" and self.peek(1) == "not": self.i += 2
            self.c.folded = True
            body = generate(inner, polite=False, embedded=True)[:-1]
            return body + "かどうか"
        inner.wh = r["wh"]; inner.question = True; self.i += 1
        nxt = self.peek(); nr = dict(self.readings(nxt)) if nxt else {}
        if tok in ("what", "which") and ("noun" in nr or "adj" in nr) and not ("verb" in nr or "aux_be" in nr):
            ja, cls = self.parse_np(); inner.wh = (("どんな" if tok == "what" else "どの") + ja, "o")
        elif tok == "how" and nxt in ("many", "much") and self.peek(1) and self.starts_np(self.peek(1)):
            self.i += 1; ja, cls = self.parse_np(); inner.wh = ("いくつの" + ja, "o")
        elif tok == "how" and ("adj" in nr or "adv" in nr):
            inner.wh = ("何歳" if nxt == "old" else "どのくらい", "adv"); self.i += 1
        if inner.wh[1] in ("o", "loc", "adv", "t") and self.peek() == "to" and self.peek(1) and self.verb_form(self.peek(1)) == "base":   # which way to go / how to do it
            self.i += 1; self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
            self.c.folded = True
            body = generate(inner, polite=False, embedded=True)[:-1]
            return (body[:-1] if body.endswith("か") else body) + "べきか"
        if self.peek() and self.starts_np(self.peek()):
            ja, cls = self.parse_np(role="subj"); inner.subj, inner.subj_cls, inner.subj_num = ja, cls, self.np_number
            nxt = self.peek(); nr = dict(self.readings(nxt)) if nxt else {}
            after = self.peek(1); ar = dict(self.readings(after)) if after else {}
            if "aux_be" in nr and (after is None or "end" in ar or "conj" in ar or after == ","):    # where the bus stop is
                inner.past = nr["aux_be"] == "past"; self.i += 1; inner.copula = ("wh",)
            else:
                self.phase = "VP0"; self.predicate()
        else:                                                   # what had happened / who came
            inner.subj = None; inner.wh = (inner.wh[0], "s"); self.phase = "VP0"; self.predicate()
        self.c = saved
        self.c.folded = True
        body = generate(inner, polite=False, embedded=True)[:-1]
        return body if body.endswith("か") else body + "か"

    def embedded_after(self, c):
        inner = Clause(); inner.inner_of = "embedded"; c.embedded = inner; self.c = inner
        ja, cls = self.parse_np(role="subj"); inner.subj, inner.subj_cls, inner.subj_num = ja, cls, self.np_number
        self.phase = "VP0"; self.predicate_noend(); self.c = c
        return self.end()

    def predicate_noend(self):
        """predicate of an embedded clause; end() returns without consuming for an inner clause"""
        self.predicate()

    def tail(self):
        """object, indirect object, particles, PPs, adverbs after a verb"""
        c = self.c
        tok = self.peek(); r = dict(self.readings(tok))
        # came to help me / went to see her: purpose infinitive after a motion verb
        if tok == "to" and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base" and c.verb and c.verb[0] in VC["motion"]:
            self.i += 1
            inner = Clause(); inner.purpose = True
            saved = self.c; self.c = inner
            self.verb_phrase(expect_base=True, from_copula=True)
            self.c = saved; c.purpose_clause = inner
            self.tail_pp_adv(); return
        if tok == "to" and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base" and c.verb:
            self.i += 1
            inner = Clause(); inner.purpose = "tame"
            saved = self.c; self.c = inner
            self.verb_phrase(expect_base=True, from_copula=True)
            self.c = saved; c.purpose_clause = inner
            self.tail_pp_adv(); return
        if tok == "to" and self.peek(1) and self.starts_np(self.peek(1)) and c.verb and c.verb[0] in VC["dative"]:
            pass
        if tok == "not" and self.peek(1) == "only" and self.peek(2) and self.starts_np(self.peek(2)):     # not only X but (also) Y
            self.i += 2; a, _ = self.parse_np()
            if self.peek() != "but": self.fail("not only ... but")
            self.i += 1
            if self.peek() == "also": self.i += 1
            b, _ = self.parse_np(); c.obj = a + "だけでなく" + b + "も"; c.obj_particle = ""
            self.tail_pp_adv(); return
        if tok == "both" and self.peek(1) and self.starts_np(self.peek(1)):           # both A and B
            self.i += 1; a, _ = self.parse_np()
            if self.peek() == "and": self.i += 1; b, _ = self.parse_np(); c.obj = a + "も" + b; c.obj_particle = "も"
            else: c.obj = "両方の" + a
            self.tail_pp_adv(); return
        if self.verb_form(tok) == "ing" and c.verb and c.verb[0] in VC["posture"]:   # stood looking at
            inner = Clause(); saved = self.c; self.c = inner
            self.verb_phrase(expect_ing=True, from_copula=True); self.c = saved
            c.manner = inner; self.tail_pp_adv(); return
        if self.object_starts(tok):
            ja, cls = self.parse_np()
            if c.verb and c.verb[0] in VC["ditrans"] and self.object_starts(self.peek()):   # show me her room: the first noun phrase is the indirect object, pinned by suite B
                c.iobj = ja; ja, cls = self.parse_np()
            c.obj = ja; self.obj_cls = cls; self.obj_reflexive = tok in PRONOUNS and PRONOUNS[tok][1] == "r"
            head = self.last_head
            lemma0 = c.verb[0] if c.verb else None
            if c.verb and c.verb[0] in VC["causative"] and self.peek() and self.verb_form(self.peek()) == "base" and (not self.has(self.peek(), "noun") or self.peek(1) and (self.starts_np(self.peek(1)) or self.has(self.peek(1), "prep") or self.has(self.peek(1), "adv"))):   # made me write
                inner = Clause(); inner.subj = ja; inner.subj_cls = cls
                saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                c.obj = None; c.control = inner; c.causative = c.verb[0]
                self.tail_pp_adv(); return
            for k in (3, 2, 1):                                                   # take NP into account / have NP to himself
                tail_toks = tuple(self.toks[self.i:self.i + k]) if self.i + k <= len(self.toks) else None
                if tail_toks is None or not c.verb: continue
                keyed = tuple("SELF" if (t in PRONOUNS and PRONOUNS[t][1] == "r") else t for t in tail_toks)
                if (c.verb[0], keyed) in IDIOM_TAIL:
                    uja, uconj, upart = IDIOM_TAIL[(c.verb[0], keyed)]; self.i += k
                    c.verb = (c.verb[0] + " _ " + " ".join(keyed), uja, uconj); c.obj_particle = upart
                    self.tail_pp_adv(); return
            if c.verb and (c.verb[0], ("as",)) in VERB_IDIOM and self.peek() == "as" and self.peek(1) and (self.has(self.peek(1), "adj") or self.starts_np(self.peek(1))):   # acknowledged it as true
                uja, uconj, upart = VERB_IDIOM[(c.verb[0], ("as",))]; self.i += 1
                if self.has(self.peek(), "adj") and not self.starts_np(self.peek()):
                    aja, ak = dict(self.readings(self.peek()))["adj"]; self.i += 1; comp = adj_pred(aja, ak).finite(False, False, False)
                else:
                    comp, _ = self.parse_np()
                c.verb = (c.verb[0] + " as", uja, uconj); c.obj_particle = upart; c.adv += comp + "と"
                self.tail_pp_adv(); return
            if c.verb and c.verb[0] == "have" and self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base":   # have no obligation to help
                self.i += 1; inner = Clause(); saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                self.c.folded = True
                c.obj = generate(inner, polite=False, embedded=True)[:-1] + c.obj
                self.tail_pp_adv(); return
            if c.verb and c.verb[0] in VC["perception"] and self.peek() and self.verb_form(self.peek()) == "ing":   # hear someone talking
                inner = Clause(); inner.subj = ja; inner.subj_cls = cls
                saved = self.c; self.c = inner
                self.verb_phrase(expect_ing=True, from_copula=True); self.c = saved
                self.c.folded = True
                c.obj = generate(inner, polite=False, embedded=True)[:-1] + "の"
                self.tail_pp_adv(); return
            if head in USE and lemma0 in USE[head] and not c.want:        # the noun's use-verb replaces the light verb
                uja, uconj, upart, stative = USE[head][lemma0]
                c.verb = (lemma0 + "+" + head, uja, uconj); c.obj_particle = upart
                if upart == "-": c.obj = None; c.obj_particle = None           # take a rest -> 休む: the noun is the verb
                if stative: c.prog = True
            # told me to go / want you to come / asked him to wait
            if self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base" and c.verb and c.verb[0] in VC["control"]:
                self.i += 1
                inner = Clause(); inner.subj = c.obj; inner.subj_cls = cls
                saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True)
                self.c = saved; c.obj = None; c.control = inner
                self.tail_pp_adv(); return
        self.tail_pp_adv()

    def tail_pp_adv(self):
        c = self.c
        while True:
            tok = self.peek(); r = dict(self.readings(tok))
            if tok is None or "end" in r: return
            if c.sub and (tok == "," or tok == "or" and self.peek(1) and self.has(self.peek(1), "not") or "conj" in r and not (tok in PREPS and self.peek(1) and self.starts_np(self.peek(1)) and self.choose(tok + " inside a clause: ends it | its PP") == 1)): return
            if tok == "," and self.peek(1) and self.has(self.peek(1), "pron") and self.peek(2) in HEARSAY and self.peek(3) and self.has(self.peek(3), "end"):   # ..., i hear .
                c.suffix = HEARSAY[self.peek(2)]; self.i += 3; continue
            if "conj" in r and tok in COORD and c is self.main and (tok != "so" or self.peek(1) and self.starts_np(self.peek(1))):
                if tok == "while" and self.choose("while: subordinate | coordinate") == 0:
                    c.pre_clause += self.sub_clause(tok); continue
                self.coord(tok); continue
            if "conj" in r and tok in SUBORD and c is self.main and (tok not in PREPS or self.choose(tok + ": PP | clause") == 1):
                c.pre_clause += self.sub_clause(tok); continue
            if tok == "," and self.peek(1) and self.has(self.peek(1), "conj"): self.i += 1; continue
            if "time" in r: c.time += r["time"]; self.i += 1; continue
            if tok == "in" and self.peek(1) and "num" in dict(self.readings(self.peek(1))) and self.peek(2) in DURATION and self.peek(2) not in ("times", "time"):   # in 50 minutes -> ５０分で
                c.adv += dict(self.readings(self.peek(1)))["num"] + DURATION[self.peek(2)] + "で"; self.i += 3; continue
            if "prep" in r and self.peek(1) and self.has(self.peek(1), "time") and not self.starts_np(self.peek(1)):
                p = r["prep"]; self.i += 1; tja = dict(self.readings(self.peek()))["time"]; self.i += 1
                p = {"until": ("まで", "まで"), "till": ("まで", "まで"), "by": ("までに", "までに"), "since": ("から", "から"), "from": ("から", "から"), "before": ("の前に", "の前に"), "after": ("の後で", "の後で")}.get(tok, p)
                c.pp.append((p, tja)); continue
            if "prep" in r and self.peek(1) and self.starts_np(self.peek(1)):
                p = r["prep"]; self.i += 1
                self.np_role = "pp_time" if tok in ("before", "after", "at", "until", "till", "around", "by") else None
                ja, cls = self.parse_np(); self.np_role = None
                head = self.last_head
                if tok == "to" and head in USE and c.verb and c.verb[0] in USE[head] and not c.want:     # came to a decision -> 決定に達した: the noun carries the verb (suite C)
                    uja, uconj, upart, stative = USE[head][c.verb[0]]
                    c.verb = (c.verb[0] + "+" + head, uja, uconj); p = (upart, upart)
                    if stative: c.prog = True
                if cls == "t" and ja.endswith("時"):
                    p = {"at": ("に", "に"), "before": ("前に", "前に"), "after": ("後に", "後に"), "around": ("ごろに", "ごろに"), "by": ("までに", "までに")}.get(tok, p)
                if tok == "with" and self.peek() and "pp" in self.verb_forms(self.peek()) and (self.peek(1) is None or self.has(self.peek(1), "end") or self.peek(1) == ","):   # with his eyes closed
                    lem, form, vja, vconj = [p for cc, p in self.readings(self.peek()) if cc == "verb"][0]; self.i += 1
                    pv = make_pred(vja, vconj) if vconj not in ("fixed",) else None
                    c.adv += ja + "を" + (pv.te() if pv else vja); continue
                if tok == "with" and cls == "t" and c.verb and c.verb[0] not in VC["comitative"]:
                    p = ("で", "で")                                               # instrument: with a pen -> ペンで
                if tok == "to" and c.from_seen:
                    p = ("まで", "まで"); c.from_seen = False
                if tok == "from": c.from_seen = True
                if p == PREPS["to"] and c.verb and c.verb[0] in VC["dative"]:
                    c.iobj = ja
                else:
                    c.pp.append((p, ja))
                continue
            if tok in ("on", "off", "up", "down", "out", "away", "back", "in") and c.verb and (c.verb[0], tok) in PARTICLE_AFTER and (self.peek(1) is None or "end" in dict(self.readings(self.peek(1))) or "prep" in dict(self.readings(self.peek(1))) or "time" in dict(self.readings(self.peek(1)))):
                ja, conj = PARTICLE_AFTER[(c.verb[0], tok)]; c.verb = (c.verb[0] + " " + tok, ja, conj); self.i += 1; continue
            if "prep" in r and c.wh and self.peek(1) and "end" in dict(self.readings(self.peek(1))):   # where do you come from ?
                c.wh = (c.wh[0] + r["prep"][1 if c.wh[1] == "loc" else 0], "pp"); self.i += 1; continue
            if tok == "to" and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base" and c.verb and c.purpose_clause is None and c.control is None:
                self.i += 1                                      # purpose infinitive after a prepositional phrase
                inner = Clause(); inner.purpose = "tame" if c.verb[0] not in VC["motion"] else True
                saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True)
                self.c = saved; c.purpose_clause = inner; continue
            if tok == "as" and self.peek(1) and self.has(self.peek(1), "adv") and self.peek(2) == "as" and self.peek(3) and self.starts_np(self.peek(3)):   # as early as my mother
                c.adv += dict(self.readings(self.peek(1)))["adv"]; self.i += 3
                ja, cls = self.parse_np(); c.compare = ("as", ja); continue
            if tok == "than" and self.peek(1) and (self.starts_np(self.peek(1)) or self.has(self.peek(1), "time")):
                self.i += 1
                if self.has(self.peek(), "time"): ja = dict(self.readings(self.peek()))["time"]; self.i += 1
                else: ja, cls = self.parse_np()
                c.compare = ("than", ja); continue
            if ("num" in r or tok in ("a", "an", "several", "many", "few", "some")) and self.peek(1) in DURATION:   # two years / several weeks / a few days
                n = r["num"] if "num" in r else {"a": "一", "an": "一", "several": "数", "many": "何", "few": "数", "some": "数"}[tok]
                c.adv += n + DURATION[self.peek(1)]; self.i += 2; continue
            if tok == "for" and self.peek(1) and ("num" in dict(self.readings(self.peek(1))) or self.peek(1) in ("a", "an", "several", "many", "few", "some")) and self.peek(2) in DURATION:
                self.i += 1; continue
            if tok == "," and self.peek(1) and (self.peek(1) in CONTRACTED or self.has(self.peek(1), "aux_do") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "modal")):   # ..., isn't it ?
                if self.peek(2) == "'t" and self.peek(3) and self.has(self.peek(3), "pron"): c.tag = True; self.i += 4; continue
                if self.peek(2) and self.has(self.peek(2), "pron") and not self.starts_np(self.peek(3)): c.tag = True; self.i += 3; continue
            if "adj" in r and self.peek(1) == "and" and self.peek(2) == tok and c.verb and c.verb[0] in VC["change"]:   # longer and longer
                ja, k = r["adj"]; c.adv += "ますます"; c.copula = ("adj", (ja[2:] if ja.startswith("より") else ja, k)); self.i += 3; continue
            if "loc" in r: c.pp.append((PREPS["at"], r["loc"])); self.i += 1; continue      # wait there -> そこで, go there -> そこに: as a PP at that place
            if tok == "too" and self.peek(1) and self.has(self.peek(1), "adv") and not self.has(self.peek(1), "verb"):   # too fast -> あまりに速く (the excess, not も)
                c.adv += "あまりに" + dict(self.readings(self.peek(1)))["adv"]; self.i += 2; continue
            if "adv" in r: c.adv += r["adv"]; self.i += 1; continue
            if "adj" in r and c.verb and c.verb[0] in VC["copular"]:
                c.copula = ("adj", r["adj"]); self.i += 1; continue
            if "adj" in r and c.verb and c.verb[0] == "make" and c.obj and not self.starts_np(tok) and r["adj"][1] in ("a", "na") and self.obj_cls in ("h", "a") and not self.obj_reflexive:   # made her (very) sad -> 悲しくさせた: a person made to feel so (suite G)
                c.verb = ("make+" + tok, adj_causative(*r["adj"]), "i"); self.i += 1; continue
            if tok == "," : self.i += 1; continue
            self.fail("tail")

    def end(self):
        if self.phase == "DONE" and self.i >= len(self.toks):   # an inner clause already consumed the end token
            return
        tok = self.peek(); r = dict(self.readings(tok))
        if "end" in r:
            self.i += 1
            if r["end"] == "?": self.main.question = True
            if self.i < len(self.toks): self.fail("two sentences")
            self.phase = "DONE"; return
        if self.c is not self.main:            # embedded / relative clause: let the outer clause finish
            return
        self.fail("end")

def parse_bounded(toks):
    """the ambiguity layer above the parser: simulates WIDTH live readings that must resolve within HOLD tokens.
    A fork splits the reading; the reading that dies first (ParseFail) loses; both alive past the window -> Overflow;
    a second fork while two readings are alive -> Overflow."""
    def run(choices):
        p = Parser(toks, choices)
        try: return p, p.parse(), None
        except ParseFail as e: return p, None, e
    choices = []
    while True:
        p0, f0, e0 = run(choices)
        pending = p0.forks[len(choices):]
        if not pending:
            if f0 is not None: return f0
            raise e0
        fpos, fname = pending[0]
        p1, f1, e1 = run(choices + [1])
        deadline = min(fpos + HOLD, len(toks) - 1)
        d0 = e0.pos if e0 is not None and e0.pos is not None and e0.pos <= deadline else None   # death inside the window
        d1 = e1.pos if e1 is not None and e1.pos is not None and e1.pos <= deadline else None
        if d0 is None and d1 is None:
            if deadline == len(toks) - 1 and (f0 is not None or f1 is not None):     # the sentence ended inside the window: resolved by preference
                if f0 is not None: return f0
                choices.append(1); continue
            raise Overflow(f"OVERFLOW: two readings held past {HOLD} tokens at '{toks[fpos]}' ({fname})", fpos)
        if d0 is not None and d1 is not None:
            raise (e0 if d0 >= d1 else e1)
        winner, loser_death, wp = (1, d0, p1) if d0 is not None else (0, d1, p0)
        for pos, name in wp.forks[len(choices) + 1:]:
            if pos < loser_death: raise Overflow(f"OVERFLOW: a third reading opened at '{toks[pos]}' ({name}) while two were alive", pos)
        choices.append(winner)
