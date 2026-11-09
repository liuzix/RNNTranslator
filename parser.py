"""The parser interface: a map from a token sequence to a frame.

    tokens (after the lexical layer's multiword merge)  ->  Frame (a Clause: the sentence's registers)

Two implementations share it:
    RuleParser    parser_rules.Parser under the ambiguity budget (parse_bounded) — the reference
    MatrixParser  parser_matrix: the same machine compiled to sparse matrices, run one token at a time
Both raise ParseFail (or its subclass Overflow, the budget crash) when the sentence is refused.
"""
import rules_ja as R
from parser_rules import ParseFail, Overflow, Clause, Parser, parse_bounded

Frame = Clause

class ParserInterface:
    name = "abstract"
    def parse(self, tokens):
        """tokens: list of merged English tokens -> Frame; raises ParseFail"""
        raise NotImplementedError
    def translate(self, tokens):
        """the Japanese for a sentence; the matrix implementation overrides this with its print head"""
        return R.generate(self.parse(tokens))

class RuleParser(ParserInterface):
    name = "rules"
    def parse(self, tokens):
        return parse_bounded(list(tokens))

def frames_equal(a, b):
    """two frames are the same if the semantic layer renders them identically"""
    return R.generate(a) == R.generate(b)

def build_matrix_parser(sentences, verbose=False):
    """compile the parser into the matrix implementation: the clause-level control is extracted from the rule parser's
    behaviour on `sentences` (lists of merged tokens); the noun-phrase module (np_module) is hand-translated and complete"""
    import parser_matrix as PM
    comp = PM.build_hybrid(sentences, verbose=verbose)
    return PM.MatrixParser(comp)        # translate() is the cell's own print head, not generate() on the frame
