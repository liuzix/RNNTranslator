"""Stubs: the interfaces of work that is not done yet, so that the acceptance tests in harness/test_acceptance.py can be
written against them today.  Every method records its call and returns TODO, a value that is empty, false and unequal
to everything, so an acceptance test runs to its assertions and fails there instead of at the first call.

harness/check_todo.py requires every method of every class here to be exercised by an acceptance test, every class to
be named by an item of TODO.yaml that is not done, and every acceptance test of an item that is not done to fail.  When
an item is done its stub leaves this file and the acceptance test imports the real module.
"""
import collections

CALLS = collections.Counter()      # "Class.method" -> times called during the acceptance run

class _Todo:
    """the value a stub returns: empty, false, unequal to anything, and it absorbs attribute access and calls"""
    def __bool__(self): return False
    def __iter__(self): return iter(())
    def __len__(self): return 0
    def __eq__(self, other): return False
    def __ne__(self, other): return True
    def __hash__(self): return 0
    def __getattr__(self, name): return self
    def __getitem__(self, key): return self
    def __call__(self, *a, **k): return self
    def __add__(self, other): return self
    def __radd__(self, other): return self
    def __contains__(self, item): return False
    def __repr__(self): return "TODO"

TODO = _Todo()

class Stub:
    item = None                    # the TODO.yaml item this stub belongs to
    def _hit(self, name): CALLS[f"{type(self).__name__}.{name}"] += 1; return TODO

class HandModules(Stub):
    """the PP machine, predicate and copula_predicate translated by hand with complete guards (item hand-modules)"""
    item = "hand-modules"
    def pp_rules(self): return self._hit("pp_rules")
    def predicate_rules(self): return self._hit("predicate_rules")
    def copula_rules(self): return self._hit("copula_rules")

class ClauseFlush(Stub):
    """inner clauses in one frame: printing at clause end, clearing, the outer predicate in tag slots (item inner-clauses)"""
    item = "inner-clauses"
    def tag_slots(self): return self._hit("tag_slots")                       # -> the tag slot names of the state
    def flush_rules(self): return self._hit("flush_rules")                   # -> the rules that run a clause's printing positions when it closes

class Layer0Lexical(Stub):
    """the lexical layer as layer 0 of the cell, with the verb block as an input (item lexical-layer0)"""
    item = "lexical-layer0"
    def hit_units(self, comp): return self._hit("hit_units")                 # -> the units that compute table hits from the window and the verb block

class Canon(Stub):
    """reduced ordered decision diagrams as the canonical form of the rules (item normal-form)"""
    item = "normal-form"
    def diagram(self, rules, prev, order): return self._hit("diagram")       # -> the diagram of one control point under a literal order
    def complete(self, diagram): return self._hit("complete")                # -> every path ends in a terminal or FAIL
    def equal(self, d1, d2): return self._hit("equal")                       # -> the two diagrams are the same
