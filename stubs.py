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

class ClauseModes(Stub):
    """print modes that end an inner clause at a stem form instead of a finite form (item clause-modes)"""
    item = "clause-modes"
    def modes(self): return self._hit("modes")                               # -> the names of the modes: te, the masu stem, the plain form, the causative with the outer tense
    def stem_rules(self): return self._hit("stem_rules")                     # -> the printing rules of those modes: the predicate's stem form, then the construction's morpheme

class Coordination(Stub):
    """the clause before a conjunction flushed in the link's mode (item coordination)"""
    item = "coordination"
    def link_rules(self): return self._hit("link_rules")                     # -> the rules at the conjunction: flush the first clause, clear, read the second

class SubordinateClause(Stub):
    """a sentence-initial subordinate clause flushed at the comma with its conjunction (item subordinate-clauses)"""
    item = "subordinate-clauses"
    def order_rules(self): return self._hit("order_rules")                   # -> the rules at the comma: flush the clause with とき / なら / ので / が, clear, read the main clause

class RelativeClause(Stub):
    """the modified noun phrase held while the relative clause is read and printed before it (item relative-clauses)"""
    item = "relative-clauses"
    def hold_rules(self): return self._hit("hold_rules")                     # -> the rules that hold the noun phrase, flush the clause in the plain form and print the phrase after it

class Infinitive(Stub):
    """an infinitive complement of a copula flushed in its mode (item infinitives)"""
    item = "infinitives"
    def mode_rules(self): return self._hit("mode_rules")                     # -> the flush rules per mode: te before the adjective, のは, のには

class PurposeClause(Stub):
    """a purpose clause flushed before the main verb held in the tags (item purpose-clauses)"""
    item = "purpose-clauses"
    def purpose_rules(self): return self._hit("purpose_rules")               # -> the flush rules: the masu stem with に, the plain form with ために

class Gerund(Stub):
    """a gerund or manner clause flushed before the verb held in the tags (item gerunds)"""
    item = "gerunds"
    def gerund_rules(self): return self._hit("gerund_rules")                 # -> the flush rules: the plain form with のを or こと, the masu stem with ながら

class ControlClause(Stub):
    """an object-control clause: the object moved into the inner subject, the clause flushed (item control-clauses)"""
    item = "control-clauses"
    def control_rules(self): return self._hit("control_rules")               # -> the move of the object and the flush rules: ように, or the causative with the outer tense
