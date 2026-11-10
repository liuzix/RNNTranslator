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

class Coordination(Stub):
    """the clause before a conjunction as a frame of its own, linked to the next (item coordination)"""
    item = "coordination"
    def link_rules(self): return self._hit("link_rules")                     # -> the rules at the conjunction: link and flush the first clause's frame, clear, read the second

class RelativeClause(Stub):
    """a relative clause as a frame printed before the noun phrase it modifies (item relative-clauses)"""
    item = "relative-clauses"
    def splice_rules(self): return self._hit("splice_rules")                 # -> the link on the modified role, and the decoder's move down before that role's noun phrase

class Infinitive(Stub):
    """an infinitive complement of a copula as a frame linked to the copula's (item infinitives)"""
    item = "infinitives"
    def link_rules(self): return self._hit("link_rules")                     # -> the rules per kind of link: te before the adjective, のは, のには

class PurposeClause(Stub):
    """a purpose clause as a frame printed before the main verb (item purpose-clauses)"""
    item = "purpose-clauses"
    def purpose_rules(self): return self._hit("purpose_rules")               # -> the link and the decoder's rules: the masu stem with に, the plain form with ために

class Gerund(Stub):
    """a gerund or manner clause as a frame printed where the object or the manner goes (item gerunds)"""
    item = "gerunds"
    def gerund_rules(self): return self._hit("gerund_rules")                 # -> the link and the decoder's rules: the plain form with のを or こと, the masu stem with ながら

class ControlClause(Stub):
    """an object-control clause: the object moved into the inner clause's frame as its subject (item control-clauses)"""
    item = "control-clauses"
    def control_rules(self): return self._hit("control_rules")               # -> the move of the object across the boundary and the decoder's rules: ように, or the causative with the outer tense
