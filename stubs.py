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
