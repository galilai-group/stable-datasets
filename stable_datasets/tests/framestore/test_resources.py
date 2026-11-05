"""Property tests for framestore/resources.py: amounts, Budget.allocate (minimums first, then round-robin growth), reserve and
release, the gate a budget hands out, and ByteGate (sequential model, and many threads at once).

    pytest tests/framestore/test_resources.py
"""

import random
import threading

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from stable_datasets.framestore import resources
from stable_datasets.framestore.resources import (
    Budget,
    ByteGate,
    ResourceError,
    Resources,
    Stage,
    parse_amount,
    resolve,
)


def budget(memory, shm=None, gpu=None, monkeypatch=None):
    """A Budget over a node with these free bytes (None: no /dev/shm, no GPU), read from nothing real."""
    mp = monkeypatch
    mp.setattr(resources, "memory_available", lambda: memory)
    mp.setattr(resources, "shm_free", lambda path="/dev/shm": shm)
    mp.setattr(resources, "gpu_free", lambda device=0: gpu)
    mp.setattr(resources, "process_bytes", lambda pid=None: 0)
    mp.setattr(resources, "usable_cpus", lambda: 8)
    return Budget("test", Resources(cpu=1.0, memory=1.0, gpu_memory=1.0))


# ---- amounts -------------------------------------------------------------------------------------------------
@given(st.integers(1, 10**6))
def test_count_string_and_number_agree(n):
    assert parse_amount(str(n)) == parse_amount(n)


@given(st.floats(0.01, 1.0))
def test_fraction_string_and_number_agree(f):
    assert parse_amount(repr(f)) == parse_amount(f)


@given(st.integers(1, 4096), st.sampled_from(["k", "m", "g", "t"]), st.booleans(), st.booleans())
def test_sizes(x, unit, binary, upper):
    s = f"{x}{unit}{'i' if binary else ''}b"
    kind, v = parse_amount(s.upper() if upper else s)
    assert kind == "absolute"
    assert v == x * (1024 if binary else 1000) ** "_kmgt".index(unit)


@given(st.one_of(st.floats(0.01, 1.0), st.integers(1, 10**12), st.floats(1.01, 1e12)), st.integers(0, 10**13))
def test_resolve_never_exceeds_what_is_available(v, available):
    r = resolve(v, available)
    assert 0 <= r <= available


@pytest.mark.parametrize("v", [0, -1, -0.5, "", "abc", "16XB", "0"])
def test_bad_amounts_refused(v):
    with pytest.raises(ValueError):
        parse_amount(v)


@given(st.integers(1, 256), st.one_of(st.floats(0.01, 1.0), st.integers(1, 512)))
def test_job_cpus_keep_two_free(monkeypatch_session, usable, want):
    monkeypatch_session.setattr(resources, "usable_cpus", lambda: usable)
    n = resources.job_cpus(want)
    assert 1 <= n <= max(1, usable - resources.SPARE_CPUS)
    assert n <= max(1, int(resolve(want, usable)))


# ---- allocate ------------------------------------------------------------------------------------------------
stages = st.lists(
    st.builds(
        Stage,
        name=st.text("abcdef", min_size=1, max_size=4),
        want=st.integers(0, 40),
        unit=st.integers(0, 10**9),
        least=st.integers(0, 4),
        shm=st.booleans(),
        gpu=st.booleans(),
    ),
    min_size=1,
    max_size=5,
)
pools_left = st.tuples(
    st.integers(0, 2 * 10**10),
    st.one_of(st.none(), st.integers(0, 10**10)),
    st.one_of(st.none(), st.integers(0, 10**10)),
)


def pools(b, s):
    return b._pools(s)


@given(stages, pools_left)
def test_allocate(monkeypatch_session, ss, left):
    memory, shm, gpu = left
    assume(gpu is not None or not any(s.gpu for s in ss))  # no GPU: a GPU stage raises (tested below)
    b = budget(memory, shm, gpu, monkeypatch_session)
    before = {p: b._left(p) for s in ss for p in pools(b, s)}
    need = {p: sum(s.least * s.unit for s in ss if p in pools(b, s)) for p in before}
    if any(need[p] > before[p] for p in before):
        with pytest.raises(ResourceError):
            b.allocate(ss)
        assert {p: b.left[p] for p in before} == before, "a refused allocation took something"
        return
    n = b.allocate(ss)
    used = {p: sum(k * s.unit for k, s in zip(n, ss) if p in pools(b, s)) for p in before}
    for p in before:
        assert used[p] <= before[p], f"{p}: allocated more than the budget had"
        assert b.left[p] == before[p] - used[p], f"{p}: the budget's books are off"
    for k, s in zip(n, ss):
        assert s.least <= k <= max(s.want, s.least)
        if k < s.want:  # it stopped growing because one of its pools could not take another unit
            assert any(b.left[p] < s.unit for p in pools(b, s)), f"{s.name} could have had more"
    # round-robin: stages that share their pools and unit grow by the same amount, to within one
    for i, s in enumerate(ss):
        for j, t in enumerate(ss):
            if i < j and pools(b, s) == pools(b, t) and s.unit == t.unit and n[i] < s.want and n[j] < t.want:
                assert abs((n[i] - s.least) - (n[j] - t.least)) <= 1


def test_gpu_stage_without_gpu_refused(monkeypatch):
    b = budget(10**10, None, None, monkeypatch)
    with pytest.raises(ResourceError):
        b.allocate([Stage("decoder", 1, 10**9, gpu=True)])


@given(st.lists(st.tuples(st.integers(0, 10**9), st.booleans()), max_size=8), st.integers(0, 10**10))
def test_reserve_release_round_trip(monkeypatch_session, items, memory):
    b = budget(memory, 10**10, None, monkeypatch_session)
    start = dict(b.left)
    taken = []
    for nbytes, shm in items:
        try:
            b.reserve("x", nbytes, shm=shm)
            taken.append((nbytes, shm))
        except ResourceError:
            pass
    for nbytes, shm in taken:
        b.release(nbytes, shm=shm)
    assert b.left == start


@given(st.integers(0, 10**10), st.integers(0, 10**10))
def test_gate_takes_the_rest(monkeypatch_session, memory, least):
    b = budget(memory, None, None, monkeypatch_session)
    if memory < least:
        with pytest.raises(ResourceError):
            b.gate("in flight", least)
        return
    g = b.gate("in flight", least)
    assert g.limit == max(1, int(memory)) and b.left["memory"] == 0


@pytest.fixture(scope="module")
def monkeypatch_session():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


# ---- ByteGate: a sequential model -------------------------------------------------------------------------------
class GateMachine(RuleBasedStateMachine):
    """try_acquire succeeds exactly when acquire would not block; used counts the holders; peak is the most used.
    Over the limit only by one item: the others in flight (if any) hold no bytes."""

    def __init__(self):
        super().__init__()
        self.gate, self.held, self.peak = None, [], 0

    @precondition(lambda self: self.gate is None)
    @rule(limit=st.integers(1, 100))
    def make(self, limit):
        self.gate = ByteGate("g", limit)

    @precondition(lambda self: self.gate is not None)
    @rule(n=st.integers(0, 150))
    def try_acquire(self, n):
        fits = sum(self.held) == 0 or sum(self.held) + n <= self.gate.limit
        assert self.gate.try_acquire(n) == fits
        if fits:
            self.held.append(n)
            self.peak = max(self.peak, sum(self.held))

    @precondition(lambda self: self.gate is not None and self.held)
    @rule(data=st.data())
    def release(self, data):
        n = self.held.pop(data.draw(st.integers(0, len(self.held) - 1)))
        self.gate.release(n)

    @invariant()
    def books(self):
        if self.gate is not None:
            assert self.gate.used == sum(self.held)
            assert self.gate.peak == self.peak
            assert self.gate.used <= self.gate.limit or sum(1 for n in self.held if n) == 1


test_gate_model = GateMachine.TestCase


# ---- ByteGate: threads ----------------------------------------------------------------------------------------
class Watched(ByteGate):
    """Checks the bound inside the gate's lock, at every acquire (holders of no bytes not counted)."""

    def __init__(self, name, limit):
        super().__init__(name, limit)
        self.holders, self.violations = 0, []

    def acquire(self, n):
        n = super().acquire(n)
        with self.cond:
            self.holders += n > 0
            if self.used > self.limit and self.holders > 1:
                self.violations.append((self.used, self.holders))
        return n

    def release(self, n):
        with self.cond:
            self.holders -= n > 0
        super().release(n)


@given(st.integers(1, 64), st.integers(2, 12), st.integers(0, 2**31))
def test_gate_threads(limit, threads, seed):
    """Producers acquire and release random sizes (some above the limit); none may wait forever (no lost wakeup)."""
    g = Watched("g", limit)
    rng = random.Random(seed)
    plans = [[rng.randint(0, 2 * limit) for _ in range(30)] for _ in range(threads)]

    def work(plan):
        for n in plan:
            g.acquire(n)
            g.release(n)

    ts = [threading.Thread(target=work, args=(p,), daemon=True) for p in plans]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=20)
    assert not any(t.is_alive() for t in ts), "a producer is still waiting: a release did not wake it"
    assert g.used == 0 and not g.violations, g.violations[:3]
