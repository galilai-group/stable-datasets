"""Frames for fuzz_memory.sh: Hypothesis's generators from test_restart_fuzz.py (valid frames, mutated frames, arbitrary
bytes), drawn without its test runner.   python make_corpus.py DIR N"""

import os
import sys
import warnings


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hypothesis import strategies as st  # noqa: E402
from hypothesis.errors import NonInteractiveExampleWarning  # noqa: E402
from test_restart_fuzz import jpegs, mutated  # noqa: E402


warnings.simplefilter("ignore", NonInteractiveExampleWarning)
out, n = sys.argv[1], int(sys.argv[2])
os.makedirs(out, exist_ok=True)
gen = st.one_of(jpegs(), mutated(), mutated(), st.binary(max_size=2048))
for i in range(n):
    open(os.path.join(out, f"{i:06d}.jpg"), "wb").write(gen.example())
print(f"{n} frames in {out}")
