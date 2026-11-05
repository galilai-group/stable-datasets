import os

from hypothesis import settings


# FUZZ_EXAMPLES=20000 pytest ...: a longer fuzzing run than the default
settings.register_profile("framestore-fuzz", max_examples=int(os.environ.get("FUZZ_EXAMPLES", 5000)), deadline=None)
settings.register_profile("framestore", max_examples=200, deadline=None)
settings.load_profile("framestore-fuzz" if os.environ.get("FUZZ_EXAMPLES") else "framestore")
