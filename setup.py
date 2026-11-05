"""Builds framestore's C helpers (stable_datasets/framestore/csrc) into the wheel; everything else is in pyproject.toml.

They are plain shared libraries loaded with ctypes (framestore/_native.py), not Python extension modules, so a wheel
does not depend on the Python version: one per platform, tagged py3-none-<platform>. Each is stamped with the hash of
its source, and _native.load uses it only if that still matches (an editable install whose C changed compiles the
new source instead). Both are optional: where they cannot be built (no compiler, no libjpeg headers), installing
still works, and framestore compiles them on first use or runs its slower fallbacks. FRAMESTORE_REQUIRE_HELPERS=1
(the release wheels) makes a failed build an error.

libjpeg-turbo: FRAMESTORE_LIBJPEG=<prefix> links <prefix>/lib/libjpeg.a in statically (the release wheels, see
.github/scripts/build_libjpeg.sh); otherwise the system's libjpeg is linked dynamically.
"""

import hashlib
import os

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext


CSRC = os.path.join("stable_datasets", "framestore", "csrc")


def helper(name, jpeg=False):
    src = os.path.join(CSRC, f"{name}.c")
    with open(src, "rb") as f:
        key = hashlib.sha1(f.read()).hexdigest()[:12]  # _native.source_key
    kw = {}
    if jpeg:
        prefix = os.environ.get("FRAMESTORE_LIBJPEG")
        if prefix:
            kw = {"include_dirs": [f"{prefix}/include"], "extra_objects": [f"{prefix}/lib/libjpeg.a"]}
        else:
            dirs = [p for p in ("/opt/homebrew", "/usr/local") if os.path.exists(f"{p}/include/jpeglib.h")]
            kw = {
                "libraries": ["jpeg"],
                "include_dirs": [f"{p}/include" for p in dirs],
                "library_dirs": [f"{p}/lib" for p in dirs],
            }
    return Extension(
        f"stable_datasets.framestore._{name}",
        [src],
        define_macros=[("FRAMESTORE_KEY", f'"{key}"')],
        extra_compile_args=["-O2"],
        optional=not os.environ.get("FRAMESTORE_REQUIRE_HELPERS"),  # release wheels must have them
        **kw,
    )


class BuildHelpers(build_ext):
    def get_ext_filename(self, fullname):  # _<name>.so: loaded with ctypes, not imported
        return os.path.join(*fullname.split(".")) + ".so"

    def get_export_symbols(self, ext):  # no PyInit_<name>: not a Python module
        return []


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):  # any Python 3, this platform
        return "py3", "none", super().get_tag()[2]


setup(
    ext_modules=[] if os.name == "nt" else [helper("pread_many"), helper("jpeg_restart", jpeg=True)],
    cmdclass={"build_ext": BuildHelpers, "bdist_wheel": PlatformWheel},
)
