"""Build Cython extensions for the multi-model pipeline.

Extensions
----------
src._feature_fast
    Accelerated inner loops for handcrafted feature extraction
    (AA frequencies, dipeptide transitions, group transitions).
    Used by src/feature_pipeline.py; falls back to pure Python if absent.

src._esm2_bench_fast
    Accelerated inner loops for the ESM2 sliding-window benchmark
    (sliding window mean/max pooling over embedding matrices,
    per-residue score accumulation).
    Used by src/benchmark_esm2.py; falls back to numpy if absent.

Usage
-----
    pip install cython numpy
    python setup_cython.py build_ext --inplace

Both extensions release the GIL (``with nogil:``) and are safe to call
from multiple Python threads concurrently.
"""

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:
    raise SystemExit("Cython is required to build this extension.\n"
                     "Install it with:  pip install cython")

import numpy as np

_compile_args = ["-O3", "-ffast-math"]
_include_dirs = [np.get_include()]
_directives = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "nonecheck": False,
}

extensions = [
    Extension(
        name="src._feature_fast",
        sources=["src/_feature_fast.pyx"],
        include_dirs=_include_dirs,
        extra_compile_args=_compile_args,
    ),
    Extension(
        name="src._esm2_bench_fast",
        sources=["src/_esm2_bench_fast.pyx"],
        include_dirs=_include_dirs,
        extra_compile_args=_compile_args,
    ),
]

setup(
    name="multi_model_pipeline",
    ext_modules=cythonize(
        extensions,
        compiler_directives=_directives,
        annotate=False,
    ),
)
