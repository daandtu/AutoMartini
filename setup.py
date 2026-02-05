"""Cython build hook for setuptools. All config is in pyproject.toml."""

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [Extension("auto_martiniM3.optimization", ["auto_martiniM3/optimization.pyx"])],
        language_level="3",
    ),
    include_dirs=[numpy.get_include()],
)
