from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy as np

setup(
    name='cyutils app',
    ext_modules=cythonize("holodeck/cyutils.pyx", annotate=True, compiler_directives={'language_level' : "3"}),
    include_dirs=[np.get_include()],
)
