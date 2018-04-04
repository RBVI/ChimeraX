# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

import os
import sys
if sys.platform == "darwin":
    os.environ["CXX"] = "clang++"
    os.environ["CC"] = "clang++"
    extra_compile_args = ["-std=c++11", "-stdlib=libc++"]
    libraries = []
elif sys.platform == "linux":
    os.environ["CXX"] = "g++"
    os.environ["CC"] = "g++"
    extra_compile_args = ["-std=c++11"]
    libraries = ["atomstruct", "element"]
else:
    extra_compile_args = []
    libraries = ["libatomstruct", "libelement"]

build_dir = os.path.split(os.path.split(sys.executable)[0])[0]
include_path = os.path.join(build_dir, "include")
lib_path = os.path.join(build_dir, "lib")

ext = Extension("chimerax.core.atomic.cymol", ["cymol.pyx"],
    include_dirs=[include_path, numpy.get_include()],
    extra_compile_args=extra_compile_args,
    libraries=libraries,
    library_dirs=[lib_path]
)
setup(ext_modules = cythonize([ext], language="c++"))
