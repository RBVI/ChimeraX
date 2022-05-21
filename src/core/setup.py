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

from setuptools import setup, Extension
from Cython.Build import cythonize 

from glob import glob
import sys

# For some reason we have to cythonize this before we can compile it
cythonize(Extension("chimerax.core._serialize", sources=["src/_serialize.pyx"]))

ext_mods = [Extension("chimerax.core._serialize", sources=["src/_serialize.cpp"])]

if sys.platform == "darwin":
    ext_mods.append(
        Extension(
            "chimerax.core._mac_util"
            , sources=glob("src/mac_util_cpp/*[!.h]")
        )
    )

setup(
    ext_modules = ext_mods
)
