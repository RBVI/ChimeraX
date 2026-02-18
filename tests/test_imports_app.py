import pkgutil
import importlib
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from import_excludes import IMPORT_EXCLUDES

import chimerax
import chimerax.core.__main__

from conftest import _ensure_chimerax_initialized  # noqa

blacklist = IMPORT_EXCLUDES

_ensure_chimerax_initialized()
modules = []

for info in pkgutil.walk_packages(
    chimerax.__path__, prefix=chimerax.__name__ + "."
):  # noqa
    module_finder, name, is_pkg = info
    if name not in blacklist:
        modules.append(name)

# Other modules that need to be tested since they're e.g. optional
# modules of dependencies, like numpy_formathandler
modules.extend(["OpenGL_accelerate.numpy_formathandler"])


@pytest.mark.parametrize("module", modules)
def test_imports(module):
    importlib.import_module(module)
