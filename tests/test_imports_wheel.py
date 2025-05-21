import importlib
import os
import sys

import pkgutil


import pytest

import chimerax


sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "build", "wheel"
    )
)

# in utils/wheel/filter_modules.py
from filter_modules import module_blacklist, fine_blacklist


modules = []


def check_if_true_error(pkg):
    if pkg not in fine_blacklist and pkg not in module_blacklist:
        raise


for info in pkgutil.walk_packages(
    chimerax.__path__, prefix=chimerax.__name__ + ".", onerror=check_if_true_error
):
    module_finder, name, is_pkg = info
    if name.endswith(".tool") or (name in fine_blacklist) or (name in module_blacklist):
        continue
    modules.append(name)


@pytest.mark.parametrize("module", modules)
@pytest.mark.wheel
def test_imports(module):
    importlib.import_module(module)
