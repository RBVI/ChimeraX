# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
"""Load the atomic Python classes in their required dependency order.

Importing this module uses Python's module lock. Publish the exports only after
all classes and residue templates are ready, including for concurrent importers.
"""
from importlib import import_module
from . import _data_model_exports

_package = import_module(__package__)
_exports = {}
for _module_name, _names in _data_model_exports.items():
    _module = import_module('.' + _module_name, __package__)
    for _name in _names:
        _exports[_name] = getattr(_module, _name)

# Configure templates before a reader or a user can construct molecular objects.
from chimerax import atomic_lib, pdb_lib
from .molobject import Residue
import os
Residue.set_templates_dir(os.path.join(atomic_lib.__path__[0], 'data'))
import chimerax
if hasattr(chimerax, 'app_dirs'):
    Residue.set_user_templates_dir(chimerax.app_dirs.user_data_dir)

vars(_package).update(_exports)
