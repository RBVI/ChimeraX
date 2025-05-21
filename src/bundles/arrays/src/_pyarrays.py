# vim: set expandtab ts=4 sw=4:

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
import os
import sys
import warnings

def path_to_src() -> str:
    return os.path.dirname(__file__)

def get_lib() -> str:
    return os.path.join(path_to_src(), 'lib')

def get_include() -> str:
    return os.path.join(path_to_src(), 'include')

if sys.platform.startswith('win'):
    os.add_dll_directory(get_lib())

from . import _arrays

def load_libarrays():
    warnings.warn(
        "load_libarrays is no longer required to link libarrays."
        " Please instead import chimerax.arrays"
        , DeprecationWarning
        , stacklevel=2
    )

from chimerax.core.toolshed import BundleAPI

class _ArraysAPI(BundleAPI):
    pass

bundle_api = _ArraysAPI()
