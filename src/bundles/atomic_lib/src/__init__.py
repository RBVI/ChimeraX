# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
import os

def path_to_src() -> str:
    return os.path.dirname(__file__)

def get_lib() -> str:
    return os.path.join(path_to_src(), 'lib')

def get_include() -> str:
    return os.path.join(path_to_src(), 'include')

from chimerax.core.toolshed import BundleAPI

class _AtomicLibAPI(BundleAPI):
    pass

bundle_api = _AtomicLibAPI()

# Load libarrays since atomic_libs C++ shared libraries use it.
import chimerax.arrays

# Include atomic_libs/lib in runtime library search path.
import sys
if sys.platform.startswith('win'):
    os.add_dll_directory(get_lib())

# Load atomic_libs libraries so they are found by other C++ modules that link to them.
from . import _load_libs
