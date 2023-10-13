# vim: set expandtab shiftwidth=4 softtabstop=4:

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

"""
chimerax.core: collection of base ChimeraX functionality
========================================================

"""
from .buildinfo import version
__version__ = version

import os
def path_to_src() -> str:
    return os.path.dirname(__file__)

def get_lib() -> str:
    return os.path.join(path_to_src(), 'lib')

def get_include() -> str:
    return os.path.join(path_to_src(), 'include')

from .toolshed import BundleAPI
BUNDLE_NAME = 'ChimeraX-Core'

_class_cache = {}
# list modules classes are found in used by session restore to recreate objects.
_class_class_init = {
    'UserAliases': '.session',
    'CameraClipPlane': 'chimerax.graphics',
    'ClipPlane': 'chimerax.graphics',
    'Color': '.colors',
    'Colormap': '.colors',
    'Drawing': 'chimerax.graphics',
    'Generic3DModel': '.generic3d',
    'Job': '.tasks',
    'Lighting': 'chimerax.graphics',
    'Material': 'chimerax.graphics',
    'Model': '.models',
    'Models': '.models',
    'MonoCamera': 'chimerax.graphics',
    'OrthographicCamera': 'chimerax.graphics',
    'Place': 'chimerax.geometry',
    'Places': 'chimerax.geometry',
    'SceneClipPlane': 'chimerax.graphics',
    'Surface': '.models',
    'Tasks': '.tasks',
    'Tools': '.tools',
    'Undo': '.undo',
    'UserColors': '.colors',
    'UserColormaps': '.colors',
    'View': 'chimerax.graphics',
    '_Input': '.nogui',
    # atomic classes moved to a separate bundle; listed here
    # for backwards compatibility
    # (perhaps remove at 1.0? No. Removing breaks loading old sessions.)
    "Atom": 'chimerax.atomic',
    "AtomicStructure": 'chimerax.atomic',
    "AtomicStructures": 'chimerax.atomic',
    "Atoms": 'chimerax.atomic',
    "Bond": 'chimerax.atomic',
    "Bonds": 'chimerax.atomic',
    "Chain": 'chimerax.atomic',
    "Chains": 'chimerax.atomic',
    "CoordSet": 'chimerax.atomic',
    "LevelOfDetail": 'chimerax.atomic',
    "MolecularSurface": 'chimerax.atomic',
    "PseudobondGroup": 'chimerax.atomic',
    "PseudobondManager": 'chimerax.atomic',
    "Pseudobond": 'chimerax.atomic',
    "Pseudobonds": 'chimerax.atomic',
    "Residue": 'chimerax.atomic',
    "Residues": 'chimerax.atomic',
    "SeqMatchMap": 'chimerax.atomic',
    "Sequence": 'chimerax.atomic',
    "Structure": 'chimerax.atomic',
    "StructureSeq": 'chimerax.atomic',
    "AttrRegistration": '.attributes',
    "_NoDefault": '.attributes',
    "RegAttrManager": '.attributes',
    "XSectionManager": 'chimerax.atomic.ribbon',
}


class _MyAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        """Return chimerax.core class for instance in a session

        Parameters
        ----------
        class_name : str
            Class name
        """
        try:
            return _class_cache[class_name]
        except KeyError:
            pass
        module_name = _class_class_init.get(class_name, None)
        if module_name is None:
            cls = None
        else:
            import importlib
            mod = importlib.import_module(module_name, __package__)
            cls = getattr(mod, class_name)
        _class_cache[class_name] = cls
        return cls


bundle_api = _MyAPI()


def profile(func):
    def wrapper(*args, **kw):
        import sys
        import cProfile
        import pstats
        prof = cProfile.Profile()
        v = prof.runcall(func, *args, **kw)
        print(func.__name__, file=sys.__stderr__)
        p = pstats.Stats(prof, stream=sys.__stderr__)
        p.strip_dirs().sort_stats("cumulative", "time").print_callers(40)
        return v
    return wrapper


def is_daily_build():
    """Supported API. Return if ChimeraX Core is from a daily build."""
    # Daily builds are development releases
    from packaging.version import Version
    ver = Version(version)
    return ver.is_devrelease
