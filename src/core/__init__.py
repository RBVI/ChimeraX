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

"""
chimerax.core: collection of base ChimeraX functionality
========================================================

"""
BUNDLE_NAME = 'ChimeraX-Core'
from .buildinfo import version
from .toolshed import BundleAPI

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
        import cProfile, pstats, sys
        prof = cProfile.Profile()
        v = prof.runcall(func, *args, **kw)
        print(func.__name__, file=sys.__stderr__)
        p = pstats.Stats(prof, stream=sys.__stderr__)
        p.strip_dirs().sort_stats("cumulative", "time").print_callers(40)
        return v
    return wrapper


# These are entry points for copying files into
# .dist-info directories of wheels when they are built

def copy_distinfo_file(cmd, basename, filename, binary=''):
    """Entry point to copy files into bundle .dist-info directory.

    File is copied as text if binary is '', and as binary if 'b'.
    """
    try:
        with open(basename, 'r' + binary) as fi:
            value = fi.read()
            from distutils import log
            log.info("copying %s", basename)
            if not cmd.dry_run:
                with open(filename, 'w' + binary) as fo:
                    fo.write(value)
    except IOError:
        # Missing file is okay
        pass


def copy_distinfo_binary_file(cmd, basename, filename):
    copy_distinfo_file(cmd, basename, filename, binary='b')


def is_daily_build():
    """Supported API. Return if ChimeraX Core is from a daily build."""
    # Daily builds are development releases
    from packaging.version import Version
    ver = Version(version)
    return ver.is_devrelease
