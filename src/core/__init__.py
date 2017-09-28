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
    'Atom': '.atomic',
    'AtomicStructure': '.atomic',
    'AtomicStructures': '.atomic',
    'Atoms': '.atomic',
    'AttrRegistration': '.atomic.attr_registration',
    'Bond': '.atomic',
    'Bonds': '.atomic',
    'Chain': '.atomic',
    'Chains': '.atomic',
    'ClipPlane': '.graphics',
    'Color': '.colors',
    'Drawing': '.graphics',
    'Generic3DModel': '.generic3d',
    'GridDataState': '.map.session',
    'Job': '.tasks',
    'LevelOfDetail': '.atomic.structure',
    'Lighting': '.graphics',
    'MapChannelsModel': '.map',
    'Map_Series': '.map.series',
    'Material': '.graphics',
    'Model': '.models',
    'Models': '.models',
    'MolecularSurface': '.atomic',
    'MonoCamera': '.graphics',
    'MultiChannelSeries': '.map',
    'NamedView': '.commands.view',
    'NamedViews': '.commands.view',
    'OrthographicCamera': '.graphics',
    'Place': '.geometry',
    'Places': '.geometry',
    'Pseudobond': '.atomic',
    'PseudobondGroup': '.atomic.pbgroup',
    'PseudobondManager': '.atomic.molobject',
    'Pseudobonds': '.atomic',
    'RegAttrManager': '.atomic.attr_registration',
    'Residue': '.atomic',
    'Residues': '.atomic',
    'SeqMatchMap': '.atomic',
    'Sequence': '.atomic',
    'Structure': '.atomic',
    'StructureSeq': '.atomic',
    'Tasks': '.tasks',
    'Tools': '.tools',
    'TriangleInfo': '.stl',
    'Undo': '.undo',
    'UserColors': '.colors',
    'UserColormaps': '.colors',
    'View': '.graphics',
    'Volume': '.map',
    'XSectionManager': '.atomic.ribbon',
    '_Input': '.ui.nogui',
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
