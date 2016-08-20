# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
chimerax.core: collection of base ChimeraX functionality
========================================================

"""
__copyright__ = (
    "Copyright \N{Copyright Sign} 2015-2016 by the Regents of the University of California."
    "  All Rights Reserved."
    "  Free for non-commercial use."
    "  See http://www.rbvi.ucsf.edu/chimerax/ for license details."
)

window_sys = None       # Set by startup script to "qt" or "wx".

_class_cache = {}
# list modules classes are found in used by session restore to recreate objects.
_class_class_init = {
    'Atom': '.atomic',
    'AtomicStructure': '.atomic',
    'AtomicStructures': '.atomic',
    'Atoms': '.atomic',
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
    'Map_Series': '.map.series',
    'Material': '.graphics',
    'Model': '.models',
    'Models': '.models',
    'MolecularSurface': '.atomic',
    'MonoCamera': '.graphics',
    'NamedView': '.commands.view',
    'NamedViews': '.commands.view',
    'Place': '.geometry',
    'Places': '.geometry',
    'Pseudobond': '.atomic',
    'PseudobondGroup': '.atomic.pbgroup',
    'PseudobondManager': '.atomic.molobject',
    'Pseudobonds': '.atomic',
    'Residue': '.atomic',
    'Residues': '.atomic',
    'Sequence': '.atomic',
    'Structure': '.atomic',
    'Tasks': '.tasks',
    'Tools': '.tools',
    'TriangleInfo': '.stl',
    'UserColors': '.colors',
    'UserColormaps': '.colors',
    'View': '.graphics',
    'Volume': '.map',
    '_Input': '.ui.nogui',
}

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
