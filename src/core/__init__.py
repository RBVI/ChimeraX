# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
chimera.core: collection of base chimera functionality
======================================================

"""
__copyright__ = (
    "Copyright \u00A9 2015 by the Regents of the University of California."
    "  All Rights Reserved."
    "  Free for non-commercial use."
    "  See http://www.cgl.ucsf.edu/chimera/ for license details."
)
_class_cache = {}
# list modules classes are found in
_class_class_init = {
    'AtomicStructure': '.atomic',
    'Generic3DModel': '.generic3d',
    'Model': '.models',
    'Models': '.models',
    'MolecularSurface': '.molsurf',
    'STLModel': '.stl',
    'Job': '.tasks',
    'Tasks': '.tasks',
    'Tools': '.tools',
    'TriangleInfo': '.stl',
    'UserColors': '.colors',
    'UserColormaps': '.colors',
    'ViewState': '.graphics.gsession',
    '_Input': '.ui.nogui',
}


def get_class(class_name):
    """Return chimera.core class for instance in a session

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
