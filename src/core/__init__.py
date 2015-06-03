"""
chimera.core: collection of base chimera functionality
======================================================

"""
_class_cache = {}


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
    if class_name == 'AtomicStructure':
        from . import structure
        cls = structure.AtomicStructure
    elif class_name == 'Generic3DModel':
        from . import generic3d
        cls = generic3d.Generic3DModel
    elif class_name == 'MolecularSurface':
        from . import molsurf
        cls = molsurf.MolecularSurface
    elif class_name == 'STLModel':
        from . import stl
        cls = stl.STLModel
    elif class_name == 'Job':
        from . import tasks
        cls = tasks.Job
    elif class_name == '_Input':
        from . import nogui
        cls = nogui._Input
    else:
        return None
    _class_cache[class_name] = cls
    return cls
