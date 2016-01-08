# vim: set expandtab shiftwidth=4 softtabstop=4:

def select(session, objects=None, add=None, subtract=None, intersect=None, clear=False):
    '''Select specified objects.

    Parameters
    ----------
    objects : Objects
      Select the specified objects (typically atoms). If no specifier is given then everything is selected.
      The current selection is replaced.
    add : Objects
      Modify the current selection by adding the specified objects.
    subtract : Objects
      Modify the current selection by unselecting the specified objects.
    intersect : Objects
      Modify the current selection keeping only those among the specified objects selected.
    clear : no value
      Clear the selection.
    '''
    if clear:
        session.selection.clear()
        if objects is None:
            return

    if objects is None and add is None and subtract is None and intersect is None:
        from . import all_objects
        objects = all_objects(session)

    if objects is not None:
        session.selection.clear()
        modify_selection(objects, 'add')

    if add is not None:
        modify_selection(add, 'add')

    if subtract is not None:
        modify_selection(subtract, 'subtract')

    if intersect is not None:
        intersect_selection(intersect, session)

    report_selection(session)

def select_add(session, objects=None):
    '''Add objects to the selection.
    If objects is None everything is selected.'''
    if objects is None:
        from . import all_objects
        objects = all_objects(session)
    modify_selection(objects, 'add')

def select_subtract(session, objects=None):
    '''Subtract objects from the selection.
    If objects is None the selection is cleared.'''
    if objects is None:
        session.selection.clear()
    else:
        modify_selection(objects, 'subtract')

def select_intersect(session, objects=None):
    '''Reduce the selection by intersecting with specified objects.'''
    intersect_selection(objects, session)

def select_clear(session, objects=None):
    '''Clear the selection.'''
    session.selection.clear()

def report_selection(session):
    s = session.selection
    from ..atomic import MolecularSurface, AtomicStructure    
    mc = len([m for m in s.models() if not isinstance(m, (AtomicStructure, MolecularSurface))])
    ac = sum([len(atoms) for atoms in s.items('atoms')], 0)
    lines = []
    if mc == 0 and ac == 0:
        lines.append('Nothing')
    if ac != 0:
        plural = ('s' if ac > 1 else '')
        lines.append('%d atom%s' % (ac, plural))
    if mc != 0:
        plural = ('s' if mc > 1 else '')
        lines.append('%d model%s' % (mc, plural))
    session.logger.status(', '.join(lines) + ' selected')

def modify_selection(objects, mode = 'add'):
    select = (mode == 'add')
    atoms, models = _atoms_and_models(objects)
    for m, matoms in atoms.by_structure:
        m.select_atoms(matoms, selected = select)
    for m in models:
        m.selected = select

def intersect_selection(objects, session):
    atoms, models = _atoms_and_models(objects)
    from .. import atomic
    selatoms = atomic.selected_atoms(session)
    subatoms = selatoms - atoms
    from ..atomic import AtomicStructure
    selmodels = set(m for m in session.selection.models() if not isinstance(m, AtomicStructure))
    submodels = selmodels.difference(models)
    for m, matoms in subatoms.by_structure:
        m.select_atoms(matoms, selected = False)
    for m in submodels:
        m.selected = False

def _atoms_and_models(objects):
    # Treat selecting molecular surface as selecting atoms.
    # Returned models list does not include atomic models
    atoms = objects.atoms
    satoms = []
    models = []
    from ..atomic import MolecularSurface, AtomicStructure
    for m in objects.models:
        if isinstance(m, MolecularSurface):
            satoms.append(m.atoms)
        elif not isinstance(m, AtomicStructure):
            models.append(m)
    if satoms:
        from ..atomic import molarray
        atoms = molarray.concatenate([atoms] + satoms)
    return atoms, models

def register_command(session):
    from . import CmdDesc, register, ObjectsArg, NoArg, create_alias
    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   keyword=[('add', ObjectsArg),
                            ('subtract', ObjectsArg),
                            ('intersect', ObjectsArg),
                            ('clear', NoArg),],
                   synopsis='select specified objects')
    register('select', desc, select)

    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   synopsis='add objects to selection')
    register('select add', desc, select_add)

    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   synopsis='subtract objects from selection')
    register('select subtract', desc, select_subtract)

    desc = CmdDesc(required=[('objects', ObjectsArg)],
                   synopsis='intersect objects with selection')
    register('select intersect', desc, select_intersect)

    desc = CmdDesc(synopsis='clear the selection')
    register('select clear', desc, select_clear)

    create_alias('~select', 'select subtract $*')
