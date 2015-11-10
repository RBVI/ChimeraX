# vim: set expandtab shiftwidth=4 softtabstop=4:

def select(session, objects=None, add=None, subtract=None, intersect=None, clear=False):
    '''Select specified objects.

    Parameters
    ----------
    objects : AtomSpecResults
      Select the specified objects (typically atoms). If no specifier is given then everything is selected.
      The current selection is replaced.
    add : AtomSpecResults
      Modify the current selection by adding the specified objects.
    subtract : AtomSpecResults
      Modify the current selection by unselecting the specified objects.
    intersect : AtomSpecResults
      Modify the current selection keeping only those among the specified objects selected.
    clear : no value
      Clear the selection.
    '''
    if clear:
        session.selection.clear()
        if objects is None:
            return

    if objects is None and add is None and subtract is None and intersect is None:
        from . import atomspec
        objects = atomspec.everything(session).evaluate(session)

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
    subatoms = atomic.selected_atoms(session) - atoms
    submodels = set(session.selection.models()).difference(models)
    for m, matoms in subatoms.by_structure:
        m.select_atoms(matoms, selected = False)
    for m in submodels:
        m.selected = False

def _atoms_and_models(objects):
    # Treat selecting molecular surface as selecting atoms.
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
    from . import CmdDesc, register, ObjectsArg, NoArg
    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   keyword=[('add', ObjectsArg),
                            ('subtract', ObjectsArg),
                            ('intersect', ObjectsArg),
                            ('clear', NoArg),],
                   synopsis='select specified objects')
    register('select', desc, select)
