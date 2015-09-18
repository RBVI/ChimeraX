# vi: set expandtab shiftwidth=4 softtabstop=4:

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

def modify_selection(objects, mode = 'add'):
    select = (mode == 'add')
    for m, atoms in objects.atoms.by_structure:
        m.select_atoms(atoms, selected = select)
    from ..atomic import AtomicStructure
    for m in objects.models:
        if not isinstance(m, AtomicStructure):
            m.selected = select

def intersect_selection(objects, session):
    from .. import atomic
    subatoms = atomic.selected_atoms(session) - objects.atoms
    for m, atoms in subatoms.by_structure:
        m.select_atoms(atoms, selected = False)
    from ..atomic import AtomicStructure
    mkeep = set(objects.models)
    for m in session.selection.models():
        if m not in mkeep:
            m.selected = False

def register_command(session):
    from . import CmdDesc, register, ObjectsArg, NoArg
    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   keyword=[('add', ObjectsArg),
                            ('subtract', ObjectsArg),
                            ('intersect', ObjectsArg),
                            ('clear', NoArg),],
                   synopsis='select specified objects')
    register('select', desc, select)
