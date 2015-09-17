# vi: set expandtab shiftwidth=4 softtabstop=4:


def select(session, objects=None, clear=False, subtract=None):
    '''Select specified objects.

    Parameters
    ----------
    objects : AtomSpecResults
      Select the specified objects (typically atoms). If no specifier is given then everything is selected.
      Objects that are already selected remain selected.
    clear : no value
      Clear the selection.
    subtract : AtomSpecResults
      Unselect the specified objects.
    '''
    if clear:
        session.selection.clear()
        if objects is None:
            return

    if objects is None and subtract is None:
        from . import atomspec
        objects = atomspec.everything(session).evaluate(session)

    if objects is not None:
        for m, atoms in objects.atoms.by_structure:
            m.select_atoms(atoms)
        from ..atomic import AtomicStructure
        for m in objects.models:
            if not isinstance(m, AtomicStructure):
                m.selected = True

    if subtract is not None:
        for m, atoms in subtract.atoms.by_structure:
            m.select_atoms(atoms, selected = False)
        from ..atomic import AtomicStructure
        for m in subtract.models:
            if not isinstance(m, AtomicStructure):
                m.selected = False
        

def register_command(session):
    from . import CmdDesc, register, ObjectsArg, NoArg
    desc = CmdDesc(optional=[('objects', ObjectsArg)],
                   keyword=[('clear', NoArg),
                            ('subtract', ObjectsArg)],
                   synopsis='select specified objects')
    register('select', desc, select)
