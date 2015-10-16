# vi: set expandtab shiftwidth=4 softtabstop=4:

def display(session, objects=None, what=None):
    '''Display specified atoms, bonds or models.

    Parameters
    ----------
    objects : AtomSpecResults or None
        Atoms, bonds or models to show.  If None then all are shown.
        Objects that are already shown remain shown.
    what : 'atoms', 'bonds', or 'models' or None
        What to show.  If None then 'atoms' if any atoms specified otherwise 'models'.
    '''
    if objects is None:
        from . import atomspec
        objects = atomspec.all_objects(session)

    if what is None:
        what = 'atoms' if objects.atoms else 'models'

    if what == 'atoms':
        objects.atoms.displays = True
    elif what == 'bonds':
        atoms = objects.atoms
        atoms.displays = True
        atoms.inter_bonds.displays = True
        from .. import atomic
        atomic.interatom_pseudobonds(atoms, session).displays = True
    elif what == 'models':
        for m in objects.models:
            m.display = True

def undisplay(session, objects=None, what=None):
    '''Hide specified atoms, bonds or models.

    Parameters
    ----------
    objects : AtomSpecResults or None
        Atoms, bonds or models to hide. If None then all are hidden.
    what : 'atoms', 'bonds', or 'models'
        What to hide.  If None then 'atoms' if any atoms specified otherwise 'models'.
    '''
    if objects is None:
        from . import atomspec
        objects = atomspec.all_objects(session)

    if what is None:
        what = 'atoms' if objects.atoms else 'models'

    if what == 'atoms':
        objects.atoms.displays = False
    elif what == 'bonds':
        atoms = objects.atoms
        atoms.inter_bonds.displays = False
        from .. import atomic
        atomic.interatom_pseudobonds(atoms, session).displays = False
    elif what == 'models':
        for m in objects.models:
            m.display = False

def register_command(session):
    from . import CmdDesc, register, ObjectsArg, EnumOf, EmptyArg, Or
    what_arg = EnumOf(('atoms', 'bonds', 'models'))
    desc = CmdDesc(optional=[('objects', Or(ObjectsArg, EmptyArg)),
                             ('what', what_arg)],
                   synopsis='display specified objects')
    register('display', desc, display)
    desc = CmdDesc(optional=[('objects', Or(ObjectsArg, EmptyArg)),
                             ('what', what_arg)],
                   synopsis='undisplay specified objects')
    register('~display', desc, undisplay)
