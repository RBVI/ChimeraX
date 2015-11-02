# vim: set expandtab shiftwidth=4 softtabstop=4:

def display(session, objects=None, what=None, only=False):
    '''Display specified atoms, bonds or models.

    Parameters
    ----------
    objects : AtomSpecResults or None
        Atoms, bonds or models to show.  If None then all are shown.
        Objects that are already shown remain shown.
    what : 'atoms', 'bonds', 'pseudobonds', 'pbonds', 'cartoons', 'ribbons', 'models' or None
        What to show.  If None then 'atoms' if any atoms specified otherwise 'models'.
    only : bool
        Show only the specified atoms/bonds/residues in each specified molecule.
        If what is models then only show then hide models that are not specified.
    '''
    if objects is None:
        from . import atomspec
        objects = atomspec.all_objects(session)

    if what is None:
        what = 'atoms' if objects.atoms else 'models'

    if what == 'atoms':
        atoms = objects.atoms
        atoms.displays = True
        if only:
            from ..atomic import structure_atoms
            other_atoms = structure_atoms(atoms.unique_structures) - atoms
            other_atoms.displays = False
    elif what == 'bonds':
        bonds = objects.atoms.inter_bonds
        bonds.displays = True
        a1, a2 = bonds.atoms
        a1.displays = True	   # Atoms need to be displayed for bond to appear
        a2.displays = True
        if only:
            mbonds = [m.bonds for m in atoms.unique_structures]
            if mbonds:
                from ..atomic import concatenate
                all_bonds = concatenate(mbonds)
                other_bonds = all_bonds - bonds
                other_bonds.displays = False
    elif what in ('pseudobonds', 'pbonds'):
        atoms = objects.atoms
        from .. import atomic
        pbonds = atomic.interatom_pseudobonds(atoms, session)
        pbonds.displays = True
        a1, a2 = pbonds.atoms
        a1.displays = True	   # Atoms need to be displayed for bond to appear
        a2.displays = True
        if only:
            pbs = sum([[pbg.pseudobonds for pbg in m.pseudobond_groups.values()]
                       for m in atoms.unique_structures], [])
            if pbs:
                from ..atomic import concatenate
                all_pbonds = concatenate(pbs)
                other_pbonds = all_pbonds - pbonds
                other_pbonds.displays = False
    elif what == 'cartoons' or what == 'ribbons':
        atoms = objects.atoms
        res = atoms.unique_residues
        res.ribbon_displays = True
        if only:
            from ..atomic import structure_residues
            other_res = structure_residues(atoms.unique_structures) - res
            other_res.ribbon_displays = False
    elif what == 'models':
        models = objects.models
        for m in models:
            m.display = True
        if only:
            mset = set(models)
            for m in session.models.list():
                if m not in mset:
                    m.display = False

def undisplay(session, objects=None, what=None):
    '''Hide specified atoms, bonds or models.

    Parameters
    ----------
    objects : AtomSpecResults or None
        Atoms, bonds or models to hide. If None then all are hidden.
    what : 'atoms', 'bonds', 'pseudobonds', 'pbonds', 'cartoons', 'ribbons', 'models' or None
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
    elif what in ('pseudobonds', 'pbonds'):
        from .. import atomic
        pbonds = atomic.interatom_pseudobonds(objects.atoms, session)
        pbonds.displays = False
    elif what == 'cartoons' or what == 'ribbons':
        res = objects.atoms.unique_residues
        res.ribbon_displays = False
    elif what == 'models':
        for m in objects.models:
            m.display = False

def register_command(session):
    from . import CmdDesc, register, ObjectsArg, EnumOf, EmptyArg, Or, NoArg
    what_arg = EnumOf(('atoms', 'bonds', 'pseudobonds', 'pbonds',
                       'cartoons', 'ribbons', 'models'))
    desc = CmdDesc(optional=[('objects', Or(ObjectsArg, EmptyArg)),
                             ('what', what_arg),
                             ('only', NoArg)],
                   synopsis='display specified objects')
    register('display', desc, display)
    desc = CmdDesc(optional=[('objects', Or(ObjectsArg, EmptyArg)),
                             ('what', what_arg)],
                   synopsis='undisplay specified objects')
    register('~display', desc, undisplay)
