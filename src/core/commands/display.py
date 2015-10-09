# vi: set expandtab shiftwidth=4 softtabstop=4:


def display(session, atoms=None, bonds=None):
    '''Display specified atoms.

    Parameters
    ----------
    atoms : Atoms
        Show the specified atoms. If atoms is None then all atoms are shown.
        Atoms that are already shown remain shown.
    bonds : bool or None
        If not None, show or hide bonds between specified atoms.
    '''
    if atoms is None:
        from .. import atomic
        atoms = atomic.all_atoms(session)

    atoms.displays = True

    if bonds is not None:
        atoms.inter_bonds.displays = bonds
        from .. import atomic
        atomic.interatom_pseudobonds(atoms, session).displays = bonds

def undisplay(session, atoms=None, bonds=None):
    '''Hide specified atoms.

    Parameters
    ----------
    atoms : atom specifier
        Hide the specified atoms. If no atom specifier is given then all atoms are hidden.
        If bonds option is specified than atoms are not hidden, only bonds are hidden.
    bonds : bool or None
        If not None, hide bonds between specified atoms.
    '''
    if atoms is None:
        from .. import atomic
        atoms = atomic.all_atoms(session)

    if bonds is None:
        atoms.displays = False

    if bonds is not None:
        atoms.inter_bonds.displays = False
        from .. import atomic
        atomic.interatom_pseudobonds(atoms, session).displays = False


def register_command(session):
    from . import CmdDesc, register, AtomsArg, NoArg
    desc = CmdDesc(optional=[("atoms", AtomsArg)],
                   keyword=[('bonds', NoArg)],
                   synopsis='display specified atoms')
    register('display', desc, display)
    desc = CmdDesc(optional=[("atoms", AtomsArg)],
                   keyword=[('bonds', NoArg)],
                   synopsis='undisplay specified atoms')
    register('~display', desc, undisplay)
