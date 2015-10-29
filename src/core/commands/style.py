# vim: set expandtab shiftwidth=4 softtabstop=4:


def style(session, atoms, atom_style):
    '''Set the atom display style.

    Parameters
    ----------
    atoms : Atoms or None
        Change the style of these atoms. If not specified then all atoms are changed.
    atom_style : "sphere", "ball" or "stick"
        Controls how atoms and bonds are depicted.
    '''
    from ..atomic import Atom
    s = {
        'sphere': Atom.SPHERE_STYLE,
        'ball': Atom.BALL_STYLE,
        'stick': Atom.STICK_STYLE,
    }[atom_style.lower()]
    if atoms is None:
        from ..atomic import all_atoms
        atoms = all_atoms(session)
    atoms.draw_modes = s


def register_command(session):
    from . import register, CmdDesc, AtomsArg, EmptyArg, EnumOf, Or
    desc = CmdDesc(required=[("atoms", Or(AtomsArg, EmptyArg)),
                             ('atom_style', EnumOf(('sphere', 'ball', 'stick')))],
                   synopsis='change atom depiction')
    register('style', desc, style)
