# vi: set expandtab shiftwidth=4 softtabstop=4:

def style(session, atom_style, atoms=None):
    '''Set the atom display style.

    Parameters
    ----------
    atom_style : "sphere", "ball" or "stick"
        Controls how atoms and bonds are depicted.
    atoms : atom specifier
        Change the style of these atoms. If not specified then all atoms are changed.
    '''
    from ..atomic import AtomicStructure
    s = {
        'sphere': AtomicStructure.SPHERE_STYLE,
        'ball': AtomicStructure.BALL_STYLE,
        'stick': AtomicStructure.STICK_STYLE,
    }[atom_style.lower()]
    if atoms is None:
        for m in session.models.list():
            if isinstance(m, AtomicStructure):
                m.atoms.draw_modes = s
    else:
        asr = atoms.evaluate(session)
        asr.atoms.draw_modes = s

def register_command(session):
    from . import cli, atomspec
    desc = cli.CmdDesc(required=[('atom_style', cli.EnumOf(('sphere', 'ball', 'stick')))],
                       optional=[("atoms", atomspec.AtomSpecArg)],
                       synopsis='change atom depiction')
    cli.register('style', desc, style)
