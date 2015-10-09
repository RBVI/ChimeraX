# vi: set expandtab shiftwidth=4 softtabstop=4:


def transparency(session, atoms, percent, target='s'):
    """Set transparency of atoms, ribbons, surfaces, ....

    Parameters
    ----------
    atoms : Atoms or None
      Which objects to set transparency.
    percent : float
      Percent transparent from 0 completely opaque, to 100 completely transparent.
    target : string
      Characters indicating what to make transparent:
      a = atoms, b = bonds, p = pseudobonds, c = cartoon, s = surfaces, A = all
    """
    if atoms is None:
        from ..atomic import all_atoms
        atoms = all_atoms(session)

    if 'A' in target:
        target = 'abpcs'

    alpha = int(2.56 * (100 - percent))
    alpha = min(255, max(0, alpha))    # 0-255 range

    what = []

    if 'a' in target:
        # atoms
        c = atoms.colors
        c[:, 3] = alpha
        atoms.colors = c
        what.append('%d atoms' % len(atoms))

    if 'b' in target:
        # bonds
        bonds = atoms.inter_bonds
        if bonds:
            c = bonds.colors
            c[:, 3] = alpha
            bonds.colors = c
            what.append('%d bonds' % len(bonds))

    if 'p' in target:
        # pseudobonds
        from .. import atomic
        bonds = atomic.interatom_pseudobonds(atoms, session)
        if bonds:
            c = bonds.colors
            c[:, 3] = alpha
            bonds.colors = c
            what.append('%d pseudobonds' % len(bonds))

    if 's' in target:
        from .. import atomic
        surfs = atomic.surfaces_with_atoms(atoms, session.models)
        for s in surfs:
            vcolors = s.vertex_colors
            amask = s.atoms.mask(atoms)
            if vcolors is None and amask.all():
                c = s.colors
                c[:, 3] = alpha
                s.colors = c
            else:
                if vcolors is None:
                    from numpy import empty, uint8
                    vcolors = empty((len(s.vertices), 4), uint8)
                    vcolors[:] = s.color
                v2a = s.vertex_to_atom_map()
                v = amask[v2a]
                vcolors[v, 3] = alpha
                s.vertex_colors = vcolors
        what.append('%d surfaces' % len(surfs))

    if 'c' in target:
        residues = atoms.unique_residues
        c = residues.ribbon_colors
        c[:, 3] = alpha
        residues.ribbon_colors = c
        what.append('%d residues' % len(residues))

    if not what:
        what.append('nothing')

    from . import cli
    session.logger.status('Set transparency of %s' % cli.commas(what, ' and'))


# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, Or, AtomsArg, EmptyArg, FloatArg, StringArg
    desc = CmdDesc(required=[('atoms', Or(AtomsArg, EmptyArg)),
                             ('percent', FloatArg)],
                   keyword=[('target', StringArg)],
                   synopsis="change object transparency")
    register('transparency', desc, transparency)
