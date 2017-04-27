# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def transparency(session, objects, percent, target='s'):
    """Set transparency of atoms, ribbons, surfaces, ....

    Parameters
    ----------
    objects : Objects or None
      Which objects to set transparency.
    percent : float
      Percent transparent from 0 completely opaque, to 100 completely transparent.
    target : string
      Characters indicating what to make transparent:
      a = atoms, b = bonds, p = pseudobonds, c = cartoon, r = cartoon, s = surfaces, A = all
    """
    if objects is None:
        from . import all_objects
        objects = all_objects(session)
    atoms = objects.atoms

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
        bonds = atoms.intra_bonds
        if bonds:
            c = bonds.colors
            c[:, 3] = alpha
            bonds.colors = c
            what.append('%d bonds' % len(bonds))

    if 'p' in target:
        # pseudobonds
        from .. import atomic
        bonds = atomic.interatom_pseudobonds(atoms)
        if bonds:
            c = bonds.colors
            c[:, 3] = alpha
            bonds.colors = c
            what.append('%d pseudobonds' % len(bonds))

    if 's' in target:
        surfs = _set_surface_transparency(atoms, objects, session, alpha)
        what.append('%d surfaces' % len(surfs))

    if 'c' in target or 'r' in target:
        residues = atoms.unique_residues
        c = residues.ribbon_colors
        c[:, 3] = alpha
        residues.ribbon_colors = c
        what.append('%d residues' % len(residues))

    if not what:
        what.append('nothing')

    from . import cli
    session.logger.status('Set transparency of %s' % cli.commas(what, ' and'))

def _set_surface_transparency(atoms, objects, session, alpha):

    # Handle surfaces for specified atoms
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
            if v2a is None:
                if amask.all():
                    v = slice(len(vcolors))
                else:
                    session.logger.info('No atom associations for surface #%s'
                                        % s.id_string())
                    continue
            else:
                v = amask[v2a]
            vcolors[v, 3] = alpha
            s.vertex_colors = vcolors

    # Handle surface models specified without specifying atoms
    from ..atomic import MolecularSurface, Structure
    from ..map import Volume
    osurfs = []
    for s in objects.models:
        if isinstance(s, MolecularSurface):
            if not s in surfs:
                osurfs.append(s)
        elif isinstance(s, Volume) or (not isinstance(s, Structure) and not s.empty_drawing()):
            osurfs.append(s)
    for s in osurfs:
        s.set_transparency(alpha)
    surfs.extend(osurfs)
            
    return surfs


# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, Or, ObjectsArg, EmptyArg, FloatArg, StringArg
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg)),
                             ('percent', FloatArg)],
                   keyword=[('target', StringArg)],
                   synopsis="change object transparency")
    register('transparency', desc, transparency, logger=session.logger)
