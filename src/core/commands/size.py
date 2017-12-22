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

def size(session, objects=None, atom_radius=None,
          stick_radius=None, pseudobond_radius=None, ball_scale=None):
    '''Set the sizes of atom and bonds.

    Parameters
    ----------
    objects : Objects
        Change the size of these atoms, bonds and pseudobonds.
        If not specified then all are changed.
    atom_radius : float or "default"
      New radius value for atoms.
    stick_radius : float
      New radius value for bonds shown in stick style.
    pseudobond_radius : float
      New radius value for pseudobonds.
    ball_scale : float
      Multiplier times atom radius for determining atom size in ball style (default 0.3).
    '''
    if objects is None:
        from . import atomspec
        objects = atomspec.all_objects(session)
    atoms = objects.atoms

    from ..undo import UndoState
    undo_state = UndoState("size")
    what = []

    if atom_radius is not None:
        if atom_radius == 'default':
            undo_state.add(atoms, "radii", atoms.radii, atoms.default_radii)
            atoms.radii = atoms.default_radii
        else:
            undo_state.add(atoms, "radii", atoms.radii, atom_radius)
            atoms.radii = atom_radius
        what.append('%d atom radii' % len(atoms))

    if stick_radius is not None:
        b = objects.bonds
        undo_state.add(b, "radii", b.radii, stick_radius)
        b.radii = stick_radius
        what.append('%d bond radii' % len(b))

    if pseudobond_radius is not None:
        pb = objects.pseudobonds
        undo_state.add(pb, "radii", pb.radii, pseudobond_radius)
        pb.radii = pseudobond_radius
        from ..atomic import concatenate
        what.append('%d pseudobond radii' % len(pb))

    if ball_scale is not None:
        mols = atoms.unique_structures
        for s in mols:
            undo_state.add(s, "ball_scale", s.ball_scale, ball_scale)
            s.ball_scale = ball_scale
        what.append('%d ball scales' % len(mols))

    if what:
        msg = 'Changed %s' % ', '.join(what)
        log = session.logger
        log.status(msg)
        log.info(msg)

    session.undo.register(undo_state)

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, ObjectsArg, EmptyArg, EnumOf, Or, FloatArg
    desc = CmdDesc(required = [('objects', Or(ObjectsArg, EmptyArg))],
                   keyword = [('atom_radius', Or(EnumOf(['default']), FloatArg)),
                              ('stick_radius', FloatArg),
                              ('pseudobond_radius', FloatArg),
                              ('ball_scale', FloatArg)],
                   synopsis='change atom and bond sizes')
    register('size', desc, size, logger=session.logger)
