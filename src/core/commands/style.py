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

def style(session, objects=None, atom_style=None, atom_radius=None,
          stick_radius=None, pseudobond_radius=None, ball_scale=None, dashes=None):
    '''Set the atom and bond display styles and sizes.

    Parameters
    ----------
    objects : Objects
        Change the style of these atoms and pseudobond groups.
        If not specified then all atoms are changed.
    atom_style : "sphere", "ball" or "stick"
        Controls how atoms and bonds are depicted.
    atom_radius : float or "default"
      New radius value for atoms.
    stick_radius : float
      New radius value for bonds shown in stick style.
    pseudobond_radius : float
      New radius value for pseudobonds.
    ball_scale : float
      Multiplier times atom radius for determining atom size in ball style (default 0.3).
    dashes : int
      Number of dashes shown for pseudobonds.
    '''
    if objects is None:
        from . import atomspec
        objects = atomspec.all_objects(session)
    atoms = objects.atoms

    from ..undo import UndoState
    undo_state = UndoState("style")
    what = []
    if atom_style is not None:
        from ..atomic import Atom
        s = {
            'sphere': Atom.SPHERE_STYLE,
            'ball': Atom.BALL_STYLE,
            'stick': Atom.STICK_STYLE,
        }[atom_style.lower()]
        undo_state.add(atoms, "draw_modes", atoms.draw_modes, s)
        atoms.draw_modes = s
        what.append('%d atom styles' % len(atoms))

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

    if dashes is not None:
        pbgs = objects.pseudobonds.unique_groups
        for pbg in pbgs:
            undo_state.add(pbg, "dashes", pbg.dashes, dashes)
            pbg.dashes = dashes
        what.append('%d pseudobond dashes' % len(pbgs))

    if what:
        msg = 'Changed %s' % ', '.join(what)
        log = session.logger
        log.status(msg)
        log.info(msg)

    session.undo.register(undo_state)

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, ObjectsArg, EmptyArg, EnumOf, Or, IntArg, FloatArg
    desc = CmdDesc(required = [('objects', Or(ObjectsArg, EmptyArg)),
                               ('atom_style', Or(EnumOf(('sphere', 'ball', 'stick')), EmptyArg))],
                   keyword = [('atom_radius', Or(EnumOf(['default']), FloatArg)),
                              ('stick_radius', FloatArg),
                              ('pseudobond_radius', FloatArg),
                              ('ball_scale', FloatArg),
                              ('dashes', IntArg)],
                   synopsis='change atom and bond depiction')
    register('style', desc, style, logger=session.logger)
