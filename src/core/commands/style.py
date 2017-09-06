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
    from ..atomic import all_atoms, Atom
    atoms = all_atoms(session) if objects is None else objects.atoms

    undo_data = {}
    what = []
    if atom_style is not None:
        s = {
            'sphere': Atom.SPHERE_STYLE,
            'ball': Atom.BALL_STYLE,
            'stick': Atom.STICK_STYLE,
        }[atom_style.lower()]
        undo_data["atom_style"] = (atoms, atoms.draw_modes, s)
        atoms.draw_modes = s
        what.append('%d atom styles' % len(atoms))

    if atom_radius is not None:
        if atom_radius == 'default':
            undo_data["atom_radii"] = (atoms, atoms.radii, atoms.default_radii)
            atoms.radii = atoms.default_radii
        else:
            undo_data["atom_radii"] = (atoms, atoms.radii, atom_radius)
            atoms.radii = atom_radius
        what.append('%d atom radii' % len(atoms))

    if stick_radius is not None:
        b = atoms.intra_bonds
        undo_data["stick_radius"] = (b, b.radii, stick_radius)
        b.radii = stick_radius
        what.append('%d bond radii' % len(b))

    if pseudobond_radius is not None:
        from ..atomic import interatom_pseudobonds, concatenate
        pb = interatom_pseudobonds(atoms)
        undo_data["pseudobond_radius"] = (pb, pb.radii, pseudobond_radius)
        pb.radii = pseudobond_radius
        pbs = [pb]
        undo_data["pbg_radius"] = upbg = []
        for pbg in pseudobond_groups(objects, session, interatom = False):
            pb = pbg.pseudobonds
            upbg.append((pb, pb.radii, pseudobond_radius))
            pb.radii = pseudobond_radius
            pbs.append(pb)
        what.append('%d pseudobond radii' % len(concatenate(pbs, remove_duplicates=True)))

    if ball_scale is not None:
        mols = atoms.unique_structures
        undo_data["ball_scale"] = ubs = []
        for s in mols:
            ubs.append((s, s.ball_scale, ball_scale))
            s.ball_scale = ball_scale
        what.append('%d ball scales' % len(mols))

    if dashes is not None:
        pbgs = pseudobond_groups(objects, session)
        undo_data["dashes"] = ud = []
        for pbg in pbgs:
            ud.append((pbg, pbg.dashes, dashes))
            pbg.dashes = dashes
        what.append('%d pseudobond dashes' % len(pbgs))

    if what:
        msg = 'Changed %s' % ', '.join(what)
        log = session.logger
        log.status(msg)
        log.info(msg)

    def undo(data=undo_data):
        _style_undo(undo_data)
    def redo(data=undo_data):
        _style_redo(undo_data)
    session.undo.register("style", undo, redo)

# -----------------------------------------------------------------------------
#
def pseudobond_groups(objects, session, interatom = True):
    from ..atomic import PseudobondGroup, all_atoms

    # Explicitly specified global pseudobond groups
    models = session.models.list() if objects is None else objects.models
    pbgs = set(m for m in models if isinstance(m, PseudobondGroup))

    if interatom:
        atoms = all_atoms(session) if objects is None else objects.atoms

        # Intra-molecular pseudobond groups with bonds between specified atoms.
        for m in atoms.unique_structures:
            molpbgs = [pbg for pbg in m.pbg_map.values()
                       if pbg.pseudobonds.between_atoms(atoms).any()]
            pbgs.update(molpbgs)

        # Global pseudobond groups with bonds between specified atoms
        gpbgs = [pbg for pbg in session.models.list(type = PseudobondGroup)
                 if pbg.pseudobonds.between_atoms(atoms).any()]
        pbgs.update(gpbgs)

    return pbgs

# -----------------------------------------------------------------------------
#
def _update_attr(undo_data, key, attr, n):
    try:
        items = undo_data[key]
    except KeyError:
        return
    if isinstance(items, list):
        for v in items:
            try:
                container = v[0]
                value = v[n]
            except (ValueError, IndexError):
                pass
            else:
                setattr(container, attr, value)
    else:
        try:
            container = items[0]
            value = items[n]
        except (KeyError, ValueError, IndexError):
            pass
        else:
            setattr(container, attr, value)

def _style_undo(undo_data):
    _update_attr(undo_data, "atom_style", "draw_modes", 1)
    _update_attr(undo_data, "atom_radii", "radii", 1)
    _update_attr(undo_data, "stick_radius", "radii", 1)
    _update_attr(undo_data, "pseudobond_radius", "radii", 1)
    _update_attr(undo_data, "pbg_radius", "radii", 1)
    _update_attr(undo_data, "ball_scale", "ball_scale", 1)
    _update_attr(undo_data, "dashes", "dashes", 1)

def _style_redo(undo_data):
    _update_attr(undo_data, "atom_style", "draw_modes", 2)
    _update_attr(undo_data, "atom_radii", "radii", 2)
    _update_attr(undo_data, "stick_radius", "radii", 2)
    _update_attr(undo_data, "pseudobond_radius", "radii", 2)
    _update_attr(undo_data, "pbg_radius", "radii", 2)
    _update_attr(undo_data, "ball_scale", "ball_scale", 2)
    _update_attr(undo_data, "dashes", "dashes", 2)


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
