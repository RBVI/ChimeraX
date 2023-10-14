# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def size(session, objects=None, atom_radius=None,
          stick_radius=None, pseudobond_radius=None, ball_scale=None):
    '''Set the sizes of atom and bonds.

    Parameters
    ----------
    objects : Objects
        Change the size of these atoms, bonds and pseudobonds.
        If not specified then all are changed.
    atom_radius : float, (bool, float) or "default"
      New radius value for atoms.  The optional boolean is whether the float is a delta.
    stick_radius : float or (bool, float)
      New radius value for bonds shown in stick style. The optional boolean is whether the float is a delta.
    pseudobond_radius : float or (bool, float)
      New radius value for pseudobonds. The optional boolean is whether the float is a delta.
    ball_scale : float or (bool, float)
      Multiplier times atom radius for determining atom size in ball style (default 0.3).
      The optional boolean is whether the float is a delta.
    '''
    if objects is None:
        from chimerax.core.objects import all_objects
        objects = all_objects(session)

    from chimerax.core.undo import UndoState
    undo_state = UndoState("size")
    what = []

    from chimerax.core.errors import UserError
    if atom_radius is not None:
        atoms = objects.atoms
        if atom_radius == 'default':
            undo_state.add(atoms, "radii", atoms.radii, atoms.default_radii)
            atoms.radii = atoms.default_radii
        else:
            if isinstance(atom_radius, tuple):
                is_delta, amount = atom_radius
            else:
                is_delta, amount = False, atom_radius
            if is_delta:
                old_radii = atoms.radii
                if amount < 0 and len(old_radii) > 0 and min(old_radii) <= abs(amount):
                    raise UserError("Cannot reduce atom radius to <= 0")
                atoms.radii += amount
                undo_state.add(atoms, "radii", old_radii, atoms.radii)
            else:
                if amount <= 0:
                    raise UserError('Atom radius must be greater than 0.')
                undo_state.add(atoms, "radii", atoms.radii, amount)
                atoms.radii = amount
        what.append('%d atom radii' % len(atoms))

    if stick_radius is not None:
        b = objects.bonds
        if isinstance(stick_radius, tuple):
            is_delta, amount = stick_radius
        else:
            is_delta, amount = False, stick_radius
        if is_delta:
            old_radii = b.radii
            if amount < 0 and len(old_radii) > 0 and min(old_radii) <= abs(amount):
                raise UserError("Cannot reduce stick radius to <= 0")
            b.radii += amount
            undo_state.add(b, "radii", old_radii, b.radii)
            # If singleton atom specified then set the single-atom stick radius.
            for s, atoms in objects.atoms.by_structure:
                if (atoms.num_bonds == 0).any():
                    if amount < 0 and s.bond_radius < amount:
                        raise UserError("Cannot reduce bond radius to <= 0")
                    s.bond_radius += amount
        else:
            if amount <= 0:
                raise UserError('Bond radius must be greater than 0.')
            undo_state.add(b, "radii", b.radii, amount)
            b.radii = amount
            # If singleton atom specified then set the single-atom stick radius.
            for s, atoms in objects.atoms.by_structure:
                if (atoms.num_bonds == 0).any():
                    s.bond_radius = amount
        what.append('%d bond radii' % len(b))

    if pseudobond_radius is not None:
        if isinstance(pseudobond_radius, tuple):
            is_delta, amount = pseudobond_radius
        else:
            is_delta, amount = False, pseudobond_radius
        pb = objects.pseudobonds
        if is_delta:
            old_radii = pb.radii
            if amount < 0 and len(old_radii) > 0 and min(old_radii) <= abs(amount):
                raise UserError("Cannot reduce pseudobond radius to <= 0")
            pb.radii += amount
            undo_state.add(pb, "radii", old_radii, pb.radii)
        else:
            if amount <= 0:
                raise UserError('Pseudobond radius must be greater than 0.')
            undo_state.add(pb, "radii", pb.radii, amount)
            pb.radii = amount
            from chimerax.atomic import concatenate
        what.append('%d pseudobond radii' % len(pb))

    if ball_scale is not None:
        if isinstance(ball_scale, tuple):
            is_delta, amount = ball_scale
        else:
            is_delta, amount = False, ball_scale
        mols = objects.residues.unique_structures
        if is_delta:
            for s in mols:
                if amount < 0 and s.ball_scale + amount <= 0:
                    raise UserError("Cannot reduce ball scale to <= 0")
                undo_state.add(s, "ball_scale", s.ball_scale, s.ball_scale + amount)
                s.ball_scale += amount
        else:
            if amount <= 0:
                raise UserError('Ball scale must be greater than 0.')
            for s in mols:
                undo_state.add(s, "ball_scale", s.ball_scale, amount)
                s.ball_scale = amount
        what.append('%d ball scales' % len(mols))

    if what:
        msg = 'Changed %s' % ', '.join(what)
        log = session.logger
        log.status(msg)
        log.info(msg)

    session.undo.register(undo_state)

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, ObjectsArg, EmptyArg, EnumOf, Or, FloatOrDeltaArg
    desc = CmdDesc(required = [('objects', Or(ObjectsArg, EmptyArg))],
                   keyword = [('atom_radius', Or(EnumOf(['default']), FloatOrDeltaArg)),
                              ('stick_radius', FloatOrDeltaArg),
                              ('pseudobond_radius', FloatOrDeltaArg),
                              ('ball_scale', FloatOrDeltaArg)],
                   synopsis='change atom and bond sizes')
    register('size', desc, size, logger=logger)
