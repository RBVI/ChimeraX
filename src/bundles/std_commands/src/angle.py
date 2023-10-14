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

from chimerax.dist_monitor import SimpleMeasurable, ComplexMeasurable

class SetAngleError(ValueError):
    pass

def angle_atoms_check(a1, a2, a3, *, move_smaller=True):
    try:
        atoms1 = a1.side_atoms(a2, a3)
        atoms2 = a3.side_atoms(a2, a1)
    except ValueError:
        raise SetAngleError("Cannot set the angle if the end atoms have a connection that does not pass"
            " through the center atom")
    if (len(atoms1) > len(atoms2) and move_smaller) or (len(atoms1) < len(atoms2) and not move_smaller):
        moving, fixed, moving_atoms = a3, a1, atoms2
    else:
        moving, fixed, moving_atoms = a1, a3, atoms1
    return moving, fixed, moving_atoms

def set_angle(a1, a2, a3, degrees, *, move_smaller=True, prev_axis=None, undo_state=None):
    moving, fixed, moving_atoms = angle_atoms_check(a1, a2, a3, move_smaller=move_smaller)
    mv = moving.scene_coord - a2.scene_coord
    fv = fixed.scene_coord - a2.scene_coord
    # Due to numeric roundoff, an angle previously set to 180 degrees won't be exactly at 180,
    # so "normalize" will succeed, so we need another kind of test for this degenrate condition
    from chimerax.geometry import normalize_vector, distance_squared, cross_product, angle, rotation
    from numpy import array, float32
    mv = normalize_vector(mv)
    fv = normalize_vector(fv)
    if distance_squared(mv, fv) < 0.0001 or distance_squared(mv, -fv) < 0.0001:
        if prev_axis is None:
            cross_axis = array([1.0, 0.0, 0.0], float32)
            # Use a different arbitrary axis if the vectors lie on the X axis
            if distance_squared(cross_axis, mv) < 0.0001 or distance_squared(cross_axis, -mv) < 0.0001:
                cross_axis = array([0.0, 0.0, 1.0], float32)
            axis = cross_product(cross_axis, mv)
            axis = normalize_vector(axis)
        else:
            axis = prev_axis
    else:
        axis = cross_product(fv, mv)
        axis = normalize_vector(axis)

    amount = degrees - angle(fv, mv)

    # actually rotate (about a2, but in 'moving's coordinate system)
    center = moving.structure.position.inverse() * a2.scene_coord
    xform = rotation(axis, amount, center=center)
    moved = xform.transform_points(moving_atoms.coords)
    if undo_state:
        undo_state.add(moving_atoms, "coords", moving_atoms.coords, moved)
    moving_atoms.coords = moved

    return axis

def angle(session, objects, degrees=None, *, move="small"):
    '''
    Report/set angle between three atoms or two objects.
    '''
    from chimerax.core.errors import UserError, LimitationError
    simples = [m for m in objects.models if isinstance(m, SimpleMeasurable)]
    complexes = [m for m in objects.models if isinstance(m, ComplexMeasurable)]
    atoms = objects.atoms
    all_simples = simples + list(atoms)
    if degrees is None:
        # report value
        arg_error_msg = "Must specify exactly 3 atoms/centroids or two measurable objects" \
            " (e.g. axes/planes)"
        if complexes:
            if len(complexes) != 2:
                raise UserError(arg_error_msg)
            angle = complexes[0].angle(complexes[1])
            if angle is NotImplemented:
                angle = complexes[1].angle(complexes[0])
            if angle is NotImplemented:
                raise LimitationError("Don't know how to measure angle between %s and %s" % tuple(complexes))
            participants = complexes
        else:
            if len(all_simples) != 3:
                raise UserError(arg_error_msg)
            if len(atoms) == 3:
                participants = atoms
            else:
                # Have to order the non-atoms correctly among the atoms
                model_order = { m:i for i, m in enumerate(objects.models) }
                from chimerax.atomic import Atom
                all_simples.sort(key=lambda s, atoms=atoms, mo=model_order:
                    (mo[s.structure], atoms.index(s)) if isinstance(s, Atom) else (mo[s], 0))
                participants = all_simple
            from chimerax import geometry
            angle = geometry.angle(*(x.scene_coord for x in participants))
        from chimerax.core.commands import commas
        session.logger.info("Angle between %s: %.3f" % (commas([str(p) for p in participants],
            conjunction="and"), angle))
        return angle

    if simples or complexes:
        raise LimitationError("Cannot set angle involving non-atoms (e.g. planes, axes)")
    if len(atoms) != 3:
        raise UserError("To set the bond angle you must specify exactly 3 bonded atoms")
    from chimerax.core.undo import UndoState
    undo_state = UndoState("angle")
    try:
        set_angle(*atoms, degrees, move_smaller=(move == "small"), undo_state=undo_state)
    except SetAngleError as e:
        raise UserError(str(e))
    session.undo.register(undo_state)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, EnumOf, ObjectsArg
    from chimerax.atomic import PseudobondsArg
    desc = CmdDesc(
        required = [('objects', ObjectsArg)],
        optional = [('degrees', FloatArg)],
        keyword = [('move', EnumOf(("small", "large")))],
        synopsis = 'set/report angle between objects/atoms')
    register('angle', desc, angle, logger=logger)
