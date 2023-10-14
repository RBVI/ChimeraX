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

from chimerax.core.errors import UserError

def cmd_torsion(session, atoms, value=None, *, move="small"):
    """Wrapper called by command line."""
    if len(atoms) != 4:
        raise UserError("Must specify exactly 4 atoms for 'torsion' command; you specified %d"
            % len(atoms))
    a1, a2, a3, a4 = atoms
    from chimerax.geometry import dihedral
    cur_torsion = dihedral(*[a.scene_coord for a in atoms])
    if value is None:
        session.logger.info("Torsion angle for atoms %s %s %s %s is %g\N{DEGREE SIGN}"
            % (a1, a2.string(relative_to=a1), a3.string(relative_to=a2), a4.string(relative_to=a3),
            cur_torsion))
        return
    for nb, bond in zip(a2.neighbors, a2.bonds):
        if nb == a3:
            break
    else:
        raise UserError("To set torsion, middle two atoms (%s %s) must be bonded;they aren't"
            % (a2, a3.string(relative_to=a2)))
    move_smaller = move == "small"
    mgr = session.bond_rotations
    from .manager import BondRotationError
    try:
        rotater = mgr.new_rotation(bond, move_smaller_side=move_smaller)
    except BondRotationError as e:
        raise UserError(str(e))

    from chimerax.core.undo import UndoState
    rotater.undo_state = UndoState("torsion")
    rotater.angle += value - cur_torsion
    rotater.undo_state = None
    mgr.delete_rotation(rotater)


def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import FloatArg, EnumOf
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(required = [('atoms', AtomsArg)],
        optional=[('value', FloatArg)],
        keyword = [('move', EnumOf(("large","small")))],
        synopsis = 'Set or report torsion angle')
    register('torsion', desc, cmd_torsion, logger=logger)
