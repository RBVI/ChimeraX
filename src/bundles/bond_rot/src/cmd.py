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

from chimerax.core.errors import UserError

def cmd_torsion_change(session, ident, angle, frames=None):
    """Wrapper called by command line."""
    if frames is not None:
        def torsion_step(session, frame):
            cmd_torsion_change(session, ident=ident, angle=angle, frames=None)
        from chimerax.core.commands import motion
        motion.CallForNFrames(torsion_step, frames, session)
        return
    from .manager import BondRotationError
    mgr = session.bond_rotations
    try:
        br = mgr.rotation_for_ident(ident)
    except BondRotationError as e:
        raise UserError(str(e))
    br.angle += angle

def cmd_torsion_create(session, ident, bond, move="small"):
    """Wrapper called by command line."""

    mgr = session.bond_rotations
    from .manager import BondRotationError
    try:
        mgr.new_rotation(bond, ident=ident, move_smaller_side=(move == "small"), one_shot=False)
    except BondRotationError as e:
        raise UserError(str(e))

def cmd_torsion_reset(session, ident):
    """Wrapper called by command line."""
    mgr = session.bond_rotations
    if ident is None:
        rotations = mgr.bond_rotaters.values()
    else:
        try:
            rotations = [mgr.rotation_for_ident(ident)]
        except BondRotationError as e:
            raise UserError(str(e))
    for rot in rotations:
        rot.angle = 0

def cmd_xtorsion(session, ident):
    """Wrapper called by command line."""
    mgr = session.bond_rotations
    if ident is None:
        mgr.delete_all_rotations()
        return
    try:
        br = mgr.rotation_for_ident(ident)
    except BondRotationError as e:
        raise UserError(str(e))
    mgr.delete_rotation(br)

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, create_alias
    from chimerax.core.commands import IntArg, FloatArg, Or, EmptyArg, EnumOf, PositiveIntArg
    from chimerax.atomic import BondArg
    if command_name == "torsion create":
        desc = CmdDesc(required=[('ident', Or(IntArg,EmptyArg)), ('bond', BondArg)],
            keyword = [('move', EnumOf(("large","small")))],
            synopsis = 'Activate bond for rotation'
        )
        register('torsion create', desc, cmd_torsion_create, logger=logger)
    elif command_name == "torsion":
        desc = CmdDesc(required=[('ident', IntArg), ('angle', FloatArg)],
            keyword = [('frames', PositiveIntArg)],
            synopsis = 'Change bond torsion'
        )
        register('torsion', desc, cmd_torsion_change, logger=logger)
    elif command_name == "torsion reset":
        desc = CmdDesc(required = [('ident', Or(IntArg,EmptyArg))],
            synopsis = 'Reset bond rotation(s) to starting position(s)')
        register('torsion reset', desc, cmd_torsion_reset, logger=logger)
    else:
        desc = CmdDesc(required = [('ident', Or(IntArg,EmptyArg))],
            synopsis = 'Deactivate bond rotation(s)')
        register('torsion delete', desc, cmd_xtorsion, logger=logger)
        create_alias('~torsion', 'torsion delete $*', logger=logger)
