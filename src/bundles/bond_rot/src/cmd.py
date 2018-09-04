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

def cmd_bondrot_change(session, ident, angle, frames=None):
    """Wrapper called by command line."""
    if frames is not None:
        def bondrot_step(session, frame):
            cmd_bondrot_change(session, ident=ident, angle=angle, frames=None)
        from chimerax.core.commands import motion
        motion.CallForNFrames(bondrot_step, frames, session)
        return
    from .manager import BondRotationError
    mgr = session.bond_rotations
    try:
        br = mgr.rotation_for_ident(ident)
    except BondRotationError as e:
        raise UserError(str(e))
    br.angle += angle

def cmd_bondrot_create(session, ident, bond, move="small"):
    """Wrapper called by command line."""

    mgr = session.bond_rotations
    from .manager import BondRotationError
    try:
        mgr.new_rotation(bond, ident=ident, move_smaller_side=(move == "small"), one_shot=False)
    except BondRotationError as e:
        raise UserError(str(e))

def cmd_bondrot_reset(session, ident):
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

def cmd_xbondrot(session, ident):
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

def cmd_torsion(session, atoms, value=None, *, move="small"):
    """Wrapper called by command line."""
    if len(atoms) != 4:
        raise UserError("Must specify exactly 4 atoms for 'torsion' command; you specified %d"
            % len(atoms))
    a1, a2, a3, a4 = atoms
    from chimerax.core.geometry import dihedral
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
    mgr = session.bond_rotations
    from .manager import BondRotationError
    try:
        rotater = mgr.new_rotation(bond, move_smaller_side=(move == "small"))
    except BondRotationError as e:
        raise UserError(str(e))

    if bond.smaller_side == a2:
        rotater.angle += value - cur_torsion
    else:
        rotater.angle -= value - cur_torsion
    mgr.delete_rotation(rotater)


def register_command(command_name, logger):
    """
    # Code for 'bondrot' command; currently not exposed
    from chimerax.core.commands import CmdDesc, register, create_alias
    from chimerax.core.commands import IntArg, FloatArg, Or, EmptyArg, EnumOf, PositiveIntArg
    from chimerax.atomic import BondArg
    if command_name == "bondrot create":
        desc = CmdDesc(required=[('ident', Or(IntArg,EmptyArg)), ('bond', BondArg)],
            keyword = [('move', EnumOf(("large","small")))],
            synopsis = 'Activate bond for rotation'
        )
        register('bondrot create', desc, cmd_bondrot_create, logger=logger)
    elif command_name == "bondrot":
        desc = CmdDesc(required=[('ident', IntArg), ('angle', FloatArg)],
            keyword = [('frames', PositiveIntArg)],
            synopsis = 'Change bond bondrot'
        )
        register('bondrot', desc, cmd_bondrot_change, logger=logger)
    elif command_name == "bondrot reset":
        desc = CmdDesc(required = [('ident', Or(IntArg,EmptyArg))],
            synopsis = 'Reset bond rotation(s) to starting position(s)')
        register('bondrot reset', desc, cmd_bondrot_reset, logger=logger)
    else:
        desc = CmdDesc(required = [('ident', Or(IntArg,EmptyArg))],
            synopsis = 'Deactivate bond rotation(s)')
        register('bondrot delete', desc, cmd_xbondrot, logger=logger)
        create_alias('~bondrot', 'bondrot delete $*', logger=logger)
    """
    from chimerax.core.commands import CmdDesc, register
    from chimerax.core.commands import FloatArg, EnumOf
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(required = [('atoms', AtomsArg)],
        optional=[('value', FloatArg)],
        keyword = [('move', EnumOf(("large","small")))],
        synopsis = 'Set or report torsion angle')
    register('torsion', desc, cmd_torsion, logger=logger)
