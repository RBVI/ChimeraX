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

def cmd_torsion_create(session, ident, bond, move="small"):
    """Wrapper called by command line."""

    mgr = session.bond_rotations
    br = mgr.rotation_for_bond(bond, create=False)
    if br:
        raise UserError("Bond rotation already active for that bond (ident: %d)" % br.ident)
    from .manager import BondRotationError
    try:
        mgr.new_rotation(bond, ident=ident, move_smaller_side=(move == "small"), one_shot=False)
    except BondRotationError as e:
        raise UserError(str(e))

def cmd_xtorsion(session, ident):
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
    from chimerax.core.commands \
        import CmdDesc, register, BondArg, IntArg, Or, EmptyArg, EnumOf, create_alias
    if command_name == "torsion create":
        desc = CmdDesc(required=[('ident', Or(IntArg,EmptyArg)), ('bond', BondArg)],
            keyword = [('move', EnumOf(("large","small")))],
            synopsis = 'Activate bond for rotation'
        )
        register('torsion create', desc, cmd_torsion_create, logger=logger)
    elif command_name == "torsion":
        #TODO
        # syntax for torsion adjustment command will be: torsion ident angle [frames]
        pass
    else:
        desc = CmdDesc(required = [('ident', Or(IntArg,EmptyArg))],
            synopsis = 'Deactivate bond rotation(s)')
        register('torsion delete', desc, cmd_xtorsion, logger=logger)
        create_alias('~torsion', 'torsion delete $*', logger=logger)
