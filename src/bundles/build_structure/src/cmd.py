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

def cmd_modify_atom(session, *args, **kw):
    from .mod import modify_atom, ParamError
    try:
        return modify_atom(*args, **kw)
    except ParamError as e:
        raise UserError(e)

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, Or, IntArg, EnumOf, StringArg
    from chimerax.atomic import AtomArg, ElementArg
    from chimerax.atomic.bond_geom import geometry_name
    desc = CmdDesc(
        required=[('atom', AtomArg), ('element', ElementArg), ('num_bonds', IntArg)],
        keyword = [('geometry', EnumOf(range(len(geometry_name)), ids=geometry_name)),
            ('name', StringArg), ('connect_back', BoolArg), ('color_by_element', BoolArg),
            ('res_name', StringArg), ('res_new_only', BoolArg)],
        synopsis = 'modify atom'
    )
    register('structure modify', desc, cmd_modify_atom, logger=logger)
