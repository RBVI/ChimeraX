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

def cmd_start_structure(session, method, model_info, subargs):
    from .manager import manager
    if manager.is_indirect(method):
        raise UserError("No command support for '%s' start-structure method" % method)
    if isinstance(model_info, str):
        from chimerax.atomic import AtomicStructure
        model = AtomicStructure(session, name=model_info)
    else:
        model = model_info
    try:
        manager.execute_command(method, model, subargs)
    except BaseException:
        if isinstance(model_info, str):
            model.delete()
        raise
    if model.num_atoms == 0:
        model.delete()
    elif isinstance(model_info, str):
        session.models.add([model])

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, Or, IntArg, EnumOf, StringArg
    from chimerax.core.commands import DynamicEnum, RestOfLine
    from chimerax.atomic import AtomArg, ElementArg, StructureArg
    from chimerax.atomic.bond_geom import geometry_name
    desc = CmdDesc(
        required=[('atom', AtomArg), ('element', ElementArg), ('num_bonds', IntArg)],
        keyword = [('geometry', EnumOf(range(len(geometry_name)), ids=geometry_name)),
            ('name', StringArg), ('connect_back', BoolArg), ('color_by_element', BoolArg),
            ('res_name', StringArg), ('new_res', BoolArg)],
        synopsis = 'modify atom'
    )
    register('build modify', desc, cmd_modify_atom, logger=logger)

    from .manager import manager
    desc = CmdDesc(
        required=[('method', DynamicEnum(lambda mgr=manager: mgr.provider_names)),
            ('model_info', Or(StructureArg, StringArg)), ('subargs', RestOfLine)],
        keyword = [],
        synopsis = 'start structure'
    )
    register('build start', desc, cmd_start_structure, logger=logger)
