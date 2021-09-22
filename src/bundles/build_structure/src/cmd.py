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

def cmd_bond(session, *args, **kw):
    from .bond import create_bonds, CreateBondError
    try:
        created = create_bonds(*args, **kw)
    except CreateBondError as e:
        raise UserError(str(e))
    from chimerax.core.commands import plural_form
    session.logger.info("Created %d %s" % (len(created), plural_form(created, "bond")))

def cmd_bond_length(session, bond, length=None, *, move="small"):
    if length is None:
        session.logger.info(("Bond length for %s is " + session.pb_dist_monitor.distance_format)
            % (bond, bond.length))
    else:
        from chimerax.atomic.struct_edit import set_bond_length
        set_bond_length(bond, length, move_smaller_side=(move=="small"))

def cmd_modify_atom(session, *args, **kw):
    from .mod import modify_atom, ParamError
    try:
        return modify_atom(*args, **kw)
    except ParamError as e:
        raise UserError(e)

def cmd_start_structure(session, method, model_info, subargs):
    from .manager import get_manager
    manager = get_manager(session)
    if manager.is_indirect(method):
        raise UserError("No command support for '%s' start-structure method" % method)
    if isinstance(model_info, str):
        from chimerax.atomic import AtomicStructure
        model = AtomicStructure(session, name=model_info, auto_style=manager.auto_style(method))
    else:
        model = model_info
    try:
        ret_val = manager.execute_command(method, model, subargs)
    except BaseException:
        if isinstance(model_info, str):
            model.delete()
        raise
    if model.num_atoms == 0:
        model.delete()
    elif isinstance(model_info, str):
        session.models.add([model])
    return ret_val

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, Or, IntArg, EnumOf, StringArg
    from chimerax.core.commands import DynamicEnum, RestOfLine, create_alias, PositiveFloatArg
    from chimerax.atomic import AtomArg, ElementArg, StructureArg, AtomsArg, BondArg
    from chimerax.atomic.bond_geom import geometry_name
    desc = CmdDesc(
        required=[('atom', AtomArg), ('element', ElementArg), ('num_bonds', IntArg)],
        keyword = [('geometry', EnumOf(range(len(geometry_name)), ids=geometry_name)),
            ('name', StringArg), ('connect_back', BoolArg), ('color_by_element', BoolArg),
            ('res_name', StringArg), ('new_res', BoolArg)],
        synopsis = 'modify atom'
    )
    register('build modify', desc, cmd_modify_atom, logger=logger)

    from .manager import get_manager
    manager = get_manager(logger.session)
    desc = CmdDesc(
        required=[('method', DynamicEnum(lambda mgr=manager: mgr.provider_names)),
            ('model_info', Or(StructureArg, StringArg)), ('subargs', RestOfLine)],
        keyword = [],
        synopsis = 'start structure'
    )
    register('build start', desc, cmd_start_structure, logger=logger)

    desc = CmdDesc(
        required=[('atoms', AtomsArg)],
        keyword = [('reasonable', BoolArg)],
        synopsis = 'add bond(s)'
    )
    register('bond', desc, cmd_bond, logger=logger)

    create_alias("~bond", "delete bonds $*", logger=logger)

    desc = CmdDesc(
        required=[('bond', BondArg)],
        optional = [('length', PositiveFloatArg)],
        keyword = [('move', EnumOf(("large", "small")))],
        synopsis = 'set bond length'
    )
    register('bond length', desc, cmd_bond_length, logger=logger)
