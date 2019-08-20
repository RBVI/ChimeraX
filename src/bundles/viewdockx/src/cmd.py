# vim: set expandtab ts=4 sw=4:

from chimerax.core.commands import CmdDesc, StringArg
from chimerax.atomic import AtomicStructure, AtomicStructuresArg


def viewdock(session, structures=None, name=None):
    if structures is None:
        structures = session.models.list(type=AtomicStructure)
    from .tool import TableTool
    tool = TableTool(session, "ViewDockX", name=name)
    tool.setup(structures)
    return tool
viewdock_desc = CmdDesc(optional=[("structures", AtomicStructuresArg),
                                  ("name", StringArg)])


def viewdock_down(session, name=None):
    from .tool import TableTool
    try:
        tool = TableTool.find(name)
    except KeyError as e:
        from chimerax.core.errors import UserError
        raise UserError(str(e))
    else:
        tool.arrow_down()
viewdock_down_desc = CmdDesc(optional=[("name", StringArg)])


def viewdock_up(session, name=None):
    from .tool import TableTool
    try:
        tool = TableTool.find(name)
    except KeyError as e:
        from chimerax.core.errors import UserError
        raise UserError(str(e))
    else:
        tool.arrow_up()
viewdock_up_desc = CmdDesc(optional=[("name", StringArg)])


command_map = {
    "viewdockx": (viewdock, viewdock_desc),
    "viewdockx down": (viewdock_down, viewdock_down_desc),
    "viewdockx up": (viewdock_up, viewdock_up_desc),
}


def register_command(ci):
    try:
        func, desc = command_map[ci.name]
    except KeyError:
        raise ValueError("trying to register unknown command: %s" % ci.name)
    if desc.synopsis is None:
        desc.synopsis = ci.synopsis
    from chimerax.core.commands import register
    register(ci.name, desc, func)
