from chimerax.core.commands import CmdDesc, StringArg
from chimerax.atomic import AtomicStructuresArg


def viewdock(session, structures=None, name=None):
    from .tool import ViewDockTool
    return ViewDockTool(session, "ViewDock")

viewdock_desc = CmdDesc(optional=[("structures", AtomicStructuresArg),
                                  ("name", StringArg)])


def viewdock_down(session, name=None):
    session.logger.info("viewdock_down not implemented yet")
viewdock_down_desc = CmdDesc(optional=[("name", StringArg)])


def viewdock_up(session, name=None):
    session.logger.info("viewdock_up not implemented yet")
viewdock_up_desc = CmdDesc(optional=[("name", StringArg)])


command_map = {
    "viewdock": (viewdock, viewdock_desc),
    "viewdock down": (viewdock_down, viewdock_down_desc),
    "viewdock up": (viewdock_up, viewdock_up_desc),
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