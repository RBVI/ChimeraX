# vim: set expandtab ts=4 sw=4:

from chimerax.core.commands import CmdDesc
from chimerax.atomic import AtomicStructure, AtomicStructuresArg


def viewdock(session, structures=None):
    if structures is None:
        structures = session.models.list(type=AtomicStructure)
    from .tool import TableTool
    tool = TableTool(session, "ViewDockX")
    tool.setup(structures)
    return tool
viewdock_desc = CmdDesc(optional=[("structures", AtomicStructuresArg)])
