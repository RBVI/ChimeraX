# vim: set expandtab ts=4 sw=4:

from chimerax.core.commands import CmdDesc, AtomicStructuresArg
from chimerax.core.atomic import AtomicStructure


def viewdock(session, structures=None):
    if structures is None:
        structures = session.models.list(type=AtomicStructure)
    from .tool import TableTool
    return TableTool(session, "ViewDockX", structures=structures)
viewdock_desc = CmdDesc(optional=[("structures", AtomicStructuresArg)])
