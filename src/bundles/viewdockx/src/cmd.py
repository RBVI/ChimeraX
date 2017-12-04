# vim: set expandtab ts=4 sw=4:

from chimerax.core.commands import CmdDesc, AtomicStructuresArg
from chimerax.core.atomic import AtomicStructure
# from . import _sample


def viewdock(session, structures=None):
    if structures is None:
        structures = session.models.list(type=AtomicStructure)
    from .tool import ViewDockTool
    return ViewDockTool(session, "viewdock", structures = structures)
viewdock_desc = CmdDesc(optional=[("structures", AtomicStructuresArg)],
                            synopsis="log model atom and bond counts")
