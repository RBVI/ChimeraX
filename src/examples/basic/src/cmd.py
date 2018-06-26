# vim: set expandtab ts=4 sw=4:

from chimerax.core.commands import CmdDesc
from chimerax.atomic import AtomicStructure, AtomicStructuresArg


def basic(session, structures=None):
    if structures is None:
        structures = session.models.list(type=AtomicStructure)
    for m in structures:
        session.logger.info("%s: %d atoms, %d bonds" %
                            (m, m.num_atoms, m.num_bonds))
basic_desc = CmdDesc(optional=[("structures", AtomicStructuresArg)])
