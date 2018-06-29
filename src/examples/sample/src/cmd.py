# vim: set expandtab ts=4 sw=4:

from chimerax.core.commands import CmdDesc
from chimerax.atomic import AtomicStructure, AtomicStructuresArg
from . import _sample


def sample_count(session, structures=None):
    if structures is None:
        structures = session.models.list(type=AtomicStructure)
    for m in structures:
        atoms, bonds = _sample.counts(m)
        session.logger.info("%s: %d atoms, %d bonds" % (m, atoms, bonds))
sample_count_desc = CmdDesc(optional=[("structures", AtomicStructuresArg)])
