# vim: set expandtab ts=4 sw=4:

from chimerax.core.commands import CmdDesc, StringArg
from chimerax.atomic import AtomicStructure, AtomicStructuresArg
from . import _sample_pyapi, _sample_pybind11

api_modules={
    'python':   (_sample_pyapi, None),
    'pybind11': (_sample_pybind11, 'cpp_pointer'),
}


def sample_count(session, structures=None, api="python"):
    if structures is None:
        structures = session.models.list(type=AtomicStructure)
    module_def = api_modules.get(api.lower(), None)
    if module_def is None:
        from chimerax.core.errors import UserError
        err_string = ('Unrecognised API! Allowed choices are: {}').format(
            ', '.join(api_modules.keys())
        )
    c_module, attribute = module_def
    for m in structures:
        if attribute is None:
            arg = m
        else:
            arg = getattr(m, attribute)
        atoms, bonds = c_module.counts(arg)
        session.logger.info("%s: %d atoms, %d bonds" % (m, atoms, bonds))
sample_count_desc = CmdDesc(
    optional=[("structures", AtomicStructuresArg),],
    keyword=[("api", StringArg),]
)
