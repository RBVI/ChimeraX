# vim: set expandtab ts=4 sw=4:

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
from chimerax.core.commands import CmdDesc, register, BoolArg, Or, NoneArg

prep_cmd_param_info = {
    'del_solvent': (BoolArg, True),
}

complete_cmd_param_info = {}
# use a function to avoid importing modules of subsidary commands until needed
def get_param_info():
    if not complete_cmd_param_info:
        for prefix, param_dict in [(None, prep_cmd_param_info)]:
            for arg_name, arg_info in param_dict.items():
                complete_cmd_param_info[arg_name if prefix is None else (prefix + '_' + arg_name)] = arg_info
    return complete_cmd_param_info

def dock_prep_caller(session, structures, memorization, *, callback=None, **kw):
    final_params = {}
    for arg_name, arg_info in get_param_info():
        arg_type, default = arg_info
        final_params[arg_name if prefix is None else (prefix + '_' + arg_name)] = default
    final_params.update(**kw)
    #TODO: log command equivalent; handle memorization; call workhorse function; make callback

def dock_prep_cmd(session, structures,  **kw):
    #TODO: if structures is None:...
    #TODO: call prerequisite command wrappers
    return key

def register_command(logger):
    from chimerax.atomic import AtomicStructuresArg
    cmd_desc = CmdDesc(
        required=[('structures', Or(AtomicStructuresArg, NoneArg))],
        keyword= [(arg_name, arg_info[0]) for arg_name, arg_info in get_param_info()],
        synopsis = 'Prepare structures for computations')
    register('dockprep', cmd_desc, dock_prep_cmd, logger=logger)
