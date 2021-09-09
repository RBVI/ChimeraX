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
from chimerax.core.commands import CmdDesc, register, BoolArg, Or, NoneArg, EnumArg

MEMORIZE_USE = "use"
MEMORIZE_SAVE = "save"
MEMORIZE_NONE = "none"

prep_cmd_param_info = {
    'del_solvent': (BoolArg, True),
}

complete_cmd_param_info = {}
# use a function to avoid importing modules of subsidary commands until needed
_cmd_sources = None
def cmd_sources():
    global _cmd_sources
    if _cmd_sources is None:
        from .prep import prep
        _cmd_sources = [(None, prep, prep_cmd_param_info)]
    return _cmd_sources

def get_param_info():
    if not complete_cmd_param_info:
        for prefix, func, param_dict in cmd_sources():
            for arg_name, arg_info in param_dict.items():
                complete_cmd_param_info[arg_name
                    if prefix is None else (prefix + '_' + arg_name)] = arg_info
    return complete_cmd_param_info

def dock_prep_caller(session, structures, memorization, *, callback=None, **kw):
    for prefix, func, param_dict in cmd_sources():
        func_kw = {}
        if memorization == MEMORIZE_NONE:
            settings = None
        else:
            settings = get_settings(session, prefix) #TODO
        #TODO: the scheme below just isn't right -- later interfaces need to be able to
        # show GUIs and have the chosen settings memorized.  Maybe somehow use 'yield'?
        for arg_name, arg_info in param_dict.items():
            arg_type, default = arg_info
            attr_name = arg_name if prefix is None else prefix + '_' _ arg_name
            if memorization == MEMORIZE_NONE:
                arg_val = kw.get(attr_name, default)
            elif memorization == MEMORIZE_USE:
                arg_val = kw.get(attr_name, getattr(settings, arg_name))
            else:
                arg_val = kw.get(attr_name, default)
                setattr(settings, arg_name, arg_val)
            func_kw[arg_name] = arg_val
        func(session, structures, **func_kw)
    #TODO: log command equivalents; handle memorization; call workhorse function; make callback

def dock_prep_cmd(session, structures,  *, memorize=MEMORIZE_NONE, **kw):
    if structures is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)
    if not structures:
        raise UserError("No atomic structures open/specified")
    dock_prep_caller(session, structures, memorize, **kw)

class MemorizationArg(EnumArg):
    values = (MEMORIZE_USE, MEMORIZE_SAVE, MEMORIZE_NONE)

def register_command(logger):
    from chimerax.atomic import AtomicStructuresArg
    cmd_desc = CmdDesc(
        required=[('structures', Or(AtomicStructuresArg, NoneArg))],
        keyword=[('memorize', MemorizationArg)] + [(arg_name, arg_info[0])
            for arg_name, arg_info in get_param_info()],
        synopsis='Prepare structures for computations')
    register('dockprep', cmd_desc, dock_prep_cmd, logger=logger)
