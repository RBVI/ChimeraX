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
from chimerax.core.commands import CmdDesc, register, BoolArg, Or, EmptyArg, EnumOf, SaveFileNameArg

MEMORIZE_USE = "use"
MEMORIZE_SAVE = "save"
MEMORIZE_NONE = "none"

#NOTES: An iterator is used to go through the requested Dock Prep steps, yielding the next step.
#  The step info provides a function to call, that should receive the session, iterator, callback,
#  memorization info, and step-associated keywords.  The callback should be invoked, with session
#  and iterator, and will call the next step, or the dock_prep_caller callback if finished.
#
#  Bundles need to provide a function that can be called to get keyword info (so that the dockprep
#  command can register (prefixed) keywords.
#
#  Bundles are in charge of their own memorization

def get_param_info():
    param_info = {}
    import importlib
    for mod_name, arg_prefix in [("dock_prep", "")]:
        full_mod_name = "chimerax." + mod_name
        mod = importlib.import_module(full_mod_name)
        for arg_name, arg_annotation in mod.dock_prep_arg_info.items():
            param_info[arg_prefix + arg_name] = arg_annotation
    return param_info

def dock_prep_steps(del_solvent=True, del_ions=False, del_alt_locs=True, change_MSE=True, change_UMP=True,
        change_UMS=True, change_CSL=True, complete_side_chains=True, add_hydrogens=True, add_charges=True):
    steps = []
    if del_solvent: # ... or del_ions.. etc.
        kw_dict = {}
        steps.append(("dock_prep", kw_dict))
        if del_solvent:
            kw_dict['del_solvent'] = True
    #TODO
    return steps

def dock_prep_caller(session, structures, *, memorization=MEMORIZE_NONE, memorize_name=None,
        callback=None, **kw):
    state = {
        'steps': dock_prep_steps(**kw),
        'memorization': memorization,
        'memorize_name': memorize_name,
        'callback': callback,
        'structures': structures,
    }
    run_steps(session, state)

def run_steps(session, state):
    steps = state['steps']
    if not steps:
        callback = state['callback']
        if callback:
            callback()
    else:
        step_memorize_name = "dock_prep"
        memorize_name = state['memorize_name']
        if memorize_name:
            step_memorize_name += '_' + memorize_name
        mod_name, kw_dict = steps.pop(0)
        import importlib
        step_mod = importlib.import_module("chimerax." + mod_name)
        step_mod.run_for_dock_prep(session, state, run_steps, state['memorization'], step_memorize_name,
            state['structures'], **kw_dict)

def dock_prep_cmd(session, structures,  *, memorize=MEMORIZE_NONE, mol2=None, **kw):
    if structures is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)
    if not structures:
        raise UserError("No atomic structures open/specified")
    dock_prep_caller(session, structures, memorization=memorize, **kw)

def register_command(logger):
    from chimerax.atomic import AtomicStructuresArg
    cmd_desc = CmdDesc(
        required=[('structures', Or(AtomicStructuresArg, EmptyArg))],
        keyword=[('memorize', EnumOf((MEMORIZE_USE, MEMORIZE_SAVE, MEMORIZE_NONE)))]
            + list(get_param_info().items()),
        synopsis='Prepare structures for computations')
    register('dockprep', cmd_desc, dock_prep_cmd, logger=logger)
