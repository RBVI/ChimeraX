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
from chimerax.core.commands import CmdDesc, register, BoolArg, Or, EmptyArg, EnumOf, DynamicEnum
from .prep import MEMORIZE_NONE, MEMORIZE_USE, MEMORIZE_SAVE

# the args directly provided by DockPrep
from .settings import defaults

#NOTES: dock_prep_caller() is the public API and also called via the command.  It assembles a series
#  of steps to execute (by calling dock_prep_steps())
#
#  The step info provides a function to call, that should receive the session, iterator, callback,
#  memorization info, and a dictionary of step-associated keywords.  The callback should be invoked,
#  with session and iterator, and will call the next step, or the dock_prep_caller callback if finished.
#
#  Bundles need to provide a public run_for_dock_prep function that can be called to execute the step
#  and a dock_prep_arg_info(session) function that returns a dictionary (arg-name: annotation) of arguments
#  to be added to the dockprep command and that will be provided to the run_for_dock_prep function.
#  These arguments will be given a prefix before being added to the dockprep command keywords, but will not
#  have that prefix when given to the run_for_dock_prep function.
#
#  Bundles are in charge of their own memorization

def get_param_info(session):
    param_info = {}
    import importlib
    for mod_name, arg_prefix in [("dock_prep", "")]:
        full_mod_name = "chimerax." + mod_name
        mod = importlib.import_module(full_mod_name)
        for arg_name, arg_annotation in mod.dock_prep_arg_info(session).items():
            param_info[arg_prefix + arg_name] = arg_annotation
    return param_info

def dock_prep_arg_info(session):
    info = { setting: BoolArg for setting in defaults }
    info['complete_side_chains'] = Or(BoolArg, EnumOf(('gly', 'ala')),
        DynamicEnum(session.rotamers.library_names))
    return info

def dock_prep_steps(add_hydrogens=True, add_charges=True, **kw):
    steps = []
    kw_dict = {}
    dp_step_needed = False
    for param, default in defaults.items():
        val = kw.get(param, default)
        if val:
            dp_step_needed = True
        kw_dict[param] = val
    if dp_step_needed:
        steps.append(("dock_prep", kw_dict))
    #TODO: external steps
    return steps

def dock_prep_caller(session, structures, *, memorization=MEMORIZE_NONE, memorize_name=None, nogui=None,
        callback=None, **kw):
    if nogui is None:
        nogui = session.in_script or not session.ui.is_gui
    state = {
        'steps': dock_prep_steps(**kw),
        'memorization': memorization,
        'memorize_name': memorize_name,
        'callback': callback,
        'structures': structures,
        'nogui': nogui,
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
            state['structures'], kw_dict)

def dock_prep_cmd(session, structures,  *, memorize=MEMORIZE_NONE, **kw):
    if structures is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)
    if not structures:
        raise UserError("No atomic structures open/specified")
    dock_prep_caller(session, structures, memorization=memorize, memorize_name="dock prep", nogui=True, **kw)

def register_command(logger):
    from chimerax.atomic import AtomicStructuresArg
    cmd_desc = CmdDesc(
        required=[('structures', Or(AtomicStructuresArg, EmptyArg))],
        keyword=[
            ('memorize', EnumOf((MEMORIZE_USE, MEMORIZE_SAVE, MEMORIZE_NONE))),
        ] + list(get_param_info(logger.session).items()),
        synopsis='Prepare structures for computations')
    register('dockprep', cmd_desc, dock_prep_cmd, logger=logger)
