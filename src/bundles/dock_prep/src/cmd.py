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
#  The step info provides a function to call, that should receive the session, state, callback,
#  memorization type (MEMORIZE_NONE/USE/SAVE). memorization name, structures, and a dictionary of
#  step-associated keywords.  The callback should be invoked, with session and state, and will call the
#  next step, or the dock_prep_caller callback if finished.
#
#  Bundles need to provide a public run_for_dock_prep function that can be called to execute the step
#  and a dock_prep_arg_info(session) function that returns a dictionary (arg-name: annotation) of arguments
#  to be added to the dockprep command and that will be provided to the run_for_dock_prep function.
#  These arguments will be given a prefix before being added to the dockprep command keywords, but will not
#  have that prefix when given to the run_for_dock_prep function.
#
#  Bundles are in charge of their own memorization, but can use the handle_memorization() function
#  to do most of the heavy lifting.

mod_to_arg_prefix = {
    "addh": "ah_",
    "add_charge": "ac_",
    "dock_prep": ""
}

def get_param_info(session):
    param_info = {}
    import importlib
    for mod_name in ["dock_prep", "addh"]:
        arg_prefix = mod_to_arg_prefix[mod_name]
        if mod_name != "dock_prep":
            param_info[arg_prefix[:-1]] = BoolArg
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

def dock_prep_steps(session, memorization, memorize_name, **kw):
    if memorization == MEMORIZE_USE:
        from .prep import handle_memorization
        active_settings = handle_memorization(session, memorization, memorize_name, "base", kw,
            defaults, None)
    else:
        active_settings = kw
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
    #TODO: add charge step
    for step_name in ["addh"]:
        if active_settings.get(step_name, True):
            step_kw = {}
            step_prefix = mod_to_arg_prefix[step_name]
            for k, v in kw.items():
                if k.startswith(step_prefix):
                    step_kw[k[len(step_prefix):]] = v
            steps.append((step_name, step_kw))
    return steps

def dock_prep_caller(session, structures, *, memorization=MEMORIZE_NONE, memorize_name=None, nogui=None,
        callback=None, **kw):
    """Supply 'memorize_name' if you want settings for your workflow to be separately memorizable from
       generic Dock Prep.  It should be a string descriptive of your workflow ("minimization", tool name,
       etc.)
    """
    if nogui is None:
        nogui = session.in_script or not session.ui.is_gui
    if memorize_name:
        final_memorize_name = "%s dock prep" % memorize_name
    else:
        final_memorize_name =  "dock prep"
    state = {
        'steps': dock_prep_steps(session, memorization, final_memorize_name, **kw),
        'memorization': memorization,
        'memorize_name': final_memorize_name,
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
        mod_name, kw_dict = steps.pop(0)
        import importlib
        step_mod = importlib.import_module("chimerax." + mod_name)
        step_mod.run_for_dock_prep(session, state, run_steps, state['memorization'], state['memorize_name'],
            state['structures'], kw_dict)

def dock_prep_cmd(session, structures,  *, memorize=MEMORIZE_NONE, **kw):
    if structures is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)
    if not structures:
        raise UserError("No atomic structures open/specified")
    dock_prep_caller(session, structures, memorization=memorize, nogui=True, **kw)

def register_command(logger):
    from chimerax.atomic import AtomicStructuresArg
    cmd_desc = CmdDesc(
        required=[('structures', Or(AtomicStructuresArg, EmptyArg))],
        keyword=[
            ('memorize', EnumOf((MEMORIZE_USE, MEMORIZE_SAVE, MEMORIZE_NONE))),
        ] + list(get_param_info(logger.session).items()),
        synopsis='Prepare structures for computations')
    register('dockprep', cmd_desc, dock_prep_cmd, logger=logger)
