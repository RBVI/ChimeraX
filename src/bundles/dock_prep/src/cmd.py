# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.errors import UserError
from chimerax.core.commands import CmdDesc, register, BoolArg, Or, EmptyArg, EnumOf, DynamicEnum, ListOf
from .prep import MEMORIZE_NONE, MEMORIZE_USE, MEMORIZE_SAVE

# the args directly provided by DockPrep
from .settings import defaults

#NOTES: dock_prep_caller() is the public API and also called via the command.  It assembles a series
#  of steps to execute (by calling dock_prep_steps())
#
#  INTERNAL IMPLEMENTATION NOTES:
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
    for mod_name in ["dock_prep", "addh", "add_charge"]:
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
    # put BoolArg last, so that unparse doesn't convert the others to bool
    info['complete_side_chains'] = Or(EnumOf(('gly', 'ala')),
        DynamicEnum(session.rotamers.library_names), BoolArg)
    from chimerax.atomic.struct_edit import standardizable_residues
    info['standardize_residues'] = ListOf(EnumOf(standardizable_residues))
    return info

def dock_prep_steps(session, memorization, memorize_name, *, from_tool=False, **kw):
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
    for step_name in ["addh", "add_charge"]:
        step_prefix = mod_to_arg_prefix[step_name]
        if active_settings.get(step_prefix[:-1], True):
            step_kw = {}
            for k, v in kw.items():
                if k.startswith(step_prefix):
                    step_kw[k[len(step_prefix):]] = v
            steps.append((step_name, step_kw))
    return steps

def dock_prep_caller(session, structures, *, memorization=MEMORIZE_NONE, memorize_name=None, nogui=None,
        callback=None, _from_tool=False, **kw):
    """Supply 'memorize_name' if you want settings for your workflow to be separately memorizable from
       generic Dock Prep.  It should be a string descriptive of your workflow ("minimization", tool name,
       etc.) since it will also be used in dialog titles.  if 'structures' is None, the user can choose
       the structures to prep (all structures in nogui mode).
    """
    if nogui is None:
        nogui = session.in_script or not session.ui.is_gui
    if (nogui or memorization == MEMORIZE_USE) and structures is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)
    if memorize_name:
        final_memorize_name = "%s dock prep" % memorize_name
        process_name = memorize_name
    else:
        final_memorize_name = process_name = "dock prep"
    if not nogui and not _from_tool:
        def callback2(structures, tool_settings, *, session=session, dpc_kw={
                'memorization': memorization, 'memorize_name': memorize_name,
                'nogui': nogui, 'callback': callback}):
            dpc_kw.update(tool_settings)
            dock_prep_caller(session, structures, _from_tool=True, **dpc_kw)
        dock_prep_info = {
            'structures': structures,
            'process name': process_name,
            'callback': callback2
        }
        from .tool import DockPrepTool
        DockPrepTool(session, dock_prep_info=dock_prep_info)
        return
    state = {
        'steps': dock_prep_steps(session, memorization, final_memorize_name, **kw),
        'memorization': memorization,
        'memorize_name': final_memorize_name,
        'process_name': process_name,
        'callback': callback,
        'nogui': nogui,
    }
    run_steps(session, state, structures)

def run_steps(session, state, structures):
    if structures is not None and not structures:
        # User has closed relevant structures
        return
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
            structures, kw_dict)

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
