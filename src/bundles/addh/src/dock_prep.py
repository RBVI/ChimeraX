# vim: set expandtab shiftwidth=4 softtabstop=4:

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

def dock_prep_arg_info(session):
    from chimerax.core.commands import BoolArg, FloatArg
    arg_info = {}
    for name, val in _get_defaults().items():
        if isinstance(val, bool):
            annotation = BoolArg
        elif isinstance(val, float):
            annotation = FloatArg
        else:
            raise ValueError("Don't know how to prepare addh arg '%s' for Dock Prep" % name)
        arg_info[name] = annotation
    return arg_info

def run_for_dock_prep(session, state, callback, memo_type, memo_name, structures, keywords, *,
        tool_settings=None):
    from chimerax.dock_prep import handle_memorization, MEMORIZE_USE
    if tool_settings is None and not state['nogui'] and memo_type != MEMORIZE_USE:
        from .tool import AddHTool
        AddHTool(session, "Add Hydrogens", process_info={
            "process name": memo_name,
            "structures": structures,
            "run command": False,
            "callback": lambda structures, tool_settings, args1=(session, state, callback, memo_type,
                memo_name), keywords=keywords: run_for_dock_prep(*(args1 + (structures, keywords)),
                tool_settings=tool_settings),
        })
        return
    active_settings = handle_memorization(session, memo_type, memo_name, "addh", keywords, _get_defaults(),
        tool_settings)
    from .cmd import cmd_addh
    cmd_addh(session, structures, **active_settings)
    callback(session, state, structures)

def _get_defaults():
    from .cmd import cmd_addh
    import inspect
    defaults = {}
    sig = inspect.signature(cmd_addh)
    for name, param in sig.parameters.items():
        if param.kind != param.KEYWORD_ONLY:
            continue
        defaults[name] = param.default
    return defaults
