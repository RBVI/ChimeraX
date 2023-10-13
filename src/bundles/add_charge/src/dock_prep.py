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
    from chimerax.core.commands import ListOf, EnumOf
    arg_info = {}
    for name, val in _get_defaults().items():
        if name == "method":
            from . import ChargeMethodArg
            annotation = ChargeMethodArg
        elif name == "standardize_residues":
            # provided directly by Dock Prep
            continue
        else:
            raise ValueError("Don't know how to prepare addcharge arg '%s' for Dock Prep" % name)
        arg_info[name] = annotation
    return arg_info

def run_for_dock_prep(session, state, callback, memo_type, memo_name, structures, kw, *, tool_settings=None):
    from chimerax.dock_prep import handle_memorization, MEMORIZE_USE
    if tool_settings is None and not state['nogui'] and memo_type != MEMORIZE_USE:
        from .tool import AddChargeTool
        AddChargeTool(session, "Add Charges", dock_prep_info={
            'process name': memo_name,
            'structures': structures,
            'callback': lambda used_structures, args1=[session, state, callback, memo_type, memo_name],
                kw=kw, tool_settings=None:
                run_for_dock_prep(*tuple(args1+[used_structures, kw]), tool_settings=tool_settings)
        })
        return
    active_settings = handle_memorization(session, memo_type, memo_name, "add_charge", kw, _get_defaults(),
        tool_settings)
    # tool adds charges directly
    if tool_settings is None:
        from .cmd import cmd_addcharge
        cmd_addcharge(session, structures.residues, **active_settings)
    callback(session, state, structures)

def _get_defaults():
    from .cmd import cmd_addcharge
    import inspect
    defaults = {}
    sig = inspect.signature(cmd_addcharge)
    for name, param in sig.parameters.items():
        if param.kind != param.KEYWORD_ONLY:
            continue
        if name == "standardize_residues":
            # provided directly by Dock Prep
            continue
        defaults[name] = param.default
    return defaults
