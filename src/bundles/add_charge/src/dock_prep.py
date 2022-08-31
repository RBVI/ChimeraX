# vim: set expandtab shiftwidth=4 softtabstop=4:

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

def dock_prep_arg_info(session):
    from chimerax.core.commands import ListOf, EnumOf
    arg_info = {}
    for name, val in _get_defaults().items():
        if name == "method":
            from . import ChargeMethodArg
            annotation = ChargeMethodArg
        elif name == "standardize_residues":
            from chimerax.atomic.struct_edit import standardizable_residues
            annotation = ListOf(EnumOf(standardizable_residues))
        else:
            raise ValueError("Don't know how to prepare addcharge arg '%s' for Dock Prep" % name)
        arg_info[name] = annotation
    return arg_info

def run_for_dock_prep(session, state, callback, memo_type, memo_name, structures, kw, *, tool_settings=None):
    from chimerax.dock_prep import handle_memorization, MEMORIZE_USE
    if tool_settings is None and not state['nogui'] and memorization != MEMORIZE_USE:
        #TODO: run tool and call back to this routine with tool_settings specified
        raise NotImplemented("call tool")
    active_settings = handle_memorization(session, memo_type, memo_name, "add_charge", kw, _get_defaults(),
        tool_settings)
    from .cmd import cmd_addcharge
    cmd_addcharge(session, structures.residues, **active_settings)
    callback(session, state)

def _get_defaults():
    from .cmd import cmd_addcharge
    import inspect
    defaults = {}
    sig = inspect.signature(cmd_addcharge)
    for name, param in sig.parameters.items():
        if param.kind != param.KEYWORD_ONLY:
            continue
        defaults[name] = param.default
    return defaults
