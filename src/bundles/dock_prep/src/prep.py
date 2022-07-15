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

MEMORIZE_USE = "use"
MEMORIZE_SAVE = "save"
MEMORIZE_NONE = "none"

def prep(session, state, callback, memorization, memorize_name, structures, keywords, *, tool_settings=None):
    if tool_settings is None and not state['nogui'] and memorization != MEMORIZE_USE:
        # run tool that calls back to this routine with from_tool=True
        #TODO: return function-to-start-tool()
        raise NotImplemented("call tool")

    from .settings import defaults
    active_settings = handle_memorization(session, memorization, memorize_name, "dock_prep", keywords,
        defaults, tool_settings)

    if active_settings['del_solvent']:
        session.logger.info("Deleting solvent")
        for s in structures:
            atoms = s.atoms
            atoms.filter(atoms.structure_categories == "solvent").delete()

    if active_settings['del_ions']:
        session.logger.info("Deleting non-metal-complex ions")
        for s in structures:
            atoms = s.atoms
            ions = atoms.filter(atoms.structure_categories == "ions")
            pbg = s.pbg_map.get(s.PBG_METAL_COORDINATION, None)
            if pbg:
                pb_atoms1, pb_atoms2 = pbg.pseudobonds.atoms
                ions = ions.subtract(pb_atoms1)
                ions = ions.subtract(pb_atoms2)
            ions.delete()

    if active_settings['del_alt_locs']:
        session.logger.info("Deleting non-current alt locs")
        for s in structures:
            s.delete_alt_locs()

    change_std = []
    from chimerax.atomic.struct_edit import standardize_residues, standardizable_residues as std_res
    for r_name in std_res:
        if active_settings['change_' + r_name]:
            change_std.append(r_name)
    if change_std:
        standardize_residues(session, structures.residues, res_types=change_std, verbose=True)
    callback(session, state)

def handle_memorization(session, memorization, memorize_requester, main_settings_name, keywords, defaults,
        tool_settings):
    from .settings import get_settings
    if memorize_requester is None or memorization == MEMORIZE_NONE or memorization == MEMORIZE_SAVE:
        base_settings = defaults
    elif memorization == MEMORIZE_USE:
        settings = get_settings(session, memorize_requester, main_settings_name, defaults)
        base_settings = { param: getattr(settings, param) for param in defaults.keys() }

    if tool_settings is None:
        if memorization == MEMORIZE_USE:
            active_settings = base_settings
        else:
            active_settings = { param: keywords.get(param, defaults[param]) for param in defaults.keys() }
    else:
        active_settings = tool_settings
    if memorize_requester is not None and memorization == MEMORIZE_SAVE:
        settings = get_settings(session, memorize_requester, main_settings_name, defaults)
        for param, val in active_settings.items():
            setattr(settings, param, val)
        settings.save()
    return active_settings
