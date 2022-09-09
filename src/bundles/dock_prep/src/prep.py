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
        # run tool that calls back to this routine with tool_settings specified
        #TODO: return function-to-start-tool()
        raise NotImplemented("call tool")

    from .settings import defaults
    active_settings = handle_memorization(session, memorization, memorize_name, "base", keywords,
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

    std_res = active_settings['standardize_residues']
    if std_res:
        from chimerax.atomic.struct_edit import standardize_residues
        standardize_residues(session, structures.residues, res_types=std_res, verbose=True)
 
    if active_settings['complete_side_chains']:
        targets = []
        for s in structures:
            for r in s.residues:
                if r.polymer_type != r.PT_AMINO:
                    continue
                if r.name != r.standard_aa_name:
                    continue
                if not r.is_missing_heavy_template_atoms():
                    continue
                for bb_name in r.aa_min_backbone_names:
                    if not r.find_atom(bb_name):
                        session.logger.warning("%s is missing heavy backbone atoms" % r)
                        break
                else:
                    targets.append(r)
        if targets:
            session.logger.info("Filling out missing side chains")
            style = active_settings['complete_side_chains']
            if style is True:
                # use default rotamer lib
                style = session.rotamers.default_command_library_name
            from chimerax.swap_res import swap_aa
            if style in ('gly', 'ala'):
                for r in targets:
                    res_type = 'gly' if not r.find_atom('CB') else style
                    swap_aa(session, [r], res_type)
            else:
                swap_aa(session, targets, "same", rot_lib=style)
    callback(session, state)

def handle_memorization(session, memorization, memorize_requester, main_settings_name, keywords, defaults,
        tool_settings):
    from .settings import get_settings
    if memorization == MEMORIZE_NONE or memorization == MEMORIZE_SAVE:
        base_settings = defaults
    else: # USE
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
