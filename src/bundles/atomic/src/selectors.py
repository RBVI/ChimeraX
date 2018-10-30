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

def register_selectors(logger):
    # Selectors
    #
    # NOTE: also need to be listed in bundle_info.xml.in
    from chimerax.core.commands import register_selector as reg
    from . import Element, Atom
    # Since IDATM has types in conflict with element symbols (e.g. 'H'), register
    # the types first so that they get overriden by the symbols
    for idatm, info in Atom.idatm_info_map.items():
        reg(idatm, lambda ses, models, results, sym=idatm: _idatm_selector(sym, models, results), logger, desc=info.description)
    for i in range(1, Element.NUM_SUPPORTED_ELEMENTS):
        e = Element.get_element(i)
        reg(e.name, lambda ses, models, results, sym=e.name: _element_selector(sym, models, results), logger, desc="%s (element)" % e.name)



def _element_selector(symbol, models, results):
    from chimerax.atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = m.atoms.filter(m.atoms.element_names == symbol)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms, bonds=True)

def _idatm_selector(symbol, models, results):
    from chimerax.atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = m.atoms.filter(m.atoms.idatm_types == symbol)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms, bonds=True)

_chains_menu_needs_update = False
_chains_menu_name = "&Chains"
_residues_menu_needs_update = False
_residues_menu_name = "&Residues"


def add_select_menu_items(session):
    mw = session.ui.main_window

    parent_menus = [_chains_menu_name]
    select_chains_menu = mw.add_select_submenu(parent_menus[:-1], parent_menus[-1])
    select_chains_menu.aboutToShow.connect(lambda ses=session: _update_select_chains_menu(ses))
    select_chains_menu.setToolTipsVisible(True)
    from . import get_triggers
    atom_triggers = get_triggers(session)
    atom_triggers.add_handler("changes", _check_chains_update_status)

    parent_menus = ["Che&mistry", "&element"]
    elements_menu = mw.add_select_submenu(parent_menus[:-1], parent_menus[-1])
    elements_menu.triggered.connect(lambda act, mw=mw: mw.select_by_mode(act.text()))
    from PyQt5.QtWidgets import QAction
    for element_name in ["C", "H", "N", "O", "P", "S"]:
        elements_menu.addAction(QAction(element_name, mw))

    from . import Element
    known_elements = [nm for nm in Element.names if len(nm) < 3]
    known_elements.sort()
    from math import sqrt
    num_menus = int(sqrt(len(known_elements)) + 0.5)
    incr = len(known_elements) / num_menus
    start_index = 0
    other_menu = mw.add_select_submenu(parent_menus, "other")
    for i in range(num_menus):
        if i < num_menus-1:
            end_index = int((i+1) * incr + 0.5)
        else:
            end_index = len(known_elements) - 1
        submenu = mw.add_select_submenu(parent_menus + ["other"], "%s-%s"
            % (known_elements[start_index], known_elements[end_index]))
        for en in known_elements[start_index:end_index+1]:
            action = QAction(en, mw)
            submenu.addAction(action)
        start_index = end_index + 1

    parent_menus = ["Che&mistry", "&IDATM type"]
    idatm_menu = mw.add_select_submenu(parent_menus[:-1], parent_menus[-1])
    idatm_menu.triggered.connect(lambda act, mw=mw: mw.select_by_mode(act.text()))
    from . import Atom
    for idatm in Atom.idatm_info_map.keys():
        idatm_menu.addAction(QAction(idatm, mw))

    parent_menus = [_residues_menu_name]
    select_residues_menu = mw.add_select_submenu(parent_menus[:-1], parent_menus[-1])
    select_residues_menu.aboutToShow.connect(lambda ses=session: _update_select_residues_menu(ses))
    select_residues_menu.setToolTipsVisible(True)
    from . import get_triggers
    atom_triggers = get_triggers(session)
    atom_triggers.add_handler("changes", _check_residues_update_status)

def _update_select_chains_menu(session):
    global _chains_menu_needs_update
    if not _chains_menu_needs_update:
        return
    mw = session.ui.main_window
    select_chains_menu = mw.add_select_submenu([], _chains_menu_name)
    from . import AtomicStructures, all_atomic_structures
    structures = AtomicStructures(all_atomic_structures(session))
    chain_info = {}
    for chain in structures.chains:
        key = chain.description if chain.description else chain.chain_id
        chain_info.setdefault(key, []).append(chain)
    chain_keys = list(chain_info.keys())
    chain_keys.sort()
    select_chains_menu.clear()
    def final_description(description):
        if len(description) < 110:
            return False, description
        return True, description[:50] + "..." + description[-50:]
    from PyQt5.QtWidgets import QAction
    for chain_key in chain_keys:
        chains = chain_info[chain_key]
        if len(chains) > 1:
            submenu = select_chains_menu.addMenu(chain_key)
            sep = submenu.addSeparator()
            chains.sort(key=lambda c: (c.structure.id, c.chain_id))
            collective_spec = ""
            for chain in chains:
                if len(structures) > 1:
                    if chain.description:
                        shortened, final = final_description(chain.description)
                        label = "[%s] chain %s" % (chain.structure, chain.chain_id)
                    else:
                        label = "[%s]" % chain.structure
                        shortened = False
                else:
                    # ...must be multiple identical descriptions...
                    label = "chain %s" % chain.chain_id
                    shortened = False
                spec = chain.string(style="command")
                action = mw.add_menu_selector(submenu, label, spec)
                collective_spec += spec
                if shortened:
                    submenu.setToolTipsVisible(True)
                    action.setToolTip(chain.description)
            mw.add_menu_selector(submenu, "all", collective_spec, insertion_point=sep)
        else:
            chain = chains[0]
            chain_id_text = str(chain)
            slash_index = chain_id_text.rfind('/')
            chain_id_text = chain_id_text[:slash_index] + " chain " + chain_id_text[slash_index+1:]
            if chain.description:
                shortened, final = final_description(chain.description)
                label = "%s (%s)" % (final, chain_id_text)
            else:
                label = chain_id_text
                shortened = False
            spec = chain.string(style="command")
            action = mw.add_menu_selector(select_chains_menu, label, spec)
            if shortened:
                select_chains_menu.setToolTipsVisible(True)
                action.setToolTip(chain.description)

    _chains_menu_needs_update = False

def _check_chains_update_status(trig_name, changes):
    global _chains_menu_needs_update
    if not _chains_menu_needs_update:
        _chains_menu_needs_update = changes.created_chains() or changes.num_deleted_chains() > 0

def _update_select_residues_menu(session):
    global _residues_menu_needs_update
    if not _residues_menu_needs_update:
        return
    mw = session.ui.main_window
    select_residues_menu = mw.add_select_submenu([], _residues_menu_name)
    select_residues_menu.clear()
    from . import AtomicStructures, all_atomic_structures
    structures = AtomicStructures(all_atomic_structures(session))
    nonstandard = set()
    amino = set()
    nucleic = set()
    from . import Sequence
    for r in structures.residues:
        if Sequence.rname3to1(r.name) == 'X':
            nonstandard.add(r.name)
        elif Sequence.amino3to1(r.name) == 'X':
            nucleic.add(r.name)
        else:
            amino.add(r.name)
    prev_entries = False
    for cat_name, members in (("all nonstandard", nonstandard), ("standard nucleic acids", nucleic),
            ("standard amino acids", amino)):
        if not members:
            continue
        collective_spec = ""
        if prev_entries:
            select_residues_menu.addSeparator()
        # remember first action so that collective spec can be inserted
        first_entry = None
        res_names = list(members)
        res_names.sort()
        for rn in res_names:
            spec = ':' + rn
            collective_spec += spec
            action = mw.add_menu_selector(select_residues_menu, rn, spec)
            if first_entry is None:
                first_entry = action
        if len(res_names) > 1:
            mw.add_menu_selector(select_residues_menu, cat_name, collective_spec, insertion_point=first_entry)
        prev_entries = True

    _residues_menu_needs_update = False

def _check_residues_update_status(trig_name, changes):
    global _residues_menu_needs_update
    if not _residues_menu_needs_update:
        _residues_menu_needs_update = changes.created_residues() or changes.num_deleted_residues() > 0

