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

def add_select_menu_items(session):
    mw = session.ui.main_window
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
