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

from chimerax.ui.widgets import ModelListWidget, ModelMenuButton, ItemListWidget, ItemMenuButton
from chimerax.atomic import Structure, AtomicStructure

class StructureListWidget(ModelListWidget):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=Structure, **kw)

class StructureMenuButton(ModelMenuButton):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=Structure, **kw)

class AtomicStructureListWidget(ModelListWidget):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=AtomicStructure, **kw)

class AtomicStructureMenuButton(ModelMenuButton):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=AtomicStructure, **kw)

def _process_chain_kw(session, list_func=None, trigger_info=None, **kw):
    if list_func is None:
        def chain_list(ses=session):
            chains = []
            for m in ses.models:
                if isinstance(m, Structure):
                    chains.extend(m.chains)
            return chains
        kw['list_func'] = chain_list
    if trigger_info is None:
        from .triggers import get_triggers
        kw['trigger_info'] = [ (get_triggers(), 'changes') ]
    return kw

class ChainListWidget(ItemListWidget):
    def __init__(self, session, **kw):
        super().__init__(**_process_chain_kw(session, **kw))

class ChainMenuButton(ItemMenuButton):
    def __init__(self, session, **kw):
        super().__init__(**_process_chain_kw(session, **kw))

def make_elements_menu(parent, *, _session=None, _parent_menus=None):
    '''keyword args for internal use only (adding Elements menu under main Select menu)'''
    if _session and _parent_menus:
        add_submenu = _session.ui.main_window.add_select_submenu
        elements_menu = add_submenu(_parent_menus[:-1], _parent_menus[-1])
    else:
        from PyQt5.QtWidgets import QMenu
        elements_menu = QMenu(parent)

    for element_name in ["C", "H", "N", "O", "P", "S"]:
        elements_menu.addAction(element_name)

    from . import Element
    known_elements = [nm for nm in Element.names if len(nm) < 3]
    known_elements.sort()
    from math import sqrt
    num_menus = int(sqrt(len(known_elements)) + 0.5)
    incr = len(known_elements) / num_menus
    start_index = 0
    if _session and _parent_menus:
        other_menu = add_submenu(_parent_menus, "Other")
    else:
        other_menu = elements_menu.addMenu("Other")
    for i in range(num_menus):
        if i < num_menus-1:
            end_index = int((i+1) * incr + 0.5)
        else:
            end_index = len(known_elements) - 1
        range_string = "%s-%s" % (known_elements[start_index], known_elements[end_index])
        if _session and _parent_menus:
            submenu = add_submenu(_parent_menus + ["Other"], range_string)
        else:
            submenu = other_menu.addMenu(range_string)
        for en in known_elements[start_index:end_index+1]:
            submenu.addAction(en)
        start_index = end_index + 1
    return elements_menu
