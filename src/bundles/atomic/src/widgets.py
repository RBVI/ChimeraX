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

class ChainListWidget(ItemListWidget):
    def __init__(self, session, *, group_identical=False, **kw):
        self._session = session
        self._group_identical = group_identical and kw.get('selection_mode', None) != 'single'
        processed_kw = _process_chain_kw(session, **kw)
        self._raw_list_func = processed_kw['list_func']
        self._requested_item_text_func = processed_kw.get('item_text_func', None)
        processed_kw['list_func'] = self._list_func
        processed_kw['item_text_func'] = self._item_text_func
        super().__init__(**processed_kw)

    @property
    def group_identical(self):
        return self._group_identical

    @group_identical.setter
    def group_identical(self, group):
        if group != self._group_identical:
            self._group_identical = group
            prev_sel = self.get_value()
            self.refresh()
            if group:
                # was individual chains
                next_sel = set()
                for chain in prev_sel:
                    for chains in self.value_map.keys():
                        if chain in chains:
                            next_sel.add(tuple(chains))
                            break
            else:
                # was groups of chains
                next_sel = []
                for chains in prev_sel:
                    next_sel.extend(chains)
            self.set_value(next_sel)

    @property
    def value(self):
        if self.group_identical:
            return [chain for chains in self.get_value() for chain in chains]
        return self.get_value()

    @value.setter
    def value(self, val):
        if self.group_identical:
            from chimerax.core.errors import LimitationError
            raise LimitationError("Cannot set grouped Chain list")
        self.set_value(val)

    @property
    def grouped_value(self):
        if self.group_identical:
            return self.get_value()
        chains = self.get_value()
        if not chains:
            return chains
        return [[chain] for chain in chains]

    def _list_func(self):
        simple_list = self._raw_list_func()
        if not self.group_identical:
            return simple_list
        groups = {}
        for chain in simple_list:
            groups.setdefault(chain.characters, []).append(chain)
        grouped_list = list(groups.values())
        grouped_list.sort(key=lambda x: x[0])
        # since the returned values will be used as dictionary keys, need to return tuples
        return [tuple(chains) for chains in grouped_list]

    def _item_text_func(self, item):
        if self._requested_item_text_func:
            if self.group_identical:
                return "; ".join([self._requested_item_text_func(chain) for chain in item])
            return self._requested_item_text_func(item)
        if self.group_identical:
            chains = item
        else:
            chains = [item]
        cur_structure = None
        specs = ""
        descriptions = []
        for chain in chains:
            if chain.structure == cur_structure:
                specs += ',' + chain.chain_id
            else:
                specs += chain.string()
                cur_structure = chain.structure
            if chain.description and chain.description not in descriptions:
                descriptions.append(chain.description)
        if descriptions:
            return specs + ": " + "; ".join(descriptions)
        return specs

class ChainMenuButton(ItemMenuButton):
    def __init__(self, session, **kw):
        super().__init__(**_process_chain_kw(session, **kw))

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
        from chimerax.core.models import ADD_MODELS
        kw['trigger_info'] = [ (get_triggers(), 'changes'), (session.triggers, ADD_MODELS) ]
    return kw

class ResidueListWidget(ItemListWidget):
    def __init__(self, session, **kw):
        super().__init__(**_process_residue_kw(session, **kw))

class ResidueMenuButton(ItemMenuButton):
    def __init__(self, session, **kw):
        super().__init__(**_process_residue_kw(session, **kw))

def _process_residue_kw(session, list_func=None, trigger_info=None, **kw):
    if list_func is None:
        from . import all_residues
        kw['list_func'] = lambda ses=session, f=all_residues: f(ses)
    if trigger_info is None:
        from .triggers import get_triggers
        from chimerax.core.models import ADD_MODELS
        kw['trigger_info'] = [ (get_triggers(), 'changes'), (session.triggers, ADD_MODELS) ]
    return kw

def make_elements_menu(parent, *, _session=None, _parent_menus=None):
    '''keyword args for internal use only (adding Elements menu under main Select menu)'''
    if _session and _parent_menus:
        add_submenu = _session.ui.main_window.add_select_submenu
        elements_menu = add_submenu(_parent_menus[:-1], _parent_menus[-1])
    else:
        from Qt.QtWidgets import QMenu
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
