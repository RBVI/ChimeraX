# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.ui.widgets import ModelListWidget, ModelMenuButton, ItemListWidget, ItemMenuButton
from chimerax.atomic import Structure, AtomicStructure

class StructureListWidget(ModelListWidget):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=Structure, **kw)

class StructureMenuButton(ModelMenuButton):
    def __init__(self, session, **kw):
        if 'no_value_button_text' not in kw:
            kw['no_value_button_text'] = "No structure chosen"
        super().__init__(session, class_filter=Structure, **kw)

class AtomicStructureListWidget(ModelListWidget):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=AtomicStructure, **kw)

class AtomicStructureMenuButton(ModelMenuButton):
    def __init__(self, session, **kw):
        if 'no_value_button_text' not in kw:
            kw['no_value_button_text'] = "No structure chosen"
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
    def __init__(self, session, no_value_button_text="No chain chosen", **kw):
        super().__init__(no_value_button_text=no_value_button_text,  **_process_chain_kw(session, **kw))

def _process_chain_kw(session, list_func=None, trigger_info=None, **kw):
    if list_func is None:
        def chain_list(ses=session):
            chains = []
            for m in ses.models:
                if isinstance(m, Structure):
                    chains.extend(m.chains)
            return chains
        kw['list_func'] = chain_list
    else:
        kw['list_func'] = list_func
    if trigger_info is None:
        from .triggers import get_triggers
        from chimerax.core.models import ADD_MODELS
        kw['trigger_info'] = [ (get_triggers(), 'changes'), (session.triggers, ADD_MODELS) ]
    else:
        kw['trigger_info'] = trigger_info
    return kw

class ResidueListWidget(ItemListWidget):
    def __init__(self, session, **kw):
        super().__init__(**_process_residue_kw(session, **kw))

class ResidueMenuButton(ItemMenuButton):
    def __init__(self, session, no_value_button_text="No residue chosen", **kw):
        super().__init__(no_value_button_text=no_value_button_text, **_process_residue_kw(session, **kw))

def _process_residue_kw(session, list_func=None, trigger_info=None, **kw):
    if list_func is None:
        from . import all_residues
        kw['list_func'] = lambda ses=session, f=all_residues: f(ses)
    else:
        kw['list_func'] = list_func
    if trigger_info is None:
        from .triggers import get_triggers
        from chimerax.core.models import ADD_MODELS
        kw['trigger_info'] = [ (get_triggers(), 'changes'), (session.triggers, ADD_MODELS) ]
    else:
        kw['trigger_info'] = trigger_info
    return kw

class AtomListWidget(ItemListWidget):
    def __init__(self, session, **kw):
        super().__init__(**_process_atom_kw(session, **kw))

class AtomMenuButton(ItemMenuButton):
    def __init__(self, session, no_value_button_text="No atom chosen", **kw):
        super().__init__(no_value_button_text=no_value_button_text, **_process_atom_kw(session, **kw))

def _process_atom_kw(session, list_func=None, trigger_info=None, **kw):
    if list_func is None:
        from . import all_atoms
        kw['list_func'] = lambda ses=session, f=all_atoms: f(ses)
    else:
        kw['list_func'] = list_func
    if trigger_info is None:
        from .triggers import get_triggers
        from chimerax.core.models import ADD_MODELS
        kw['trigger_info'] = [ (get_triggers(), 'changes'), (session.triggers, ADD_MODELS) ]
    else:
        kw['trigger_info'] = trigger_info
    return kw

# Defined in selectors so the Select menu can be built without importing the
# list and menu widgets above.
from .selectors import make_elements_menu
