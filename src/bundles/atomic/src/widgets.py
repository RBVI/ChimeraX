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
        kw['trigger_info'] = [ (get_triggers(session), 'changes') ]
    return kw

class ChainListWidget(ItemListWidget):
    def __init__(self, session, **kw):
        super().__init__(**_process_chain_kw(session, **kw))

class ChainMenuButton(ItemMenuButton):
    def __init__(self, session, **kw):
        super().__init__(**_process_chain_kw(session, **kw))
