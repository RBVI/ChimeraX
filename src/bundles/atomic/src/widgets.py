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

from chimerax.ui.widgets import ModelListWidget, ModelMenuButton
from chimerax.atomic import Structure, AtomicStructure

class StructureListWidget(ModelListWidget):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=Structure, **kw)

class AtomicStructureListWidget(ModelListWidget):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=AtomicStructure, **kw)

class StructureMenuButton(ModelMenuButton):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=Structure, **kw)

class AtomicStructureMenuButton(ModelMenuButton):
    def __init__(self, session, **kw):
        super().__init__(session, class_filter=AtomicStructure, **kw)
