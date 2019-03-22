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

from chimerax.ui.widgets.model_chooser import ModelItems, ModelListWidgetBase
from chimerax.atomic import Structure, AtomicStructure

class StructureItems(ModelItems):
    column_title = "Structure"
    class_filter = Structure

class AtomicStructureItems(StructureItems):
    class_filter = AtomicStructure

class StructureListWidget(ModelListWidgetBase, StructureItems):
    autoselect_default = "all"

    def __init__(self, session, **kw):
        self.session = session
        super().__init__(**kw)

class AtomicStructureListWidget(ModelListWidgetBase, AtomicStructureItems):
    autoselect_default = "all"

    def __init__(self, session, **kw):
        self.session = session
        super().__init__(**kw)
