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

from chimerax.mouse_modes import SelectContextMenuAction

class MutateMenuEntry(SelectContextMenuAction):
    def label(self, session):
        return "Mutate Residue"

    def criteria(self, session):
        from chimerax.atomic import selected_residues
        return len([r for r in selected_residues(session) if r.polymer_type == r.PT_AMINO]) == 1

    def callback(self, session):
        from chimerax.core.commands import run
        tool = run(session, "ui tool show Rotamers")

def add_selection_context_menu_items(session):
    from chimerax.mouse_modes import SelectMouseMode
    SelectMouseMode.register_menu_entry(MutateMenuEntry())
