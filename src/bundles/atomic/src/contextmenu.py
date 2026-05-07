# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.mouse_modes import SelectContextMenuAction

# Add hide and delete atoms/bonds/pseudobonds to double-click selection context menu
class HideObjectsMenuEntry(SelectContextMenuAction):
    def __init__(self, type):
        self.type = type
    def label(self, ses):
        n = _num_selected_objects(ses, self.type)
        return ("Hide %s" if n == 1 else "Hide %ss") % self.type.capitalize()
    def criteria(self, ses):
        n = _num_selected_objects(ses, self.type)
        return n > 0
    def callback(self, ses):
        from chimerax.core.commands import run
        run(ses, 'hide sel %ss' % self.type)

class DeleteObjectsMenuEntry(SelectContextMenuAction):
    dangerous = True
    def __init__(self, type):
        self.type = type
    def label(self, ses):
        n = _num_selected_objects(ses, self.type)
        return ("Delete %s" if n == 1 else "Delete %ss") % self.type.capitalize()
    def criteria(self, ses):
        n = _num_selected_objects(ses, self.type)
        return n > 0
    def callback(self, ses):
        n = _num_selected_objects(ses, self.type)
        from chimerax.ui.ask import ask
        if ask(ses, "Really delete %s %s(s)" % (n, self.type),
               title="Deletion Request") == "no":
            return
        from chimerax.core.commands import run
        run(ses, 'delete %ss sel' % self.type)

class SelectInterveningResiduesMenuEntry(SelectContextMenuAction):
    def label(self, ses):
        return "Select Intervening Residues"
    def criteria(self, ses):
        from chimerax.atomic import selected_residues
        sel_res = selected_residues(ses)
        if len(sel_res) != 2:
            return False
        r1, r2 = sel_res
        if r1.chain != r2.chain or r1.chain is None:
            return False
        existing = r1.chain.existing_residues
        i1 = existing.index(r1)
        i2 = existing.index(r2)
        return abs(i1-i2) > 1
    def callback(self, ses):
        from chimerax.core.commands import run
        run(ses, "ks ri")

def _num_selected_objects(ses, type):
    from chimerax.atomic import selected_atoms, selected_bonds, selected_pseudobonds
    sel_objects = {'atom':selected_atoms,
                   'bond':selected_bonds,
                   'pseudobond':selected_pseudobonds}[type]
    return len(sel_objects(ses))

def add_selection_context_menu_items(session):
    from chimerax.mouse_modes import SelectMouseMode
    SelectMouseMode.register_menu_entry(SelectInterveningResiduesMenuEntry())
    for type in ('atom', 'bond', 'pseudobond'):
        SelectMouseMode.register_menu_entry(HideObjectsMenuEntry(type))
        SelectMouseMode.register_menu_entry(DeleteObjectsMenuEntry(type))
