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

class MutateMenuEntry(SelectContextMenuAction):
    def label(self, session):
        return "Mutate Residue"

    def criteria(self, session):
        from chimerax.atomic import selected_residues
        num_mutatable = 0
        for r in selected_residues(session):
            if r.polymer_type == r.PT_AMINO:
                if r.find_atom('CA') and r.find_atom('N') and r.find_atom('C'):
                    num_mutatable += 1
                    if num_mutatable > 1:
                        return False
        return num_mutatable == 1

    def callback(self, session):
        from chimerax.core.commands import run
        tool = run(session, "ui tool show Rotamers")

def add_selection_context_menu_items(session):
    from chimerax.mouse_modes import SelectMouseMode
    SelectMouseMode.register_menu_entry(MutateMenuEntry())
