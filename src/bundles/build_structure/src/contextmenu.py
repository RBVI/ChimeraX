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

class AdjustTorsionMenuEntry(SelectContextMenuAction):
    def label(self, session):
        return "Adjust Torsion"

    def criteria(self, session):
        return self._torsion_bond(session) != None

    def callback(self, session):
        from chimerax.core.commands import run
        tool = run(session, "ui tool show 'Build Structure'")
        tool.show_category("Adjust Torsions")
        try:
            session.bond_rotations.new_rotation(self._torsion_bond(session), one_shot=False)
        except session.bond_rotations.BondRotationError as e:
            from chimerax.core.errors import UserError
            raise UserError(str(e))

    def _torsion_bond(self, session):
        from chimerax.atomic import selected_bonds
        sel_bonds = selected_bonds(session)
        if len(sel_bonds) != 1:
            return None
        bond = sel_bonds[0]
        for end in bond.atoms:
            if len(end.neighbors) < 2:
                return None
        return bond

def add_selection_context_menu_items(session):
    from chimerax.mouse_modes import SelectMouseMode
    SelectMouseMode.register_menu_entry(AdjustTorsionMenuEntry())
