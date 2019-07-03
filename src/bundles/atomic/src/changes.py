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

def check_for_changes(session):
    """Check for, and propagate ChimeraX atomic data changes.

    This is called once per frame, and whenever otherwise needed.
    """
    ct = getattr(session, 'change_tracker', None)
    if not ct or not ct.changed:
        return
    ul = session.update_loop
    ul.block_redraw()
    try:
        global_changes, structure_changes = ct.changes
        ct.clear()
        from . import get_triggers
        get_triggers().activate_trigger("changes", Changes(global_changes))
        for s, s_changes in structure_changes.items():
            s.triggers.activate_trigger("changes", (s, Changes(s_changes)))
    finally:
        ul.unblock_redraw()

class Changes:
    """Present a function-call API to the changes data, rather than direct access"""

    def __init__(self, changes):
        self._changes = changes

    def atom_reasons(self):
        return self._changes["Atom"].reasons

    def atomic_structure_reasons(self):
        return self._changes["Structure"].reasons

    def bond_reasons(self):
        return self._changes["Bond"].reasons

    def chain_reasons(self):
        return self._changes["Chain"].reasons

    def coordset_reasons(self):
        return self._changes["CoordSet"].reasons

    def created_atomic_structures(self):
        return self._atomic_structures(self._changes["Structure"].created)

    def created_atoms(self, include_new_structures=True):
        return self._created_objects("Atom", include_new_structures)

    def created_bonds(self, include_new_structures=True):
        return self._created_objects("Bond", include_new_structures)

    def created_chains(self, include_new_structures=True):
        return self._created_objects("Chain", include_new_structures)

    def created_coordsets(self, include_new_structures=True):
        return self._created_objects("CoordSet", include_new_structures)

    def created_structures(self):
        """includes atomic structures"""
        return self._changes["Structure"].created

    def created_pseudobond_groups(self):
        return self._changes["PseudobondGroup"].created

    def created_pseudobonds(self):
        return self._changes["Pseudobond"].created

    def created_residues(self, include_new_structures=True):
        return self._created_objects("Residue", include_new_structures)

    structure_reasons = atomic_structure_reasons

    def modified_atomic_structures(self):
        return self._atomic_structures(self._changes["Structure"].modified)

    def modified_atoms(self):
        return self._changes["Atom"].modified

    def modified_bonds(self):
        return self._changes["Bond"].modified

    def modified_chains(self):
        return self._changes["Chain"].modified

    def modified_coordsets(self):
        return self._changes["CoordSet"].modified

    def modified_structures(self):
        return self._changes["Structure"].modified

    def modified_pseudobond_groups(self):
        return self._changes["PseudobondGroup"].modified

    def modified_pseudobonds(self):
        return self._changes["Pseudobond"].modified

    def modified_residues(self):
        return self._changes["Residue"].modified

    def num_deleted_atoms(self):
        return self._changes["Atom"].total_deleted

    def num_deleted_bonds(self):
        return self._changes["Bond"].total_deleted

    def num_deleted_chains(self):
        return self._changes["Chain"].total_deleted

    def num_deleted_coordsets(self):
        return self._changes["CoordSet"].total_deleted

    def num_deleted_pseudobond_groups(self):
        return self._changes["PseudobondGroup"].total_deleted

    def num_deleted_pseudobonds(self):
        return self._changes["Pseudobond"].total_deleted

    def num_deleted_residues(self):
        return self._changes["Residue"].total_deleted

    def num_deleted_structures(self):
        """Not possible to distinguish between AtomicStructures and Structures"""
        return self._changes["Structure"].total_deleted

    def pseudobond_reasons(self):
        return self._changes["Pseudobond"].reasons

    def pseudobond_group_reasons(self):
        return self._changes["PseudobondGroup"].reasons

    def residue_reasons(self):
        return self._changes["Residue"].reasons

    def _atomic_structures(self, collection):
        from . import AtomicStructure, AtomicStructures
        ass = []
        for g in collection:
            if isinstance(g, AtomicStructure):
                ass.append(g)
        from . import AtomicStructures
        return AtomicStructures(ass)

    def _created_objects(self, class_name, include_new_structures):
        in_existing = self._changes[class_name].created
        if not include_new_structures:
            return in_existing
        from . import concatenate
        attr_name = class_name.lower() + 's'
        new = concatenate([getattr(s, attr_name) for s in self._changes["Structure"].created],
            in_existing.objects_class)
        if not in_existing:
            return new
        elif not new:
            return in_existing
        return concatenate([new, in_existing])
