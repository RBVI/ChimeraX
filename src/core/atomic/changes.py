# vim: set expandtab shiftwidth=4 softtabstop=4:
def check_for_changes(session):
    """Check for, and propagate ChimeraX atomic data changes.

    This is called once per frame, and whenever otherwise needed.
    """
    ct = session.change_tracker
    if not ct.changed:
        return
    ul = session.update_loop
    ul.block_redraw()
    try:
        changes = Changes(ct.changes)
        ct.clear()
        session.triggers.activate_trigger("atomic changes", changes)
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

    def created_atomic_structures(self):
        return _atomic_structures(self._changes["Structure"].created)

    def created_atoms(self, include_new_structures=True):
        return self._created_objects("Atom", include_new_structures)

    def created_bonds(self, include_new_structures=True):
        return self._created_objects("Bond", include_new_structures)

    def created_chains(self, include_new_structures=True):
        return self._created_objects("Chain", include_new_structures)

    def created_graphs(self):
        """includes atomic structures"""
        return self._changes["Structure"].created

    def created_pseudobond_groups(self):
        return self._changes["PseudobondGroup"].created

    def created_pseudobonds(self):
        return self._changes["Pseudobond"].created

    def created_residues(self, include_new_structures=True):
        return self._created_objects("Residue", include_new_structures)

    graph_reasons = atomic_structure_reasons

    def modified_atomic_structures(self):
        return _atomic_structures(self._changes["Structure"].modified)

    def modified_atoms(self):
        return self._changes["Atom"].modified

    def modified_bonds(self):
        return self._changes["Bond"].modified

    def modified_chains(self):
        return self._changes["Chain"].modified

    def modified_graphs(self):
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

    def num_deleted_pseudobond_groups(self):
        return self._changes["PseudobondGroup"].total_deleted

    def num_deleted_pseudobonds(self):
        return self._changes["Pseudobond"].total_deleted

    def num_deleted_residues(self):
        return self._changes["Residue"].total_deleted

    def num_deleted_structures(self):
        """Not possible to distinguish between AtomicStructures and Structures"""
        return self._changes["Structure"].total_deleted
    num_deleted_graphs = num_deleted_structures

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
        return AtomicStructrures(ass)

    def _created_objects(self, class_name, include_new_structures):
        in_existing = self._changes[class_name].created
        if not include_new_structures:
            return in_existing
        new = self._changes["StructureData"].created.atoms
        if not in_existing:
            return new
        elif not new:
            return in_existing
        from . import concatenate
        return concatenate([new, in_existing])
