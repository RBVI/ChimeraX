# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
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
        if 'selected changed' in global_changes['Atom'].reasons:
            _update_sel_info(session)
            session.selection.trigger_fire_needed = True
        elif global_changes['Atom'].created.num_selected > 0:
            _update_sel_info(session)
            session.selection.trigger_fire_needed = True
        elif global_changes['Structure'].created.atoms.num_selected > 0:
            # For efficiency, atoms in new structures don't show up
            # in changes['Atom'].created, so need this
            _update_sel_info(session)
            session.selection.trigger_fire_needed = True
        elif 'selected changed' in global_changes['Bond'].reasons \
        or 'selected changed' in global_changes['Pseudobond'].reasons:
            session.selection.trigger_fire_needed = True
        from . import get_triggers
        global_triggers = get_triggers()
        global_triggers.activate_trigger("changes", Changes(global_changes))
        for s, s_changes in structure_changes.items():
            s.triggers.activate_trigger("changes", (s, Changes(s_changes)))
        global_triggers.activate_trigger("changes done", None)
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

    def created_atoms(self, include_new_structures=True):
        return self._created_objects("Atom", include_new_structures)

    def created_bonds(self, include_new_structures=True):
        return self._created_objects("Bond", include_new_structures)

    def created_chains(self, include_new_structures=True):
        return self._created_objects("Chain", include_new_structures)

    def created_coordsets(self, include_new_structures=True):
        return self._created_objects("CoordSet", include_new_structures)

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
    num_destroyed_atoms = num_deleted_atoms

    def num_deleted_bonds(self):
        return self._changes["Bond"].total_deleted
    num_destroyed_bonds = num_deleted_bonds

    def num_deleted_chains(self):
        return self._changes["Chain"].total_deleted
    num_destroyed_chains = num_deleted_chains

    def num_deleted_coordsets(self):
        return self._changes["CoordSet"].total_deleted
    num_destroyed_coordsets = num_deleted_coordsets

    def num_deleted_pseudobond_groups(self):
        return self._changes["PseudobondGroup"].total_deleted
    num_destroyed_pseudobond_groups = num_deleted_pseudobond_groups

    def num_deleted_pseudobonds(self):
        return self._changes["Pseudobond"].total_deleted
    num_destroyed_pseudobonds = num_deleted_pseudobonds

    def num_deleted_residues(self):
        return self._changes["Residue"].total_deleted
    num_destroyed_residues = num_deleted_residues

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

def selected_atoms(session=None):
    global _full_sel, _ordered_sel
    check_for_changes(session)
    from .molarray import Atoms
    return _ordered_sel if _ordered_sel is not None else (_full_sel if _full_sel is not None else Atoms())

_ordered_sel = None
_full_sel = None

def _update_sel_info(session):
    from .structure import Structure
    from .molarray import concatenate, Atoms
    global _full_sel, _ordered_sel
    alist = []
    for m in session.models.list(type = Structure):
        m_atoms = m.atoms
        alist.append(m_atoms.filter(m.atoms.selecteds == True))
    _full_sel = concatenate(alist, Atoms)
    if _ordered_sel is None:
        if len(_full_sel) == 1:
            _ordered_sel = _full_sel
    else:
        len_full, len_ordered = len(_full_sel), len(_ordered_sel)
        if len_full < len_ordered - 1 or len_full > len_ordered + 1:
            # definitely not a single-atom change
            if len_full == 1:
                _ordered_sel = _full_sel
            else:
                _ordered_sel = None
        else:
            # might be a single-atom change; check
            len_intersection = len(_full_sel & _ordered_sel)
            # ensure the intersection size is within 1 of both
            if len_full == len_intersection == len_ordered:
                # no change (in atoms)
                pass
            elif len_full <= len_intersection < len_ordered:
                # lost an atom
                lost = _ordered_sel - _full_sel
                # Collection subtraction doesn't preserve order, so need to do this "by hand":
                import numpy
                pointers = numpy.array([ptr for ptr in _ordered_sel.pointers if ptr not in lost.pointers],
                    dtype=numpy.uintp)
                _ordered_sel = Atoms(pointers)
            elif len_full > len_intersection >= len_ordered:
                # added an atom
                added = _full_sel - _ordered_sel
                _ordered_sel = concatenate([_ordered_sel, added])
            else:
                if len_full == 1:
                    _ordered_sel = _full_sel
                else:
                    _ordered_sel = None
