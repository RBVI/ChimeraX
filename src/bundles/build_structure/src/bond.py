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

from chimerax.atomic import Element

class CreateBondError(ValueError):
    pass

def create_bonds(atoms, *, reasonable=True, bond_length_tolerance=0.4):
    """Create bonds among the given Atoms Collection.  If 'reasonable' is True, then only bonds whose
       lengths are chemically reasonable will be created.  Otherwise, all possible bonds will be created.

       For adding connectivity to an entire molecule, Structure.connect_structure() is better/faster.
    """

    if len(atoms) < 2:
        raise CreateBondError("Must specify two or more atoms")

    from chimerax.geometry import distance
    from chimerax.atomic import Element
    def is_reasonable(a1, a2, tolerance=bond_length_tolerance):
        return distance(a1.coord, a2.coord) - Element.bond_length(a1.element, a2.element) < tolerance

    from chimerax.atomic.struct_edit import add_bond
    if len(atoms) == 2:
        a1, a2 = atoms
        if a1 in a2.neighbors:
            return []
        if reasonable and not is_reasonable(a1, a2):
            return []
        return [add_bond(a1, a2)]

    from chimerax.atom_search import AtomSearchTree
    created_bonds = []
    for struct, struct_atoms in atoms.by_structure:
        if reasonable:
            search_tree = AtomSearchTree(struct_atoms, scene_coords=False)
            for a1 in struct_atoms:
                hits = search_tree.search(a1, 4.0)
                for a2 in hits:
                    if a1 == a2 or a1 in a2.neighbors:
                        continue
                    if is_reasonable(a1, a2):
                        created_bonds.append(add_bond(a1, a2))
        else:
            for i, a1 in enumerate(struct_atoms):
                for a2 in struct_atoms[i+1:]:
                    if a1 not in a2.neighbors:
                        created_bonds.append(add_bond(a1, a2))
    return created_bonds
