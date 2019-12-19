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
#from chimerax.atomic.struct_edit import add_atom
#from chimerax.atomic.colors import element_color
#from chimerax.atomic.bond_geom import linear

class ParamError(ValueError):
    pass

def modify_atom(atom, element, num_bonds, *, geometry=None, name=None, connect_back=True,
        color_by_element=True, res_name=None, res_new_only=False):

    if atom.num_bonds > num_bonds:
        raise ParamError("Atom already has more bonds that requested.\n"
            "Either delete some bonds or choose a different number of requested bonds.")
    if geometry is None:
        if num_bonds < 2:
            # geometry irrelevant
            geometry = 4
        else:
            from chimerax.atomic.idatm import type_info
            try:
                geometry = type_info[atom.idatm_type].geometry
            except KeyError:
                geometry = 4
    if num_bonds > geometry:
        raise ParamError("Requested number of bonds more than the coordination geometry can support.")

    changed_atoms = [atom]

    # if we only have one bond, correct its length
    if atom.num_bonds == 1:
        neighbor = atom.neighbors[0]
        new_length = bond_length(atom, geometry, neighbor.element, a2_info=(neighbor, num_bonds))
        set_bond_length(atom.bonds[0], new_length, move_smaller_side=True)

    if num_bonds == atom.num_bonds:
        return changed_atoms

    #TODO
    return changed_atoms

element_radius = {}
for i in range(Element.NUM_SUPPORTED_ELEMENTS):
	element = Element.get_element(i)
	element_radius[element] = 0.985 * Element.bond_radius(element)
element_radius[Element.get_element('C')] = 0.7622
element_radius[Element.get_element('H')] = 0.1869
element_radius[Element.get_element('N')] = 0.6854
element_radius[Element.get_element('O')] = 0.6454
element_radius[Element.get_element('P')] = 0.9527
element_radius[Element.get_element('S')] = 1.0428

def bond_length(a1, geom, e2, *, a2_info=None):
    if e2.number == 1:
        from chimera.atomic.addh import bond_with_H_length
        return bond_with_H_length(a1, geom)
    e1 = a1.element
    base_len = element_radius[e1] + element_radius[e2]
    if geom == 1:
        return base_len
    # a2_info has to be supplied for non-hydrogens
    neighbor, num_bonds = a2_info
    from chimerax.atomic.idatm import type_info
    try:
        nb_geom = type_info[neighbor.idatm_type].geometry
    except KeyError:
        return base_len
    if nb_geom == 1:
        return base_len
    if num_bonds == 1 or neighbor.num_bonds == 1:
        # putative double bond
        return 0.88 * base_len
    elif geom == 4 or nb_geom == 4:
        return base_len
    return 0.92 * base_len

def set_bond_length(bond, bond_length, *, move_smaller_side=True, status=None):
    bond.structure.idatm_valid = False
    if bond.rings(cross_residue=True):
        if status:
            status("Bond is involved in ring/cycle.\nMoved bonded atoms (only) equally.", color="blue")
        mid = sum([a.coord for a in bond.atoms]) / 2
        factor = bond_length / bond.length
        for a in bond.atoms:
            a.coord = (a.coord - mid) * factor + mid
        return

    smaller = bond.smaller_side
    bigger = bond.other_atom(smaller)
    if move_smaller_side:
        moving = smaller
        fixed = bigger
    else:
        moving = bigger
        fixed = smaller
    mp = moving.coord
    fp = fixed.coord
    v1 = mp - fp
    from numpy.linalg import norm
    v1_len = norm(v1)
    v1 *= bond_length / v1_len
    delta = v1 - (mp - fp)
    moving_atoms = bond.side_atoms(moving)
    moving_atoms.coords = moving_atoms.coords + delta

