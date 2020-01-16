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
    if name:
        atom.name = name
    atom.element = element
    if color_by_element:
        from chimerax.atomic.colors import element_color
        atom.color = element_color(element.number)

    # if we only have one bond, correct its length
    if atom.num_bonds == 1:
        neighbor = atom.neighbors[0]
        new_length = bond_length(atom, geometry, neighbor.element, a2_info=(neighbor, num_bonds))
        set_bond_length(atom.bonds[0], new_length, move_smaller_side=True)

    if num_bonds == atom.num_bonds:
        handle_res_params(changed_atoms, res_name, res_new_only)
        return changed_atoms

    from chimerax.atomic.bond_geom import bond_positions
    coplanar = None
    if geometry == 3 and atom.num_bonds == 1:
        nb = atom.neighbors[0]
        if nb.num_bonds == 3:
            coplanar = [nn.coord for nn in nb.neighbors if nn != atom]

    away = None
    if geometry == 4 and atom.num_bonds == 1:
        nb = atom.neighbors[0]
        if nb.num_bonds > 1:
            nn = nb.neighbors[0]
            if nn == atom:
                nn = nb.neighbors[1]
            away = nn.coord
    hydrogen = Element.get_element('H')
    positions = bond_positions(atom.coord, geometry, bond_length(atom, geometry, hydrogen),
        [nb.coord for nb in atom.neighbors], coplanar=coplanar, away=away)[:num_bonds - atom.num_bonds]

    if connect_back:
        if atom.structure.num_atoms < 100:
            test_atoms = list(atom.structure.atoms)
        else:
            from chimerax.atomic.search import AtomSearchTree
            tree = AtomSearchTree(atom.structure.atoms, sep_val=2.5, scene_coords=False)
            test_atoms = tree.search(atom.coord, 5.0)
    else:
        test_atoms = []

    from chimerax.core.geometry import distance_squared, angle
    from chimerax.atomic.struct_edit import add_bond, gen_atom_name, add_atom
    h_num = 1
    for pos in positions:
        for ta in test_atoms:
            if ta == atom:
                continue
            test_len = bond_length(ta, 1, hydrogen)
            test_len2 = test_len * test_len
            if distance_squared(ta.coord, pos) < test_len2:
                bonder = ta
                # possibly knock off a hydrogen to accomodate the bond...
                for bn in bonder.neighbors:
                    if bn.element.number > 1:
                        continue
                    if angle(atom.coord - ta.coord, bn.coord - ta.coord) > 45.0:
                        continue
                    try:
                        test_atoms.remove(bn)
                    except ValueError:
                        pass
                    atom.structure.delete_atom(bn)
                    break
                add_bond(atom, bonder)
                break
        else:
            bonded_Hs = [h for h in atom.neighbors if h.element.number == 1]
            if bonded_Hs:
                if len(bonded_Hs) == 1:
                    bonded_name = bonded_Hs[0].name
                    if bonded_name[-1].isdigit():
                        name_base = bonded_name[:-1]
                    else:
                        name_base = "H%s" % atom.name
                else:
                    use_default = False
                    from os.path import commonprefix
                    name_base = commonprefix([h.name for h in bonded_Hs])
                    if not name_base:
                        use_default = True
                    else:
                        for h in bonded_Hs:
                            if not h.name[len(name_base)+1:].isdigit():
                                use_default = True
                                break
                    if use_default:
                        name_base = "H%s" % atom.name[1:]
                n = 1
                while atom.residue.find_atom("%s%d" % (name_base, n)):
                    n += 1
                possible_name = "%s%d" % (name_base, n)
            else:
                h_name = None
                if len(positions) == 1:
                    possible_name = "H" + atom.name[1:]
                    if not atom.residue.find_atom(possible_name):
                        h_name = possible_name
                if h_name is None and len(atom.name) < 4:
                    for n in range(h_num, len(positions)+1):
                        possible_name = "H%s%d" % (atom.name[1:], n)
                        if atom.residue.find_atom(possible_name):
                            possible_name = None
                            break
                    else:
                        possible_name = "H%s%d" % (atom.name[1:], h_num)
            h_name = possible_name \
                if possible_name is not None and len(possible_name) <= 4 \
                else gen_atom_name(hydrogen, atom.residue)
            bonder = add_atom(h_name, hydrogen, atom.residue, pos, bonded_to=atom)
            changed_atoms.append(bonder)
            if color_by_element:
                # element_color previously imported if color_by_element is True
                bonder.color = element_color(1)
            else:
                bonder.color = atom.color
            h_num += 1
    handle_res_params(changed_atoms, res_name, res_new_only)
    return changed_atoms

def handle_res_params(atoms, res_name, res_new_only):
    if not res_name:
        return
    if res_new_only:
        a = atoms[0]
        chain_id = a.residue.chain_id
        pos = 1
        while a.structure.find_residue(chain_id, pos):
            pos += 1
        r = a.structure.new_residue(res_name, chain_id, pos)
        for a in atoms:
            a.residue.remove_atom(a)
            r.add_atom(a)
    else:
        atoms[0].residue.name = res_name

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
        from chimerax.atomic.addh import bond_with_H_length
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

