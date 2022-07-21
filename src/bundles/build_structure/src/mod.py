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

class BindError(ValueError):
    pass

def modify_atom(atom, element, num_bonds, *, geometry=None, name=None, connect_back=True,
        color_by_element=True, res_name=None, new_res=False):

    neighbor_Hs = [nb for nb in atom.neighbors if nb.element.number == 1]
    if atom.num_bonds -len(neighbor_Hs) > num_bonds:
        raise ParamError("Atom already has more bonds to heavy atoms than requested.\n"
            "Either delete some of those bonds/atoms or choose a different number of requested bonds.")

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

    for h in neighbor_Hs:
        h.structure.delete_atom(h)

    changed_atoms = [atom]
    if name:
        atom.name = name if name else default_changed_name(atom, element.name)
    atom.element = element
    if color_by_element:
        from chimerax.atomic.colors import element_color
        atom.color = element_color(element.number)

    # if we only have one bond, correct its length
    if atom.num_bonds == 1:
        neighbor = atom.neighbors[0]
        new_length = bond_length(atom, geometry, neighbor.element, a2_info=(neighbor, num_bonds))
        from chimerax.atomic.struct_edit import set_bond_length
        set_bond_length(atom.bonds[0], new_length, move_smaller_side=True)

    if num_bonds == atom.num_bonds:
        handle_res_params(changed_atoms, res_name, new_res)
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
            from chimerax.atom_search import AtomSearchTree
            tree = AtomSearchTree(atom.structure.atoms, sep_val=2.5, scene_coords=False)
            test_atoms = tree.search(atom.coord, 5.0)
    else:
        test_atoms = []

    from chimerax.geometry import distance_squared, angle
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
                if h_name is None:
                    if len(atom.name) < 4:
                        for n in range(h_num, len(positions)+1):
                            possible_name = "H%s%d" % (atom.name[1:], n)
                            if atom.residue.find_atom(possible_name):
                                possible_name = None
                                break
                        else:
                            possible_name = "H%s%d" % (atom.name[1:], h_num)
                    else:
                        possible_name = None
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
    handle_res_params(changed_atoms, res_name, new_res)
    return changed_atoms

def handle_res_params(atoms, res_name, new_res):
    a = atoms[0]
    if res_name == "auto":
        res_name = unknown_res_name(a.residue)
    if res_name:
        if not new_res and a.residue.name == res_name:
            return
    elif new_res:
        res_name = unknown_res_name(a.residue)
    else:
        return
    if new_res:
        chain_id = a.residue.chain_id
        pos = 1
        while a.structure.find_residue(chain_id, pos):
            pos += 1
        r = a.structure.new_residue(res_name, chain_id, pos)
        for a in atoms:
            a.residue.remove_atom(a)
            r.add_atom(a)
    else:
        a.residue.name = res_name

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
        from chimerax.addh import bond_with_H_length
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

def default_changed_name(a, element_name):
    if a.element.name == element_name:
        return a.name
    counter = 1
    while True:
        test_name = "%s%d" % (element_name, counter)
        if len(test_name) > 4:
            test_name = "X"
            break
        if not a.residue.find_atom(test_name):
            break
        counter += 1
    return test_name

def unknown_res_name(res):
    from chimerax.atomic import Residue
    return {
        Residue.PT_NONE: "UNL",
        Residue.PT_AMINO: "UNK",
        Residue.PT_NUCLEIC: "N"
    }[res.polymer_type]

def cn_peptide_bond(c, n, moving, length, dihedral, phi=None, *, log_chain_remapping=False):
    """Make bond between C-terminal carbon in one model and N-terminal nitrogen in another.

       'c' is the carbon and 'n' is the nitrogen.  'moving' should either be the c or the n again,
       depending on which model you want moved.

       If you want a particular value for the newly-established phi angle, provide the 'phi' parameter.

       Returns (c,n) of combined model
    """
    from chimerax.atomic.bond_geom import bond_positions, planar
    from chimerax.atomic.struct_edit import add_atom

    # process C terminus
    if c.element.name != "C":
        raise BindError('C-terminal "carbon" is a %s!' % c.element.name)
    # C-term: find CA
    nbs = c.neighbors
    if len(nbs) > 3:
        raise BindError("More than 3 atoms connected to C-terminal carbon [%s]" % c)
    nb_elements = [a.element.name for a in nbs]
    if nb_elements.count("C") != 1:
        raise BindError("C-terminal carbon not bonded to exactly one carbon")
    cca = nbs[nb_elements.index("C")]
    # C-term: find OXT or equivalent
    added = False
    oxys = [a for a in nbs if a.element.name == "O"]
    if len(oxys) == 0:
        if len(nbs) > 1:
            raise BindError("C-terminal carbon bonded to no oxygens yet bonded to %d other atoms"
                % len(nbs))
        pos = bond_positions(c.coord, planar, 1.0, [cca.coord])[0]
        ac = add_atom("TMP", c.element, c.residue, pos, serial_number=0, bonded_to=c)
        added = True
    elif len(oxys) == 1:
        if len([o for o in oxys if o.name == "OXT"]) == 1:
            ac = oxys[0]
        else:
            if len(nbs) == 2:
                pos = bond_positions(c.coord, planar, 1.0, [cca.coord, oxys[0].coord])[0]
                ac = add_atom("TMP", c.element, c.residue, pos, serial_number=0, bonded_to=c)
                added = True
            elif len(nbs) == 3:
                ac = [a for a in nbs if a not in (c, oxys[0])][0]
                if ac.num_bonds > 1:
                    raise BindError("Unexpected branching atom (%s) connected to C-terminal carbon"
                        %ac )
    else:
        oxts = [o for o in oxys if o.name == "OXT"]
        if len(oxts) == 1:
            ac = oxts[0]
        else:
            ac = oxys[0]

    # process N terminus
    try:
        if n.element.name != "N":
            raise BindError('N-terminal "nitrogen" is a $s!' % n.element.nme)
        # N-term: find CA
        nbs = n.neighbors
        nb_elements = [a.element.name for a in nbs]
        ncs = [nb for i, nb in enumerate(nbs) if nb_elements[i] == "C"]
        if len(ncs) == 1:
            nca = ncs[0]
        else:
            if n.residue.name in ["PRO", "HYP"]:
                if nb_elements.count("C") != 2:
                    raise BindError("Proline N-terminal nitrogen not bonded to exactly two carbons")
                ncas = [nc for nc in ncs if nc.name == "CA"]
                if len(ncas) == 1:
                    nca = ncas[0]
                else:
                    raise BindError("Not exactly one CA bonded to N-terminal nitrogen")
            else:
                raise BindError("Non-proline N-terminal nitrogen not bonded to exactly one carbon")
        if phi is not None:
            # alse need to know the backbone C bonded to nca
            for nca_nb in nca.neighbors:
                if nca_nb.element.name == "C" and nca_nb.is_backbone():
                    n_c = nca_nb
                    break
            else:
                raise BindError("Could not find second C atom for phi angle")
        # N-term: clean the N
        for nb in nbs:
            if nb not in ncs and nb.num_bonds > 1:
                raise BindError("Unexpected branching atom [%s] attached to N terminus" % nb)
        hyds = [a for a in nbs if a.element.number == 1]
        hs = [h for h in hyds if h.name == "H"]
        if hs:
            h = hs[0]
        else:
            h = None
        for nb in nbs:
            if nb not in [n, h] + ncs:
                nb.structure.delete_atom(nb)
        coords = [nc.coord for nc in ncs]
        if h:
            coords.append(h.coord)
        pos = bond_positions(n.coord, planar, 1.0, coords)[0]
        an = add_atom("TMP", n.element, n.residue, pos, serial_number=0, bonded_to=n)
    except Exception:
        if added:
            ac.structure.delete_atom(ac)
        raise

    # call bind
    if moving == c:
        a1, a2 = an, ac
    else:
        a1, a2 = ac, an
    dihed_info = [((cca, c, n, nca), dihedral)]
    # though it might seem simpler to adjust the phi angle after establishing the bond,
    # that may move a chain relative to the other chains in its original model
    if phi is not None:
        dihed_info.append(((c, n, nca, n_c), phi))

    b = bind(a1, a2, length, dihed_info, renumber=an, log_chain_remapping=log_chain_remapping)
    b1, b2 = b.atoms
    if b1.element.name == "C":
        c, n = b1, b2
    else:
        c, n = b2, b1
    c.idatm_type = "Cac"
    n.idatm_type = "Npl"
    nbs = c.neighbors
    if len(nbs) < 3:
        pos = bond_positions(c.coord, planar, 1.23, [a.coord for a in nbs])[0]
        add_atom("O", "O", c.residue, pos, bonded_to=c)
    nbs = n.neighbors
    if hyds and len(nbs) < 3:
        pos = bond_positions(n.coord, planar, 1.01, [a.coord for a in nbs])[0]
        add_atom("H", "H", n.residue, pos, bonded_to=n)
    return (c,n)

def bind(a1, a2, length, dihed_info, *, renumber=None, log_chain_remapping=False):
    """Make bond between two models.

       The models will be combined and the 'a2' model closed.  If the new bond forms a chain,
       the chain ID will be the same as a1's chain ID.

       a1/a2 are atoms in different models, each bonded to exactly one other atom.  In the
       final structure, a1/a2 will be eliminated and their bond partners will be bonded together.

       a2 and atoms in its model will be moved to form the bond.  'length' is the bond length.
       'dihed_info' is a two-tuple of a sequence of four atoms and a dihedral angle that the
       four atoms should form.  dihed_info can be None if insufficent atoms.

       If renumbering of the combined chain should be done, then 'renumber' should be a1 or a2 to
       indicate which side gets renumbered.
    """

    s1, s2 = a1.structure, a2.structure
    if s1 == s2:
        raise BindError("Atoms must be in different models")

    try:
        b1, b2 = a1.neighbors + a2.neighbors
    except ValueError:
        raise BindError("Atoms must be bonded to exactly one atom apiece")

    if renumber:
        renumber_side, static_side = (b1, b2) if renumber == a1 else (b2, b1)

    # move b2 to a1's position
    from chimerax.geometry import translation, angle, cross_product, rotation, distance, dihedral, \
        length as vector_length
    mv = a1.scene_coord - b2.scene_coord
    b2.structure.position = translation(mv) * b2.structure.position

    # rotate to get b1-a1 colinear with b2-a2
    cur_ang = angle(b1.scene_coord, a1.scene_coord, a2.scene_coord)
    rot_axis = cross_product(b1.scene_coord - b2.scene_coord, a2.scene_coord - a1.scene_coord)
    if sum([v * v for v in rot_axis]):
        b2.structure.position = rotation(rot_axis, -cur_ang, center=b2.scene_coord) * b2.structure.position

    # then get the distance correct
    cur_vec = b2.scene_coord - b1.scene_coord
    dv = (length/vector_length(cur_vec) - 1) * cur_vec
    b2.structure.position = translation(dv) * b2.structure.position

    # then dihedral (omega/phi for peptide)
    for atoms, dihed_val in dihed_info:
        p1, p2, p3, p4 = [a.scene_coord for a in atoms]
        if atoms[3].structure != s2:
            p1, p2, p3, p4 = p4, p3, p2, p1
        axis = p3 - p2
        if sum([v * v for v in axis]):
            cur_dihed = dihedral(p1, p2, p3, p4)
            delta = dihed_val - cur_dihed
            b2.structure.position = rotation(axis, delta, center=p3) * b2.structure.position
            if atoms[2].structure == s2:
                p1, p2, p3, p4 = [a.scene_coord for a in atoms]
            else:
                p4, p3, p2, p1 = [a.scene_coord for a in atoms]

    # delete a1/a2
    s1.delete_atom(a1)
    s2.delete_atom(a2)

    # compute needed remapping of chain IDs
    seen_ids = set(s1.residues.unique_chain_ids)
    chain_id_mapping = {}
    chain_ids = sorted(s2.residues.unique_chain_ids)
    for chain_id in chain_ids:
        if chain_id == b2.residue.chain_id:
            # get b1's chain ID
            chain_id_mapping[chain_id] = b1.residue.chain_id
        elif chain_id in seen_ids:
            from chimerax.atomic import next_chain_id
            new_id = next_chain_id(chain_id)
            while new_id in seen_ids or new_id in chain_ids:
                new_id = next_chain_id(new_id)
            if log_chain_remapping:
                s1.session.logger.info("Remapping chain ID '%s' in %s to '%s'" % (chain_id, s2, new_id))
            chain_id_mapping[chain_id] = new_id
            seen_ids.add(new_id)

    # remember where b2 is
    b2_index = s2.atoms.index(b2)

    # renumber part of the new chain if appropriate
    if renumber:
        if renumber_side.residue.chain:
            renumber_side_residues = renumber_side.residue.chain.existing_residues
        else:
            renumber_side_residues = [renumber_side.residue]
        renumber_side_numbers = set([r.number for r in renumber_side_residues])
        if static_side.residue.chain:
            static_side_numbers = set(static_side.residue.chain.existing_residues.numbers)
        else:
            static_side_numbers = set([static_side.residue.number])
        if not static_side_numbers.isdisjoint(renumber_side_numbers):
            # renumbering necessary
            #
            # if lowest renumber_side number is at least one, just add highest static_side number, otherwise
            # add an additional offset to make the lowest number at least 1 more than highest static_side
            low_renumber_side = min(renumber_side_numbers)
            high_static_side = max(static_side_numbers)
            offset = 1 - low_renumber_side if low_renumber_side < 1 else 0
            for r in renumber_side_residues:
                r.number += high_static_side + offset

    # combine
    s1.combine(s2, chain_id_mapping, s1.scene_position)

    # make bond; close s2; return new bond
    from chimerax.atomic.struct_edit import add_bond
    new_b2 = s1.atoms[s1.num_atoms - s2.num_atoms + b2_index]
    b = add_bond(b1, new_b2)
    s1.session.models.close([s2])
    return b
