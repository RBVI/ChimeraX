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

def estimate_net_charge(atoms):
    charge_info = {
        'Cac': 2,
        'N3+': 2,
        'N2+': 2,
        'N1+': 2,
        'Ntr': 4,
        'Ng+': _ng_charge,
        'N2': _n2_charge,
        'Oar+': 2,
        'O2-': -2,
        'O3-': -2,
        'S3-': -2,
        'S3+': 2,
        'Sac': 4,
        'Son': 4,
        'Sxd': 2,
        'Pac': 2,
        'Pox': 2,
        'P3+': 2,
    }
    charge_total = 0 # really totals twice the charge...
    rings = set()
    subs = {}
    for a in atoms:
        if len(a.bonds) == 0:
            if a.element.is_alkali_metal:
                charge_total += 2
                continue
            if a.element.is_metal:
                charge_total += 4
                continue
            if a.element.is_halogen:
                charge_total -= 2
                continue
        from chimerax.atomic.idatm import type_info
        try:
            subs[a] = type_info[a.idatm_type].substituents
        except KeyError:
            pass
        else:
            # missing/additional protons
            charge_total += 2 * (a.num_bonds - subs[a])
        a_rings = a.rings()
        rings.update([ar for ar in a_rings if ar.aromatic])
        if a.idatm_type == "C2" and not a_rings:
            for nb in a.neighbors:
                nb_rings = nb.rings()
                if not nb_rings or not nb_rings[0].aromatic:
                    break
            else:
                # all ring neighbors in aromatic rings
                charge_total += 2
        try:
            info = charge_info[a.idatm_type]
        except KeyError:
            continue
        if type(info) == int:
            charge_total += info
        else:
            charge_total += info(a)
    for ring in rings:
        # since we are only handling aromatic rings, any non-ring bonds are presumably single bond
        # (or matched aromatic bonds)
        electrons = 0
        for a in ring.atoms:
            if a in subs:
                electrons += a.element.number + subs[a] - 2
            else:
                electrons += a.element.number + a.num_bonds - 2
            if a.idatm_type[-1] in "+-":
                electrons += 1
        if electrons % 2 == 1:
            charge_total += 2
    return charge_total // 2


def _get_aname(base, known_names):
    anum = 1
    while True:
        name = "%s%d" % (base, anum)
        if name not in known_names:
            known_names.add(name)
            break
        anum += 1
    return name

def _methylate(na, n, atom_names):
    added = []
    from chimerax.atom import Element
    from chimerax.atomic.struct_edit import add_atom
    nn = add_atom(_get_aname("C", atom_names), Element.get_element("C"), na.residue, n.coord)
    added.append(nn)
    na.structure.new_bond(na, nn)
    from chimerax.atomic.bond_geom import bond_positions
    for pos in bond_positions(nn.coord, 4, 1.1, [na.coord]):
        nh = add_atom(_get_aname("H", atom_names), Element.get_element("H"), na.residue, pos)
        added.append(nh)
        na.structure.new_bond(nn, nh)
    return added

def _ng_charge(atom):
    c2 = None
    for nb in atom.neighbors:
        if nb.idatm_type == "C2":
            if c2:
                return 0
            c2 = nb
    if not c2:
        return 1
    ng_pluses = [a for a in c2.neighbors if a.idatm_type == "Ng+"]
    if len(ng_pluses) == 2:
        return 1
    heavys = 0
    for nb in atom.neighbors:
        if nb.element.number > 1:
            heavys += 1
    if heavys > 1:
        return 0
    countable = 0
    for ngp in ng_pluses:
        heavys = 0
        for nb in ngp.neighbors:
            if nb.element.number > 1:
                heavys += 1
        if heavys < 2:
            countable += 1
    if countable > 1:
        return 1
    return 2

def _n2_charge(atom):
    # needed in order to get nitrite ions correct
    for nb in atom.neighbors:
        if nb.idatm_type != "O2-":
            return 0
    return 2

def _nonstd_charge(session, residues, net_charge, method, gaff_type, status):
    r = residues[0]
    if status:
        status("Copying residue %s" % r.name)

    # create a fake Structure that we can write to a Mol2 file
    from chimerax.atomic import AtomicStructure
    s = AtomicStructure(session)
    s.name = r.name

    # write out the residue's atoms first, since those are the ones we will be caring about
    nr = s.new_residue(r.name, ' ', 1)
    atom_map = {}
    atom_names = set()
    r_atoms = list(ratoms)
    # use same ordering of atoms as they had in input, to improve consistency of antechamber charges
    r_atoms.sort(key=lambda a: a.coord_index)
    from chimerax.atomic.struct_edit import add_atom
    for a in r_atoms:
        atom_map[a] = add_atom(a.name, a.element, nr, a.coord)
        atom_names.add(a.name)

    # add the intraresidue bonds and remember the interresidue ones
    nearby = set()
    for a in r_atoms:
        na = atom_map[a]
        for nb in a.neighbors:
            if nb.residue != r:
                nearby.add(nb)
                continue
            nnb = atom_map[nb]
            if nnb not in na.neighbors:
                s.new_bond(na, nnb)
    from chimerax.atomic.idatm import type_info
    extras = set()
    while nearby:
        nb = nearby.pop()
        na = add_atom(_get_aname(nb.element.name, atom_names), nb.element, nr, nb.coord)
        atom_map[nb] = na
        for nbnb in nb.neighbors:
            if nbnb in atom_map:
                s.new_bond(na, atom_map[nbnb])
            else:
                try:
                    ti = type_info[nbnb.idatm_type]
                except KeyError:
                    fc = 0
                    geom = 4
                else:
                    fc = estimate_net_charge([nbnb])
                    geom = ti.geometry
                if fc or geom != 4:
                    nearby.add(nbnb)
                else:
                    extras.update(_methylate(na, nbnb, atom_names))
    total_net_charge = net_charge = estimate_net_charge(extras)

    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        import os.path
        ante_in = os.path.join(temp_dir, "ante.in.mol2")
        from chimera.mol2 import write_mol2
        write_mol2([s], ante_in, status=status)

        #TODO: initially, try to run Chimera's antechamber using hardcoded paths
