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

class ChargeError(RuntimeError):
    pass

ion_types = {
	"Br": "Br-",
	"Ca": "C0",
	"Cl": "Cl-",
	"Cs": "Cs+",
	"Cu": "CU",
	"F":  "F-",
	"Fe": "FE",
	"I":  "I-",
	"K":  "K+",
	"Li": "Li+",
	"Mg": "MG",
	"Na": "Na+",
	"Rb": "Rb+",
	"Zn": "Zn"
}

def add_nonstandard_res_charges(session, residues, net_charge, method="am1-bcc", *,
            gaff_type=True, status=None):
        """Add Antechamber charges to non-standard residue
        
           'residues' is a list of residues of the same type.  The first
           residue in the list will be used as an exemplar for the whole
           type for purposes of charge determination, but charges will be
           added to all residues in the list.

           'net_charge' is the net charge of the residue type.

           'method' is either 'am1-bcc' or 'gasteiger'

           'gaff_type' is a boolean that determines whether GAFF
           atom types are assigned to atoms in non-standard residues

           'status' is where status messages go (e.g. replyobj.status)

           Hydrogens need to be present.
        """
        r0 = residues[0]
        session.logger.info("Assigning partial charges to residue %s (net charge %+d) with %s method"
            % (r0.name, net_charge, method))
        # special case for single-atom residues...
        if r0.num_atoms == 1:
            for r in residues:
                a = r.atoms[0]
                a.charge = net_charge
                session.change_tracker.add_modified(a, "charge changed")
                if gaff_type:
                    if a.element.name in ion_types:
                        a.gaff_type = ion_types[a.element.name]
                    else:
                        session.logger.info("Could not determine GAFF type for atom %s" % a)
            return

        # detect tautomers by checking bonds
        varieties = {}
        for r in residues:
            atom_map = {}
            ordered_atoms = list(r.atoms)
            ordered_atoms.sort(key=lambda a: (a.name, a.coord_index))
            for i, a in enumerate(ordered_atoms):
                atom_map[a] = i
            bonds =[]
            for a in r.atoms:
                i1 = atom_map[a]
                for nb in a.neighbors:
                    i2 = atom_map.get(nb, None)
                    if i2 is None or i1 < i2:
                        bonds.append((i1, i2))
                    else:
                        bonds.append((i2, i1))
            bonds.sort()
            varieties.setdefault(tuple(bonds), []).append(r)
        if len(varieties) > 1:
            session.logger.info("%d tautomers of %s; charging separately" % (len(varieties), r0.name))
        for tautomer_residues in varieties.values():
            _nonstd_charge(session, tautomer_residues, net_charge, method, gaff_type, status)

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
    # use same ordering of atoms as they had in input, to improve consistency of antechamber charges
    r_atoms = sorted(r.atoms, key=lambda a: a.coord_index)
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
    total_net_charge = net_charge + estimate_net_charge(extras)

    from contextlib import contextmanager
    @contextmanager
    def managed_structure(s):
        try:
            yield s
        finally:
            s.delete()

    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir, managed_structure(s) as s:
        import os, os.path
        ante_in = os.path.join(temp_dir, "ante.in.mol2")
        from chimerax.mol2 import write_mol2
        write_mol2(session, ante_in, models=[s], status=status)

        ante_out = os.path.join(temp_dir, "ante.out.mol2")
        from chimerax.amber_info import amber_bin, amber_home
        command = [amber_bin + "/antechamber"]
        if method.lower().startswith("am1"):
            mth = "bcc"
            command.extend(["-ek", "qm_theory='AM1',"])
        elif method.lower().startswith("gas"):
            mth = "gas"
        else:
            raise ValueError("Unknown charge method: %s" % method)

        command.extend([
            "-i", ante_in,
            "-fi", "mol2",
            "-o", ante_out,
            "-fo", "mol2",
            "-c", mth,
            "-nc", str(total_net_charge),
            "-j", "5",
            "-s", "2",
            "-dr", "n"])
        if status:
            status("Running ANTECHAMBER for residue %s" % r.name)
        from subprocess import Popen, STDOUT, PIPE
        # For some reason in Windows, if shell==False then antechamber cannot run bondtype via system()
        session.logger.info("Running ANTECHAMBER command: %s" % " ".join(command))
        os.environ['AMBERHOME'] = amber_home
        ante_messages = Popen(command, stdin=PIPE, stdout=PIPE, stderr=STDOUT, cwd=temp_dir, bufsize=1,
            encoding="utf8").stdout
        while True:
            line = ante_messages.readline()
            if not line:
                break
            session.logger.status("(%s) %s" % (r.name, line.rstrip()))
            session.logger.info("(%s) <code>%s</code>" % (r.name, line.rstrip()), is_html=True)
        ante_failure_msg = "Failure running ANTECHAMBER for residue%s\nCheck reply log for details" % r.name
        if not os.path.exists(ante_out):
            raise ChargeError(ante_failure_msg)
        if status:
            status("Reading ANTECHAMBER output for residue %s" % r.name)
        try:
            mols, status_message = session.open_command.open_data(ante_out)
        except Exception as e:
            raise IOError("Problem reading ANTECHAMBER output file: %s" % str(e))
        if not mols:
            raise RuntimeError("No molecules in ANTECHAMBER output for residue %s" % r.name)
        mol = mols[0]
        if mol.num_atoms != s.num_atoms:
            raise RuntimeError("Wrong number of atoms (%d, should be %d) in ANTECHAMBER output for residue"
                " %s" % (mol.num_atoms, s.num_atoms, r.anme))
        charged_atoms = mol.atoms
        if status:
            status("Assigning charges for residue %s" % r.name)
        # put charges in template
        template_atoms = list(s.atoms)
        # can't rely on order...
        template_atoms.sort(key=lambda a: a.serial_number)
        non_zero = False
        added_charge_sum = 0.0
        _total_charge = 0.0
        for ta, ca in zip(template_atoms, charged_atoms):
            _total_charge += ca.charge
            if ta in extras:
                added_charge_sum += ca.charge
                continue
            if ca.charge:
                non_zero = True
        # it is okay for O2 and similar moieties to be all zero charge...
        if not non_zero and len(set([a.element for a in charged_atoms])) > 1:
            raise ChargeError(ante_failure_msg)

        # adjust charges to compensate for added atoms...
        adjustment = (added_charge_sum - (total_net_charge - net_charge)) / (
            len(template_atoms) - len(extras))
        for ta, ca in zip(template_atoms, charged_atoms):
            if ta in extras:
                continue
            ta.charge = ca.charge + adjustment
            if gaff_type:
                ta.gaff_type = ca.mol2_type
        # map template charges onto first residue
        assigned = set()
        for fa, ta in atom_map.items():
            if ta in extras:
                continue
            fa.charge = ta.charge
            if gaff_type:
                fa.gaff_type = ta.gaff_type
            assigned.add(fa)
        # map charges onto remaining residues
        r_atoms = sorted(r.atoms)
        for rr in residues[1:]:
            rr_atoms = sorted(rr.atoms)
            for fa, ra in zip(r_atoms, rr_atoms):
                ra.charge = fa.charge
                if gaff_type:
                    ra.gaff_type = fa.gaff_type
                assigned.add(ra)
        session.change_tracker.add_modified(assigned, "charge changed")
        if gaff_type:
            session.change_tracker.add_modified(assigned, "gaff_type changed")
        if status:
            status("Charges for residue %s determined" % r.name)
        session.logger.info("Charges for residue %s determined" % r.name)

