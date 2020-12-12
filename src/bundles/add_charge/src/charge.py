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

def add_standard_charges(session, models=None, *, status=None, phosphorylation=None, query_user=True):
    """add AMBER charges to well-known residues

       'models' restricts the addition to the specified models

       'status' is where status messages go (e.g. session.logger.status)

       'phosphorylation' controls whether chain-terminal nucleic acids will have their phosphorylation
       state changed to correspond to AMBER charge files (3' phosphorylated, 5' not).  A value of None
       means that the user will be queried if possible [treated as True if not possible], though if
       'query_user' is False, the user will not be queried.

       The return value is a 2-tuple of dictionaries:  the first of which details the residues that did
       not receive charges [key: residue type, value: list of residues], and the second lists remaining
       uncharged atoms [key: (residue type, atom name), value: list of atoms]

       Hydrogens need to be present.
    """
    import os.path
    #TODO: C-terminal definitions need to cover OT2 as well as OXT
    attr_file = os.path.join(os.path.split(__file__)[0], "amber_name.defattr")
    if status:
        status("Defining AMBER residue types")
    from chimerax.std_commands.defattr import defattr
    defattr(session, attr_file, restriction=models)

    if models is None:
        from chimerax.atomic import all_atomic_structures
        structures = all_atomic_structures(session)
    else:
        structures = models

    #TODO logic needs to be adjusted to put 5' phosphates in non-standard list (particularly if phosphorylation == False)
    if phosphorylation != False:
        if status:
            status("Checking phosphorylation of chain-terminal nucleic acids")
        deletes = []
        for s in structures:
            for r in s.residues:
                amber_name = getattr(r, 'amber_name', "UNK")
                if len(amber_name) != 2 or amber_name[0] not in 'DR' or amber_name[1] not in 'ACGTU' \
                or not r.find_atom('P'):
                    continue
                p = r.find_atom('P')
                for nb in p.neighbors:
                    if nb.residue != r:
                        break
                else:
                    # trailing phosphate
                    deletes.append(r)
        if deletes:
            if phosphorylation is None:
                if query_user and not session.in_script:
                    from chimerax.ui import ask
                    phosphorylation = ask(session, "Delete 5' terminal phosphates from nucleic acid chains?",
                            info="The AMBER charge set lacks parameters for terminal phosphates, and if"
                            " retained, such residues will be treated as non-standard",
                            title="Delete 5' phosphates?") == "yes"
                else:
                    phosphorylation = True
            if phophorylation:
                _phosphorylate(session, status, deletes)
    if status:
        status("Adding standard charges")
    uncharged_res_types = {}
    uncharged_atoms = {}
    uncharged_residues = set()
    #TODO: create data.py
    from .data import heavy_charge_type_data, hyd_charge_type_data
    from chimerax.atomic import Atom
    Atom.register_attr(session, "gaff_type", "add charge", attr_type=str)
    from chimerax.amber_info import amber_version
    session.logger.info("Using Amber %s recommended default charges and atom types for standard residues"
        % amber_version)
    for s in structures:
        for r in s.residues:
            if not hasattr(r, "amber_name"):
                uncharged_residues.add(r)
                uncharged_res_types.setdefault(r.name, []).append(r)
        modified_atoms = []
        # since hydrogen names are so unreliable, look them up based on the atom they are bonded to
        # rather than their own name
        hydrogen_data = []
        for a in s.atoms:
            if a.residue.name in uncharged_res_types:
                continue
            modified_atoms.add(a)
            if a.element.number == 1:
                if a.num_bonds != 1:
                    raise ChargeError("Hydrogen %s not bonded to exactly one other atom" % a)
                hydrogen_data.append((a, a.neighbors[0]))
                continue
            try:
                a.charge, a.gaff_type = heavy_charge_type_data[(a.residue.amber_name, a.name.lower())]
            except KeyError:
                raise ChargeError("Nonstandard name for heavy atom %s" % a)
        for h, heavy in hydrogen_data:
            if h.residue != heavy.residue:
                raise ChargeError("Hydrogen %s bonded to atom in diffent residue (%s)" % (h, heavy))
            if heavy.element.number < 2:
                raise ChargeError("Hydrogen %s bonded to non-heavy atom %s" % (h, heavy))
            try:
                h.charge, h.gaff_type = hyd_charge_type_data[(h.residue.amber_name, heavy.name.lower())]
            except KeyError:
                raise ChargeError("Hydrogen %s bonded to atom that should not have hydrogens (%s)"
                    % (h, heavy))
        session.change_tracker.add_modified(modified_atoms, "charge changed")
        session.change_tracker.add_modified(modified_atoms, "gaff_type changed")
    #TODO

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

       'status' is where status messages go (e.g. session.logger.status)

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

    electrons = net_charge
    for a in r0.atoms:
        electrons += a.element.number
    if electrons % 2 == 1:
        # cannot compute charges for radical species
        raise ChargeError("%s: number of electrons (%d) + formal charge (%+d) is odd; cannot compute"
            " charges for radical species" % (r0.name, electrons - net_charge, net_charge))

    # detect tautomers by checking bonds
    varieties = {}
    for r in residues:
        atom_map = {}
        ordered_atoms = list(r.atoms)
        ordered_atoms.sort(key=lambda a: (a.name, a.coord_index))
        for i, a in enumerate(r.atoms):
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
        ante_failure_msg = "Failure running ANTECHAMBER for residue %s\nCheck reply log for details" % r.name
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

def _phosphorylate(session, status, deletes):
    session.logger.info("Deleting 5' phosphates from: %s" % ", ".join([str(r) for r in deletes]))
    from chimerax.atomic.struct_edit import add_atom
    for r in deletes:
        r.amber_name += "5"
        p = r.find_atom("P")
        o = None
        for nb in p.neighbors:
            for nnb in nb.neighbors:
                if nnb == p:
                    continue
                if nnb.element.number > 1:
                    o = nb
                    continue
                r.structure.delete_atom(nnb)
            if nb != o:
                r.structure.delete_atom(nb)
        if o is None:
            from chimerax.core.errors import UserError
            raise UserError("Atom P in residue %s is not connected to remainder of residue via an oxygen"
                % r)
        v = p.coord - o.coord
        sn = getattr(p, "serial_number", None)
        r.structure.delete_atom(p)
        from chimerax.geometry import normalize_vector
        v = normalize_vector(v) * 0.96
        add_atom("HO5'", 'H', r, o.coord + v, serial_number=sn, bonded_to=o)

