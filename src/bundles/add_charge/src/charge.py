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

from chimerax.atomic.struct_edit import standardizable_residues
default_standardized = list(standardizable_residues)[:]
default_standardized.remove("MSE")

def add_charges(session, residues=None, *, method="am1-bcc", phosphorylation=None, query_user=True,
        status=None, standardize_residues=default_standardized):
    uncharged_res_types = add_standard_charges(session, residues, status=status, query_user=query_user,
        phosphorylation=phosphorylation, standardize_residues=standardize_residues)
    for res_list in uncharged_res_types.values():
        add_nonstandard_res_charges(session, res_list, estimate_net_charge(res_list[0].atoms),
            method=method, status=status)

def add_standard_charges(session, residues=None, *, status=None, phosphorylation=None, query_user=True,
        standardize_residues=default_standardized):
    """add AMBER charges to well-known residues

       'residues' restricts the addition to the specified residues

       'status' is where status messages go (e.g. session.logger.status)

       'phosphorylation' controls whether chain-terminal nucleic acids will have their phosphorylation
       state changed to correspond to AMBER charge files (3' phosphorylated, 5' not).  A value of None
       means that the user will be queried if possible [treated as True if not possible], though if
       'query_user' is False, the user will not be queried.

       'standardize_residues' controls how residues that were modified to assist in crystallization
       are treated.  If True, the are changed to their normal counterparts (e.g. MSE->MET).  If
       False, they are left as is, which means that they will be treated as non-standard (except for
       MSE which gets special treatment due to its commonality) which also means that the charge
       calculation will likely fail since these residues contain bromine or selenium.
       'standardize_residues' can also be a list of residue names to standardize.

       The return value is a dictionary  which details the residues that did not receive charges
       [key: residue type, value: list of residues].

       Hydrogens need to be present.
    """
    from chimerax.atomic import Atom, Residues
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)
    elif not isinstance(residues, Residues):
        residues = Residues(residues)
    structures = residues.unique_structures

    if standardize_residues:
        if status:
            status("Standardizing residues")
        from chimerax.atomic.struct_edit import standardize_residues as sr
        if standardize_residues is True:
            sr(session, residues)
        else:
            sr(session, residues, res_types=standardize_residues)

    import os.path
    attr_file = os.path.join(os.path.split(__file__)[0], "amber_name.defattr")
    if status:
        status("Defining AMBER residue types")
    from chimerax.std_commands.defattr import defattr
    defattr(session, attr_file, restriction=structures, summary=False)

    if status:
        status("Checking phosphorylation of chain-terminal nucleic acids")
    deletes = []
    for r in residues:
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
                from chimerax.ui.ask import ask
                phosphorylation = ask(session, "Delete 5' terminal phosphates from nucleic acid chains?",
                        info="The AMBER charge set lacks parameters for terminal phosphates, and if"
                        " retained, such residues will be treated as non-standard",
                        title="Delete 5' phosphates?") == "yes"
            else:
                phosphorylation = True
        if phosphorylation:
            _phosphorylate(session, status, deletes)
        else:
            session.logger.info("Treating 5' terminal nucleic acids with phosphates as non-standard")
            for r in deletes:
                delattr(r, 'amber_name')
    if status:
        status("Adding standard charges")
    Atom.register_attr(session, "charge", "add charge", attr_type=float)
    Atom.register_attr(session, "gaff_type", "add charge", attr_type=str)
    from chimerax.amber_info import amber_version
    session.logger.info("Using Amber %s recommended default charges and atom types for standard residues"
        % amber_version)
    from .data import heavy_charge_type_data, hyd_charge_type_data
    uncharged_res_types = {}
    uncharged_residues = set()
    modified_atoms = []
    for r in residues:
        if not hasattr(r, "amber_name"):
            uncharged_residues.add(r)
            uncharged_res_types.setdefault(r.name, []).append(r)
            continue
        # since hydrogen names are so unreliable, look them up based on the atom they are bonded to
        # rather than their own name
        hydrogen_data = []
        for a in r.atoms:
            modified_atoms.append(a)
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
    if modified_atoms:
        session.change_tracker.add_modified(modified_atoms, "charge changed")
        session.change_tracker.add_modified(modified_atoms, "gaff_type changed")

    # merge connected non-standard residues into a "mega" residue.
    # also any standard residues directly connected
    # N.B. Can't iterate over .items() directly since we are changing the dictionary on the fly
    for urt, urs in list(uncharged_res_types.items()):
        for ur in urs[:]:
            if urt not in uncharged_res_types:
                break
            if ur not in uncharged_res_types[urt]:
                # connected to residue of same type and previously removed
                continue
            connected = [ur]
            queue = [ur]
            while queue:
                cur_res = queue.pop(0)
                neighbors = set()
                std_connects = {}
                for a in cur_res.atoms:
                    for na, nb in zip(a.neighbors, a.bonds):
                        na_res = na.residue
                        if na_res == cur_res or na_res in connected:
                            continue
                        # don't add standard residue if connected through chain bond
                        if na_res not in uncharged_residues:
                            if nb.polymeric_start_atom and na.name not in std_connects.get(na_res, set()):
                                std_connects.setdefault(na_res, set()).add(na.name)
                                continue
                        neighbors.add(na_res)
                neighbors = list(neighbors)
                neighbors.sort(key=lambda r: r.name)
                connected.extend(neighbors)
                queue.extend([nb for nb in neighbors if nb in uncharged_residues])
            # avoid using atom names with the trailing "-number" distiguisher if possible...
            if len(connected) > 1:
                fr = FakeRes(connected)
            else:
                fr = connected[0]
            uncharged_res_types.setdefault(fr.name, []).append(fr)
            for cr in connected:
                if cr in uncharged_residues:
                    uncharged_res_types[cr.name].remove(cr)
                    if not uncharged_res_types[cr.name]:
                        del uncharged_res_types[cr.name]
                    continue
    # split isolated atoms (e.g. metals) into separate "residues"
    urt_list = list(uncharged_res_types.items())
    for res_type, residues in urt_list:
        bond_residues = residues
        br_type = res_type
        while True:
            if len(bond_residues[0].atoms) == 1:
                break
            isolated_names = []
            for a in bond_residues[0].atoms:
                if a.bonds or a.name in isolated_names:
                    continue
                isolated_names.append(a.name)
                has_iso = [r for r in bond_residues if r.find_atom(a.name)]
                if len(has_iso) == len(bond_residues):
                    rem = []
                else:
                    rem = [r for r in bond_residues if r not in has_iso]
                iso = []
                non_iso = rem
                iso_type = "%s[%s]" % (res_type, a.name)
                br_type = "%s[non-%s]" % (br_type, a.name)
                for r in has_iso:
                    iso_res = FakeRes(iso_type, [fa for fa in r.atoms if fa.name == a.name])
                    iso.append(iso_res)
                    non_iso_atoms = [fa for fa in r.atoms if fa.name != a.name]
                    if not non_iso_atoms:
                        br_type = None
                        continue
                    non_iso_res = FakeRes(br_type, non_iso_atoms)
                    non_iso.append(non_iso_res)
                urt_list.append((iso_type, iso))
                uncharged_res_types[iso_type] = iso
                bond_residues = non_iso
            else:
                # no isolated atoms
                break
        if br_type != res_type:
            del uncharged_res_types[res_type]
            if br_type != None:
                uncharged_res_types[br_type] = bond_residues

    # despite same residue type, residues may still differ -- particularly terminal vs. non-terminal...
    # can't modify a dictionary while you're iterating over it, so...
    for res_type, residues in list(uncharged_res_types.items()):
        if len(residues) < 2:
            continue
        varieties = {}
        for r in residues:
            key = tuple(sorted([a.name for a in r.atoms]))
            varieties.setdefault(key, []).append(r)
        if len(varieties) == 1:
            continue
        # in order to give the varieties distinguishing names, find atoms in common
        keys = list(varieties.keys())
        common = set(keys[0])
        for k in keys[1:]:
            common = common.intersection(set(k))
        uncommon = set()
        for k in keys:
            uncommon = uncommon.union(set(k) - common)
        del uncharged_res_types[res_type]
        for k, residues in varieties.items():
            names = set(k)
            more = names - common
            less = uncommon - names
            new_key = res_type
            if more:
                new_key += " (w/%s)" % ",".join(list(more))
            if less:
                new_key += " (wo/%s" % ",".join(list(less))
            uncharged_res_types[new_key] = residues
    if status:
        status("Standard charges added")
    return uncharged_res_types

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

def add_nonstandard_res_charges(session, residues, net_charge, method="am1-bcc", *, status=None):
    """Add Antechamber charges to non-standard residue

       'residues' is a list of residues of the same type.  The first
       residue in the list will be used as an exemplar for the whole
       type for purposes of charge determination, but charges will be
       added to all residues in the list.

       'net_charge' is the net charge of the residue type.

       'method' is either 'am1-bcc' or 'gasteiger'

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
        bonds.sort(key=lambda b: (b[0], (-1 if b[1] is None else b[1])))
        varieties.setdefault(tuple(bonds), []).append(r)
    if len(varieties) > 1:
        session.logger.info("%d tautomers of %s; charging separately" % (len(varieties), r0.name))
    for tautomer_residues in varieties.values():
        nonstd_charge(session, tautomer_residues, net_charge, method, status=status)

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
    all_atom_rings = {}
    aro_atom_rings = {}
    ring_atoms = {}
    # _ng_charge() can call b.smaller_side, so need to cache ring info...
    for a in atoms:
        if a not in all_atom_rings:
            all_atom_rings[a] = [id(ar) for ar in a.rings()]
            aro_atom_rings[a] = [id(ar) for ar in a.rings() if ar.aromatic]
            ring_atoms.update({id(ar):ar.atoms for ar in a.rings()})
        for nb in a.neighbors:
            if nb not in all_atom_rings:
                all_atom_rings[nb] = [id(ar) for ar in nb.rings()]
                aro_atom_rings[nb] = [id(ar) for ar in nb.rings() if ar.aromatic ]
                ring_atoms.update({id(ar):ar.atoms for ar in nb.rings()})
    subs = {}
    rings = set()
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
        rings.update(aro_atom_rings[a])
        if a.idatm_type == "C2" and not all_atom_rings[a]:
            for nb in a.neighbors:
                if not aro_atom_rings[nb]:
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
        for a in ring_atoms[ring]:
            if a in subs:
                electrons += a.element.number + subs[a] - 2
            else:
                electrons += a.element.number + a.num_bonds - 2
            if a.idatm_type[-1] in "+-":
                electrons += 1
        if electrons % 2 == 1:
            charge_total += 2
    return charge_total // 2


def _get_aname(element, element_counts):
    element_count = element_counts.setdefault(element, 0) + 1
    a_name = element.name + "%d" % element_count
    element_counts[element] = element_count
    return a_name

def _methylate(na, n, element_counts):
    added = []
    from chimerax.atomic import Element
    from chimerax.atomic.struct_edit import add_atom
    C = Element.get_element("C")
    nn = add_atom(_get_aname(C, element_counts), C, na.residue, n.coord)
    added.append(nn)
    na.structure.new_bond(na, nn)
    from chimerax.atomic.bond_geom import bond_positions
    H = Element.get_element("H")
    for pos in bond_positions(nn.coord, 4, 1.1, [na.coord]):
        nh = add_atom(_get_aname(H, element_counts), H, na.residue, pos)
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
    # the below is more resilient to modified groups (e.g. 2MR) than the previous method
    b = atom.bonds[atom.neighbors.index(nb)]
    try:
        return int(b.smaller_side == nb)
    except ValueError:
        # bond is in a cycle
        return 1

def _n2_charge(atom):
    # needed in order to get nitrite ions correct
    for nb in atom.neighbors:
        if nb.idatm_type != "O2-":
            return 0
    return 2

def nonstd_charge(session, residues, net_charge, method, *, status=None, temp_dir=None):
    """Underlying "workhorse" function for add_nonstandard_res_charges()

       Other than 'temp_dir', the arguments are the same as for add_nonstandard_res_charges(),
       and in almost all situations you should use that function instead, but if you need
       access to the input/output files to/from Antechamber, you can use this function to specify
       a path for 'temp_dir' and all Antechamber files will be written into that directory and
       the directory will not be deleted afterward.  Otherwise a temporary directory is used that
       is deleted afterward.
    """
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
    # use same ordering of atoms as they had in input, to improve consistency of antechamber charges
    r_atoms = sorted(r.atoms, key=lambda a: a.coord_index)
    from chimerax.atomic.struct_edit import add_atom
    element_counts = {}
    for a in r_atoms:
        # Antechamber expects the atom names to start with the case-sensitive atomic symbol, and for
        # single-letter atomic symbols, _not_ to be followed with upper case letters, though it
        # makes exceptions for common cases like carbon.  Nonetheless it doesn't make an exception for
        # flourine, which fouls up the FAC/FAD/FAE atoms in J8A of 6ega, so...
        atom_map[a] = add_atom(_get_aname(a.element, element_counts), a.element, nr, a.coord)

    # add the intraresidue bonds and remember the interresidue ones
    electrons = 0
    nearby = set()
    for a in r_atoms:
        electrons += a.element.number
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
        na = add_atom(_get_aname(nb.element, element_counts), nb.element, nr, nb.coord)
        extras.add(na)
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
                    extras.update(_methylate(na, nbnb, element_counts))
    for ea in extras:
        electrons += ea.element.number
    total_net_charge = net_charge + estimate_net_charge(extras)

    if (electrons + total_net_charge) % 2 == 1 and method == "am1-bcc":
        # cannot compute charges for radical species with AM1-BCC
        raise ChargeError("%s: number of electrons (%d) + formal charge (%+d) is odd; cannot compute charges"
            " for radical species using AM1-BCC method" % (r.name, electrons, total_net_charge))


    from contextlib import contextmanager
    @contextmanager
    def managed_structure(s):
        try:
            yield s
        finally:
            s.delete()

    if not temp_dir:
        import tempfile
        # hold direct reference so that the directory isn't immediately deleted
        temp_dir_ref = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_ref.name
    with managed_structure(s) as s:
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
            if status:
                status("(%s) %s" % (r.name, line.rstrip()))
            session.logger.status("(%s) <code>%s</code>" % (r.name, line.rstrip()), is_html=True, log=True)
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
            ta.gaff_type = ca.mol2_type
        # map template charges onto first residue
        assigned = set()
        for fa, ta in atom_map.items():
            if ta in extras:
                continue
            fa.charge = ta.charge
            fa.gaff_type = ta.gaff_type
            if isinstance(fa, FakeAtom):
                assigned.add(fa.fa_atom)
            else:
                assigned.add(fa)
        # map charges onto remaining residues
        r_atoms = sorted(r.atoms)
        for rr in residues[1:]:
            rr_atoms = sorted(rr.atoms)
            for fa, ra in zip(r_atoms, rr_atoms):
                ra.charge = fa.charge
                ra.gaff_type = fa.gaff_type
                if isinstance(ra, FakeAtom):
                    assigned.add(ra.fa_atom)
                else:
                    assigned.add(ra)
        session.change_tracker.add_modified(assigned, "charge changed")
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

class FakeAtom:
    def __init__(self, atom, res, name=None):
        if isinstance(atom, FakeAtom):
            self.fa_atom = atom.fa_atom
        else:
            self.fa_atom = atom
        self.fa_res = res
        self.fa_name = name

    def __eq__(self, other):
        if isinstance(other, FakeAtom):
            return other is self
        return self.fa_atom == other

    def __getattr__(self, attr_name):
        if attr_name == "name" and self.fa_name:
            return self.fa_name
        if attr_name == "residue":
            return self.fa_res
        if attr_name == "neighbors":
            real_neighbors = self.fa_atom.neighbors
            lookup = {}
            for fa in self.fa_res.atoms:
                lookup[fa.fa_atom] = fa
            return [lookup.get(x, x) for x in real_neighbors]
        return getattr(self.fa_atom, attr_name)

    def __hash__(self):
        return hash(self.fa_atom)

    def __lt__(self, other):
        if isinstance(other, FakeAtom):
            return self.fa_atom < other.fa_atom
        return self.fa_atom < other

    def __setattr__(self, name, val):
        if name.startswith("fa_"):
            self.__dict__[name] = val
        else:
            setattr(self.fa_atom, name, val)

class FakeRes:
    def __init__(self, name, atoms=None):
        if atoms is None:
            # mega residue
            residues = name
            name = "+".join([r.name for r in residues])
            # do our best to keep atom names 4 characters or less
            atom_names = set()
            atoms = []
            import string
            for r in residues:
                for a in r.atoms:
                    if a.name in atom_names:
                        for c in string.digits + string.ascii_uppercase:
                            fa_name = a.name[:3] + c
                            if fa_name not in atom_names:
                                break
                        else:
                            raise ValueError("Could not come up with unique atom name in mega-residue")
                        fa = FakeAtom(a, self, fa_name)
                    else:
                        fa = FakeAtom(a, self)
                    atoms.append(fa)
                    atom_names.add(fa.name)
        else:
            atoms = [FakeAtom(a, self) for a in atoms]
        self.name = name
        self.atoms = atoms
        self.structure = atoms[0].structure

    def find_atom(self, atom_name):
        for a in self.atoms:
            if a.name == atom_name:
                return a
        return None

    @property
    def num_atoms(self):
        return len(self.atoms)

