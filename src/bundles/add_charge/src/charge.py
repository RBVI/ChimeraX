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

class ChargeError(RuntimeError):
    pass

from chimerax.atomic.struct_edit import standardizable_residues
default_standardized = list(standardizable_residues)[:]
default_standardized.remove("MSE")

def add_charges(session, residues=None, *, method="am1-bcc",
        status=None, standardize_residues=default_standardized):
    uncharged_res_types = add_standard_charges(session, residues, status=status,
        standardize_residues=standardize_residues)
    for res_list in uncharged_res_types.values():
        add_nonstandard_res_charges(session, res_list, estimate_net_charge(res_list[0].atoms),
            method=method, status=status)

def add_standard_charges(session, residues=None, *, status=None, standardize_residues=default_standardized):
    """add AMBER charges to well-known residues

       'residues' restricts the addition to the specified residues

       'status' is where status messages go (e.g. session.logger.status)

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
    warn_UNK = False
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
                print(a.residue, a.residue.amber_name)
                raise ChargeError("Nonstandard name for heavy atom %s" % a)
        if r.name == 'UNK' and r.amber_name == "ALA":
            # we treat actual polymeric UNK residues as ALA, so that chains of
            # them aren't collated into a huge mega-residue; change the CB (which is missing
            # hydrogens) to neutral
            warn_UNK = True
            cb = r.find_atom('CB')
            if cb and cb.num_bonds == 1:
                cb.charge = 0.0
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
    if warn_UNK:
        session.logger.warning("There are UNK residues in the structure.  Charges in those regions will"
            " be inaccurate.")

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
            session.change_tracker.add_modified(
                a.fa_atom if isinstance(a, FakeAtom) else a, "charge changed")
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
    # Antechamber/sqm misbehaves for empty residue names [#9597], so use a real name if needed
    if not r.name:
        r_name = "UNL"
    else:
        r_name = r.name
    s.name = r_name

    # write out the residue's atoms first, since those are the ones we will be caring about
    nr = s.new_residue(r_name, ' ', 1)
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
        if method.lower().startswith("am1"):
            mth = "bcc"
            use_ek_flags = [True, False]
        elif method.lower().startswith("gas"):
            mth = "gas"
            use_ek_flags = [False]
        else:
            raise ValueError("Unknown charge method: %s" % method)

        # Using the -ek is significantly fasrer but slightly more likely to result in a convergence failure
        # so try using the flag first and fall back to not using it if convergence fails.  This only
        # applies to the AM1-BCC method.  See ticket #5729
        for use_ek_flag in use_ek_flags:
            command = [amber_bin + "/antechamber"]
            if use_ek_flag:
                command.extend(["-ek", "qm_theory='AM1',"])
            rerun = mth == "bcc" and not use_ek_flag

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
                status("%s ANTECHAMBER for residue %s"
                    % (("Re-running" if rerun else "Running"), r.name))
            from subprocess import Popen, STDOUT, PIPE
            # For some reason in Windows, if shell==False then antechamber cannot run bondtype via system()
            session.logger.info("Running ANTECHAMBER command: %s" % " ".join(command))
            os.environ['AMBERHOME'] = amber_home
            try:
                ante_messages = Popen(command, stdin=PIPE, stdout=PIPE, stderr=STDOUT, cwd=temp_dir,
                    bufsize=1, encoding="utf8").stdout
            except OSError as e:
                import sys
                if sys.platform == "darwin" and "Bad CPU type in executable" in str(e):
                    from chimerax.core.errors import LimitationError
                    raise LimitationError(
                        "The executable used to compute charges is an Intel executable, which needs the"
                        " Rosetta 2 emulator to be installed in order to run.  To install the emulator,\n"
                        " open Terminal.app (found in the Utilities sub-folder of the system Applications\n"
                        " folder) and type or paste the following command and hit Return:\n\n"
                        "    softwareupdate --install-rosetta")
                raise

            while True:
                line = ante_messages.readline()
                if not line:
                    break
                if status:
                    status("(%s) %s" % (r.name, line.rstrip()))
                # Using <code> avoids extra newlines
                session.logger.status("(%s) <code>%s</code>" % (r.name, line.rstrip()),
                    is_html=True, log=True)
            ante_failure_msg = "Failure running ANTECHAMBER for residue %s\nCheck reply log for details" \
                % r.name
            if not os.path.exists(ante_out):
                sqm_out = os.path.join(temp_dir, "sqm.out")
                if os.path.exists(sqm_out) and os.stat(sqm_out).st_size > 0:
                    with open(sqm_out) as f:
                        sqm_info = "<br><i>Contents of sqm.out:</i><pre>%s</pre>" % f.read()
                    if "No convergence in SCF" in sqm_info and mth == "bcc" and not rerun:
                        session.logger.status("Charges failed to converge using fast method;"
                            " re-running using slower more stable method", log=True)
                        continue
                    session.logger.info(sqm_info, is_html=True)
                raise ChargeError(ante_failure_msg)
            break
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

    @property
    def deleted(self):
        return self.fa_atom.deleted

def find_fake_name(base_name, known_names):
    import string
    for c in string.digits:
        fa_name = base_name + c
        if fa_name not in known_names:
            return fa_name
    if len(base_name)+1 < 4:
        for c in string.digits:
            fa_name = find_fake_name(base_name + c, known_names)
            if fa_name is not None:
                return fa_name
    return None

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
                        fa_name = find_fake_name(a.element.name.upper(), atom_names)
                        if not fa_name:
                            raise ChargeError(
                                f"Could not come up with unique atom name in mega-residue {name}")
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

    @property
    def deleted(self):
        self.atoms = [a for a in self.atoms if not a.deleted]
        return not self.atoms

    def find_atom(self, atom_name):
        for a in self.atoms:
            if a.name == atom_name:
                return a
        return None

    @property
    def num_atoms(self):
        return len(self.atoms)
