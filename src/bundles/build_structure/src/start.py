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
from chimerax.core.errors import LimitationError

def place_fragment(structure, fragment_name, res_name, position=None):
    '''Place a structure fragment

    *structure* is an AtomicStructure to add the fragment to.

    *fragment_name* is name of the fragment, as given in the "name" argument of its provider tag.
    A list of all names can be obtained by calling
    chimerax.build_structure.fragment_manager.get_manager(session).fragment_names

    *res_name* will be the name given to the residue created for the fragment

    Position is where the center of the chains will be positioned.  If None, at the center of the
    current view.

    Returns the created residue.
    '''
    session = structure.session
    from .fragment_manager import get_manager, FragmentNotInstalledError
    mgr = get_manager(session)
    try:
        fragment = mgr.fragment(fragment_name)
    except FragmentNotInstalledError as e:
        from chimerax.core import toolshed
        bi = mgr.provider_bundle(fragment_name)
        session.logger.info('<a href="%s">Install the %s bundle</a> to use "%s" fragment.' % (
            toolshed.get_toolshed().bundle_url(bi.name), bi.short_name, fragment_name), is_html=True)
        raise LimitationError("%s; see log for more info" % e)

    res = structure.new_residue(res_name, "het", _next_het_res_num(structure))
    need_focus = _need_focus(structure)

    position = final_position(structure, position)

    from chimerax.atomic.struct_edit import add_atom, gen_atom_name, add_bond
    from tinyarray import array
    atoms = []
    for element, xyz in fragment.atoms:
        atoms.append(add_atom(gen_atom_name(element, res), element, res, array(xyz)))
    for indices, depict in fragment.bonds:
        add_bond(atoms[indices[0]], atoms[indices[1]])

    # find fragment center
    atoms = res.atoms
    coords = atoms.coords
    center = coords.mean(0)
    correction = position - center
    atoms.coords = coords + correction

    if need_focus:
        from chimerax.core.commands import run
        run(session, "view", log=False)
    return res

def place_helium(structure, res_name, position=None):
    '''If position is None, place in the center of view'''
    res = structure.new_residue(res_name, "het", _next_het_res_num(structure))

    position = final_position(structure, position)

    from chimerax.atomic.struct_edit import add_atom
    helium = Element.get_element("He")
    a = add_atom("He", helium, res, position)
    from chimerax.atomic.colors import element_color
    a.color = element_color(helium.number)
    a.draw_mode = a.BALL_STYLE
    return a

import os
nuc_data_dir = os.path.join(os.path.dirname(__file__), "nuc-data")
nucleic_forms = []
for entry in os.listdir(nuc_data_dir):
    if entry.endswith(".xform"):
        nucleic_forms.append(entry[:-6])

class NucleicError(ValueError):
    pass

def place_nucleic_acid(structure, sequence, *, form='B', type="dna", position=None):
    """
    Place a nucleotide sequence (and its complementary chain).

    *structure* is an AtomicStructure to add the peptide to.

    *sequence* contains the sequence of the first chain.

    *form* is the (upper case) form (e.g. A); the supported forms are in the
    chimerax.build_structure.start.nucleic_forms list variable.

    If *type* is "dna", then both strands are DNA.  If "rna", then both are RNA.
    If "hybrid", then the first is DNA (and the sequence should be a DNA sequence) and the second DNA.

    Position is where the center of the chains will be positioned.  If None, at the center of the
    current view.

    The chains will be given the first two empty chain IDs.

    Returns a Chains collection containing the two chains.
    """

    if not sequence:
        raise NucleicError("No sequence supplied")
    sequence = sequence.upper()
    type = type.lower()
    if type == "rna":
        alphabet = "ACGU"
    else:
        alphabet = "ACGT"
    for let in sequence:
        if let not in alphabet:
            raise NucleicError("Sequence letter %s is illegal for %s"
                % (let, "RNA" if type == "rna" else "DNA"))
    if type == "rna":
        # treat U as T for awhile...
        sequence = sequence.replace('U', 'T')

    session = structure.session

    open_models = session.models[:]
    need_focus = _need_focus(structure)

    position = final_position(structure, position)

    if form not in nucleic_forms:
        raise NucleicError(form + "-form RNA/DNA not supported")
    xform_file = os.path.join(nuc_data_dir, form + '.xform')
    xform_values = []
    with open(xform_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            xform_values.append([float(x) for x in line.split()])
    from chimerax.geometry import Place
    xform = Place(xform_values)

    # find consecutive unused chain IDs
    existing_chain_ids = set(structure.residues.chain_ids)
    from chimerax.atomic import next_chain_id, chain_id_characters
    cid = chain_id_characters[0]
    while cid in existing_chain_ids or next_chain_id(cid) in existing_chain_ids:
        cid = next_chain_id(cid)
    chain_id1 = cid
    chain_id2 = next_chain_id(cid)

    # get bond info
    bonds = []
    with open(os.path.join(nuc_data_dir, "bonds"), "r") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 3:
                fields.append(None)
            bonds.append(fields)

    # build nucleotide
    from chimerax.atomic.struct_edit import add_atom
    serial_number = None
    residues1 = []
    residues2 = []
    cur_xform = Place()
    complement = { 'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A' }
    coord_cache = {}
    prev_residues = structure.residues
    from tinyarray import array
    for i, let in enumerate(sequence):
        if let not in coord_cache:
            coord_cache[let] = coords = {}
            coord_file = os.path.join(nuc_data_dir, "%s-%s%s.coords" % (form, let, complement[let]))
            with open(coord_file, "r") as f:
                for line in f:
                    strand, at_name, *xyz = line.strip().split()
                    strand = int(strand)
                    coords[(strand, at_name)] = array([float(crd) for crd in xyz])
        else:
            coords = coord_cache[let]
        type1 = let
        type2 = complement[let]
        if type != "rna":
            type1 = "D" + type1
        if type == "DNA":
            type2 = "D" + type2
        r1 = structure.new_residue(type1, chain_id1, i+1)
        r2 = structure.new_residue(type2, chain_id2, len(sequence)-i)
        residues1.append(r1)
        residues2.append(r2)
        for at_info, crd in coords.items():
            strand, at_name = at_info
            residue = r1 if strand == 0 else r2
            a = add_atom(at_name, at_name[0], residue, cur_xform * crd, serial_number=serial_number)
            serial_number = a.serial_number + 1
        for b1, b2, restriction in bonds:
            for r in (r1, r2):
                if restriction and r.name[-1] not in restriction:
                    continue
                a1, a2 = r.find_atom(b1), r.find_atom(b2)
                if a1 and a2:
                    structure.new_bond(a1, a2)
        if len(residues1) > 1:
            for res_list, a1, a2 in [(residues1, "O3'", "P"), (residues2, "P", "O3'")]:
                r1, r2 = res_list[-2:]
                structure.new_bond(r1.find_atom(a1), r2.find_atom(a2))
        cur_xform = cur_xform * xform
    structure.reorder_residues(list(prev_residues) + residues1 + list(reversed(residues2)))

    # strip dangling phosphorus
    for res in [residues1[0], residues2[-1]]:
        for aname in ["P", "OP1", "OP2"]:
            structure.delete_atom(res.find_atom(aname))

    # if RNA: modify sugar ring, swap U for T (including changing res type)
    if type != "dna":
        if type == "rna":
            residues = residues1 + residues2
        else:
            residues = residues2
        from chimerax.atomic.bond_geom import bond_positions, tetrahedral, planar
        from chimerax import geometry
        for res in residues:
            c1p, c2p, c3p, o3p = [res.find_atom(x) for x in ["C1'", "C2'", "C3'", "O3'"]]
            positions = bond_positions(c2p.coord, tetrahedral, 1.43, [c1p.coord, c3p.coord])
            # want the position nearer the O3'
            angle = geometry.angle(positions[0] - c2p.coord, o3p.coord - c3p.coord)
            if angle < 90.0:
                pos = positions[0]
            else:
                pos = positions[1]
            a = add_atom("O2'", "O", res, pos, serial_number=serial_number, bonded_to=c2p)
            serial_number = a.serial_number + 1

        for res in residues:
            if res.name != "T":
                continue
            structure.delete_atom(res.find_atom("C7"))
            res.name = "U"

    # reposition center to 'position'
    from chimerax.atomic import Chains
    chains = Chains([residues1[0].chain, residues2[0].chain])
    atoms = chains.existing_residues.atoms
    coords = atoms.coords
    center = coords.mean(0)
    correction = position - center
    atoms.coords = coords + correction

    if need_focus:
        from chimerax.core.commands import run
        run(session, "view", log=False)
    return chains

class PeptideError(ValueError):
    pass

def place_peptide(structure, sequence, phi_psis, *, position=None, rot_lib=None, chain_id=None):
    """
    Place a peptide sequence.

    *structure* is an AtomicStructure to add the peptide to.

    *sequence* contains the sequence as a string.

    *phi_psis* is a list of phi/psi tuples, one per residue.

    *position* is either an array/sequence specifying an xyz world coordinate position or None.
    If None, the peptide is positioned at the center of the view.

    *rot_lib* is the name of the rotamer library to use to position the side chains.
    session.rotamers.library_names lists installed rotamer library names.  If None, a default
    library will be used.

    *chain_id* is the desired chain ID for the peptide.  If None, earliest alphabetical chain ID
    not already in use in the structure will be used (upper case taking precedence over lower
    case).

    returns a Residues collection of the added residues
    """

    if not sequence:
        raise PeptideError("No sequence supplied")
    sequence = sequence.upper()
    if not sequence.isupper():
        raise PeptideError("Sequence contains non-alphabetic characters")
    from chimerax.atomic import Sequence
    for c in sequence:
        try:
            r3 = Sequence.protein1to3[c]
        except KeyError:
            raise PeptideError("Unrecognized protein 1-letter code: %s" % c)
        if r3[-1] == 'X':
            raise PeptideError("Peptide sequence cannot contain ambiguity codes (i.e. '%s')" % c)

    if len(sequence) != len(phi_psis):
        raise PeptideError("Number of phi/psis not equal to sequence length")

    session = structure.session

    open_models = session.models[:]
    need_focus = _need_focus(structure)

    position = final_position(structure, position)

    prev = [None] * 3
    pos = 1
    from chimerax.atomic.struct_edit import DIST_N_C, DIST_CA_N, DIST_C_CA, DIST_C_O, \
        find_pt, add_atom, add_dihedral_atom
    serial_number = None
    residues = []
    prev_psi = 0
    if chain_id is None:
        chain_id = unused_chain_id(structure)
    from numpy import array
    for c, phi_psi in zip(sequence, phi_psis):
        phi, psi = phi_psi
        while structure.find_residue(chain_id, pos):
            pos += 1
        r = structure.new_residue(Sequence.protein1to3[c], chain_id, pos)
        residues.append(r)
        for backbone, dist, angle, dihed in [('N', DIST_N_C, 116.6, prev_psi),
                ('CA', DIST_CA_N, 121.9, 180.0), ('C', DIST_C_CA, 110.1, phi)]:
            if prev[0] is None:
                pt = array([0.0, 0.0, 0.0])
            elif prev[1] is None:
                pt = array([dist, 0.0, 0.0])
            elif prev[2] is None:
                pt = find_pt(prev[0].coord, prev[1].coord, array([0.0, 1.0, 0.0]),
                    dist, angle, 0.0)
            else:
                pt = find_pt(prev[0].coord, prev[1].coord, prev[2].coord, dist, angle, dihed)
            a = add_atom(backbone, backbone[0], r, pt,
                serial_number=serial_number, bonded_to=prev[0])
            serial_number = a.serial_number + 1
            prev = [a] + prev[:2]
        o = add_dihedral_atom("O", "O", prev[0], prev[1], prev[2], DIST_C_O, 120.4, 180 + psi,
            bonded=True)
        prev_psi = psi
    # C terminus O/OXT at different angle than mainchain O
    structure.delete_atom(o)
    add_dihedral_atom("O", "O", prev[0], prev[1], prev[2], DIST_C_O, 117.0, 180 + psi, bonded=True)
    add_dihedral_atom("OXT", "O", prev[0], prev[1], prev[2], DIST_C_O, 117.0, psi, bonded=True)

    from chimerax.atomic import Residues
    residues = Residues(residues)

    from chimerax.swap_res import swap_aa
    # swap_aa is capable of swapping all residues in one call, but need to process one by one
    # since side-chain clashes are only calculated against pre-existing side chains
    kw = {}
    if rot_lib:
        kw['rot_lib'] = rot_lib
    for r in residues:
        swap_aa(session, [r], "same", criteria="cp", log=False, **kw)

    # find peptide center
    atoms = residues.atoms
    coords = atoms.coords
    center = coords.mean(0)
    correction = position - center
    atoms.coords = coords + correction

    from chimerax.std_commands.dssp import compute_ss
    compute_ss(session, structure)

    if need_focus:
        from chimerax.core.commands import run
        run(session, "view", log=False)
    return residues

def unused_chain_id(structure):
    from string import ascii_uppercase as uppercase, ascii_lowercase as lowercase, digits
    existing_ids = set([chain.chain_id for chain in structure.chains])
    for chain_characters in [uppercase, uppercase + digits + lowercase]:
        for id_length in range(1, 5):
            chain_id = _gen_chain_id(existing_ids, "", chain_characters, id_length-1)
            if chain_id:
                break
        else:
            continue
        break
    if chain_id is None:
        raise LimitationError("Could not find unused legal chain ID for peptide!")
    return chain_id

def _gen_chain_id(existing_ids, cur_id, legal_chars, rem_length):
    for c in legal_chars:
        next_id = cur_id + c
        if rem_length > 0:
            chain_id = _gen_chain_id(existing_ids, next_id, legal_chars, rem_length-1)
            if chain_id:
                return chain_id
        else:
            if next_id not in existing_ids:
                return next_id
    return None

def _need_focus(structure):
    open_models = structure.session.models[:]
    if len(open_models) == 0:
        return True
    if len(open_models) == 1:
        if open_models[0] == structure:
            return not structure.atoms
        return False
    return False

def _next_het_res_num(structure):
    max_existing = 0
    for r in structure.residues:
        if r.chain_id == "het" and r.number > max_existing:
            max_existing = r.number
    return max_existing+1

def final_position(structure, position):
    if position is None:
        position = structure.session.main_view.center_of_rotation
    from numpy import array
    pos = array(position)
    return structure.scene_position.inverse() * pos

