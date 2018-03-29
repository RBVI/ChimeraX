# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2018 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
Write mmCIF files that ChimeraX would like to read.
"""
from chimerax.core.atomic import mmcif
from chimerax.core.utils import flattened
import platform
from chimerax import app_dirs
import sys

WRITER_VERSION = 'v7'

# chains only keep the single letter for missing residues, but entity_poly_seq
# wants the multiletter version, so fake the names for the missing residues
_rna1to3 = {
    'A': 'A',
    'G': 'G',
    'C': 'C',
    'U': 'U',
    'I': 'I',
    'N': 'N',
}
_dna1to3 = {
    'A': 'DA',
    'G': 'DG',
    'C': 'DC',
    'T': 'DT',
    'I': 'DI',
    'N': 'DN',
}
_protein1to3 = {
    'A': "ALA",
    'B': "ASX",
    'C': "CYS",
    'D': "ASP",
    'E': "GLU",
    'F': "PHE",
    'G': "GLY",
    'H': "HIS",
    'I': "ILE",
    'K': "LYS",
    'L': "LEU",
    'M': "MET",
    'N': "ASN",
    'P': "PRO",
    'Q': "GLN",
    'R': "ARG",
    'S': "SER",
    'T': "THR",
    'V': "VAL",
    'W': "TRP",
    'Y': "TYR",
    'Z': "GLX",
}


def write_mmcif(session, path, models=None):
    from . import Structure
    if models is None:
        models = session.models.list(type=Structure)
    else:
        models = [m for m in models if isinstance(m, Structure)]

    if not models:
        session.logger.info("no structures to save")
        return

    # Need to figure out which ChimeraX models should be grouped together
    # as mmCIF models.  For now assume all models with the same "parent"
    # id (other than blank) are actually a nmr ensemble.
    # TODO: fix for docking models and hierarchical models
    grouped = {}
    for m in models:
        grouped.setdefault(m.id[:-1], []).append(m)

    used_data_names = set()
    with open(path, 'w') as f:
        for g in grouped:
            models = grouped[g]
            # TODO: make sure ensembles are distinguished from IHM models
            ensemble = all(m.name == models[0].name for m in models)
            if ensemble:
                save_structure(session, f, models, used_data_names)
            else:
                for m in models:
                    save_structure(session, f, [m], used_data_names)


ChimeraX_audit_conform = mmcif.MMCIFTable(
    "audit_conform",
    (
        "dict_name",
        "dict_version",
        "dict_location",
        "pdbx_keywords_flag",
        "pdbx_fixed_width_columns"
    ), (
        "mmcif_pdbx.dic",
        "4.007",
        "http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic",
        "Y",
        "atom_site,atom_site_anisotrop"
    )
)

ChimeraX_citation = mmcif.MMCIFTable(
    "citation",
    (
        'id', 'title', 'journal_abbrev', 'journal_volume', 'year',
        'page_first', 'page_last',
        'journal_issue', 'pdbx_database_id_PubMed', 'pdbx_database_id_DOI'
    ), [
        'chimerax',
        "UCSF ChimeraX: Meeting Modern Challenges in Visualization and Analysis",
        "Protein Sci.",
        '27',
        '2018',
        '14',
        '25',
        '1',
        '28710774',
        '10.1002/pro.3235',
    ],
)

ChimeraX_authors = mmcif.MMCIFTable(
    "citation_author",
    (
        'citation_id', 'name', 'ordinal'
    ), [
        'chimerax', 'Goddard TD', '1',
        'chimerax', 'Huang CC', '2',
        'chimerax', 'Meng EC', '3',
        'chimerax', 'Pettersen EF', '4',
        'chimerax', 'Couch GS', '5',
        'chimerax', 'Morris JH', '6',
        'chimerax', 'Ferrin TE', '7',
    ]
)
_CHAIN_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

system = platform.system()
if system == 'Darwin':
    system = 'macOS'  # TODO? Mac OS X (thru 10.7)/OS X thru 10.11/macOS
ChimeraX_software_values = [
    '%s %s' % (app_dirs.appauthor, app_dirs.appname),
    "%s/%s" % (app_dirs.version, WRITER_VERSION),
    'https://www.rbvi.ucsf.edu/chimerax/',
    'model building',
    system,
    'package',
    'chimerax',
    'ORIDINAL',
]
ChimeraX_software = mmcif.MMCIFTable(
    "software",
    (
        'name',
        'version',
        'location',
        'classification',
        'os',
        'type',
        'citation_id',
        'pdbx_ordinal',
    ), ChimeraX_software_values
)
del system


def _mmcif_chain_id(i):
    # TODO: want A..Z, AA..AZ getting A..Z, BA..BZ
    assert i > 0
    i = i - 1
    if i == 0:
        return 'A'
    output = []
    num_chars = len(_CHAIN_CHARS)
    while i > 0:
        output.append(_CHAIN_CHARS[i % num_chars])
        i //= num_chars
    output.reverse()
    return ''.join(output)


def _save_metadata(model, categories, file):
    tables = mmcif.get_mmcif_tables_from_metadata(model, categories)
    printed = False
    for t in tables:
        if t is None:
            continue
        t.print(file, fixed_width=True)
        printed = True
    return printed


def save_structure(session, file, models, used_data_names):
    # save mmCIF data section for a structure
    # 'models' should only have more than one model if NMR ensemble
    # All 'models' should have the same metadata.
    # All 'models' should have the same number of atoms, but in PDB files
    # then often don't, so pick the model with the most atoms.
    #
    if len(models) <= 1:
        best_m = models[0]
    else:
        # TODO: validate that the models are actually similar, ie.,
        # same number of chains which same chain ids and sequences,
        # and same kinds of HET residues
        tmp = list(models)
        tmp.sort(key=lambda m: m.num_atoms)
        best_m = tmp[0]
    name = ''.join(best_m.name.split())  # Remove all whitespace from name
    name = name.encode('ascii', errors='ignore').decode('ascii')  # Drop non-ascii characters
    if name in used_data_names:
        count = 0
        while True:
            count += 1
            n = '%s_%d' % (name, count)
            if n not in used_data_names:
                break
        name = n
    used_data_names.add(name)

    print('data_%s' % name, file=file)
    print('#', file=file)

    # TODO? entry
    # entry, = mmcif.get_mmcif_tables_from_metadata(best_m, ['entry'])
    # entry.print(file)

    ChimeraX_audit_conform.print(file=file, fixed_width=True)

    citation, citation_author, software = mmcif.get_mmcif_tables_from_metadata(
        best_m, ['citation', 'citation_author', 'software'])
    if not citation:
        citation = ChimeraX_citation
        citation_author = ChimeraX_authors
    elif not citation.field_has('citation_id', 'chimerax'):
        citation.extend(ChimeraX_citation)
        citation_author.extend(ChimeraX_authors)
    citation.print(file, fixed_width=True)
    citation_author.print(file, fixed_width=True)
    if not software:
        ChimeraX_software_values[-1] = '1'
        software = ChimeraX_software
    elif software.field_has('citation_id', 'chimerax'):
        pass  # TODO: update with current version
    else:
        ChimeraX_software_values[-1] = str(software.num_rows() + 1)
        software.extend(ChimeraX_software)
    software.print(file, fixed_width=True)
    del citation, citation_author, software

    save_components(best_m, file)

    from chimerax.core.atomic import Residue
    old_entity, old_asym = mmcif.get_mmcif_tables_from_metadata(best_m, ['entity', 'struct_asym'])
    try:
        if not old_entity or not old_asym:
            raise ValueError
        old_mmcif_chain_to_entity = old_asym.mapping('id', 'entity_id')
        old_entity_to_description = old_entity.mapping('id', 'pdbx_description')
    except ValueError:
        old_mmcif_chain_to_entity = {}
        old_entity_to_description = {}

    entity_info = {}    # { entity_id: (type, pdbx_description) }
    asym_info = {}      # { auth_chain_id: (entity_id, label_asym_id) }
    het_asym_info = {}  # { mmcif_chain_id: (entity_id, label_asym_id) }
    poly_info = []      # [(entity_id, type, one-letter-seq)]
    poly_seq_info = []  # [(entity_id, num, mon_id)]
    residue_info = {}   # { residue: (label_asym_id, label_seq_id) }
    asym_id = 0

    seq_entities = {}   # { chain.characters : entity_id }
    for c in best_m.chains:
        chars = c.characters
        if chars in seq_entities:
            eid = seq_entities[chars]
        else:
            mcid = c.existing_residues[0].mmcif_chain_id
            try:
                descrip = old_entity_to_description[old_mmcif_chain_to_entity[mcid]]
            except KeyError:
                descrip = '?'
            eid = len(entity_info) + 1
            entity_info[eid] = ('polymer', descrip)
            seq_entities[chars] = eid
            # _1to3 is reverse map to handle missing residues
            if c.polymer_type == Residue.PT_AMINO:
                _1to3 = _protein1to3
                poly_info.append((eid, 'polypeptide(L)', chars))  # TODO: or polypeptide(D)
            else:
                # figure out if DNA
                names = set(c.existing_residues.names)
                if names.isdisjoint(set(_dna1to3)):
                    _1to3 = _rna1to3
                    poly_info.append((eid, 'polydeoxyribonucleotide', chars))
                else:
                    _1to3 = _dna1to3
                    poly_info.append((eid, 'polyribonucleotide', chars))
            for seq_id, ch, r in zip(range(1, sys.maxsize), chars, c.residues):
                label_seq_id = str(seq_id)
                if r:
                    name = r.name
                else:
                    name = _1to3.get(ch, 'UNK')
                poly_seq_info.append((eid, label_seq_id, name))
        asym_id += 1
        label_asym_id = _mmcif_chain_id(asym_id)
        asym_info[c.chain_id] = (label_asym_id, eid)

    het_entities = {}   # { het_name: { 'entity': entity_id, chain: (label_entity_id, label_asym_id) } }
    residues = best_m.residues
    het_residues = residues.filter(residues.polymer_types == Residue.PT_NONE)
    for r in het_residues:
        mcid = r.mmcif_chain_id
        n = r.name
        if n in het_entities:
            eid = het_entities[n]['entity']
        else:
            if n == 'HOH':
                etype = 'water'
            else:
                etype = 'non-polymer'
            try:
                descrip = old_entity_to_description[old_mmcif_chain_to_entity[mcid]]
            except KeyError:
                descrip = '?'
            eid = len(entity_info) + 1
            entity_info[eid] = (etype, descrip)
            het_entities[n] = {'entity': eid}
        if mcid in het_entities[n]:
            continue
        asym_id += 1
        label_asym_id = _mmcif_chain_id(asym_id)
        het_asym_info[mcid] = (label_asym_id, eid)
        het_entities[n][mcid] = (eid, label_asym_id)

    entity = mmcif.MMCIFTable('entity', ['id', 'type', 'pdbx_description'], flattened(entity_info.items()))
    entity.print(file, fixed_width=True)
    entity_poly = mmcif.MMCIFTable('entity_poly', ['entity_id', 'type', 'pdbx_seq_one_letter_code_can'], flattened(poly_info))
    entity_poly.print(file, fixed_width=True)
    entity_poly_seq = mmcif.MMCIFTable('entity_poly_seq', ['entity_id', 'num', 'mon_id'], flattened(poly_seq_info))
    entity_poly_seq.print(file, fixed_width=True)
    import itertools
    struct_asym = mmcif.MMCIFTable(
        'struct_asym', ['id', 'entity_id'],
        flattened(itertools.chain(asym_info.values(), het_asym_info.values())))
    struct_asym.print(file, fixed_width=True)
    del entity, entity_poly_seq, struct_asym

    elements = list(set(best_m.atoms.elements))
    elements.sort(key=lambda e: e.number)
    atom_type_data = [e.name for e in elements]
    atom_type = mmcif.MMCIFTable("atom_type", ["symbol"], atom_type_data)
    atom_type.print(file, fixed_width=True)
    del atom_type_data, atom_type

    atom_site_data = []
    atom_site = mmcif.MMCIFTable("atom_site", [
        'group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_alt_id',
        'label_comp_id', 'label_asym_id', 'label_entity_id', 'label_seq_id',
        'Cartn_x', 'Cartn_y', 'Cartn_z',
        'auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code',
        'occupancy', 'B_iso_or_equiv', 'pdbx_PDB_model_num'
    ], atom_site_data)
    atom_site_anisotrop_data = []
    atom_site_anisotrop = mmcif.MMCIFTable("atom_site_anisotrop", [
        'id', 'type_symbol',
        'U[1][1]', 'U[1][2]', 'U[1][3]',
        'U[2][2]', 'U[2][3]', 'U[3][3]',
    ], atom_site_anisotrop_data)
    serial_num = 0

    def atom_site_residue(residue, group, seq_id, asym_id, entity_id, model_num):
        nonlocal serial_num, atom_site_data, residue_info
        residue_info[residue] = (asym_id, seq_id)
        atoms = residue.atoms
        for atom in atoms:
            elem = atom.element.name
            aname = atom.name
            rname = residue.name
            cid = residue.chain_id
            if cid == ' ':
                cid = '.'
            rnum = residue.number
            rins = residue.insertion_code
            if not rins:
                rins = '?'
            original_alt_loc = atom.alt_loc
            for alt_loc in atom.alt_locs or '.':
                if alt_loc is not '.':
                    atom.set_alt_loc(alt_loc, False)
                xyz = ['%.3f' % f for f in atom.scene_coord]
                occ = "%.2f" % atom.occupancy
                bfact = "%.2f" % atom.bfactor
                serial_num += 1
                atom_site_data.append((
                    group, serial_num, elem, aname, alt_loc, rname, asym_id,
                    entity_id, seq_id, *xyz, cid, rnum, rins, occ, bfact,
                    model_num))
                u6 = atom.aniso_u6
                if u6 is not None and len(u6) > 0:
                    u6 = ['%.4f' % f for f in u6]
                    atom_site_anisotrop_data.append((serial_num, elem, u6))
            if alt_loc is not '.':
                atom.set_alt_loc(original_alt_loc, False)

    for m, model_num in zip(models, range(1, sys.maxsize)):
        residues = m.residues
        het_residues = residues.filter(residues.polymer_types == Residue.PT_NONE)
        for c in m.chains:
            for seq_id, r in zip(range(1, sys.maxsize), c.residues):
                if r is None:
                    continue
                asym_id, entity_id = asym_info[r.chain_id]
                atom_site_residue(r, 'ATOM', seq_id, asym_id, entity_id, model_num)
            chain_id = c.chain_id
            chain_het = het_residues.filter(het_residues.chain_ids == chain_id)
            het_residues -= chain_het
            for r in chain_het:
                asym_id, entity_id = het_asym_info[r.mmcif_chain_id]
                atom_site_residue(r, 'HETATM', '.', asym_id, entity_id, model_num)
        for r in het_residues:
            asym_id, entity_id = het_asym_info[r.mmcif_chain_id]
            atom_site_residue(r, 'HETATM', '.', asym_id, entity_id, model_num)

    atom_site_data[:] = flattened(atom_site_data)
    atom_site.print(file, fixed_width=True)
    atom_site_anisotrop_data[:] = flattened(atom_site_anisotrop_data)
    atom_site_anisotrop.print(file, fixed_width=True)
    del atom_site_data, atom_site, atom_site_anisotrop_data, atom_site_anisotrop

    struct_conn_data = []
    struct_conn = mmcif.MMCIFTable("struct_conn", [
        "id", "conn_type_id",
        "ptnr1_label_atom_id",
        "pdbx_ptnr1_label_alt_id",
        "ptnr1_label_asym_id",
        "ptnr1_label_seq_id",
        "ptnr1_auth_asym_id",
        "ptnr1_auth_seq_id",
        "pdbx_ptnr1_PDB_ins_code",
        "ptnr1_label_comp_id",
        "ptnr1_symmetry",
        "ptnr2_label_atom_id",
        "pdbx_ptnr2_label_alt_id",
        "ptnr2_label_asym_id",
        "ptnr2_label_seq_id",
        "ptnr2_auth_asym_id",
        "ptnr2_auth_seq_id",
        "pdbx_ptnr2_PDB_ins_code",
        "ptnr2_label_comp_id",
        "ptnr2_symmetry",
        "pdbx_dist_value",
    ], struct_conn_data)

    struct_conn_type_data = []
    struct_conn_type = mmcif.MMCIFTable("struct_conn_type", [
        "id",
    ], struct_conn_type_data)

    def struct_conn_bond(tag, b, a0, a1):
        nonlocal count
        r0 = a0.residue
        r1 = a1.residue
        r0_asym, r0_seq = residue_info[r0]
        r1_asym, r1_seq = residue_info[r1]
        cid0 = r0.chain_id
        if cid0 == ' ':
            cid0 = '.'
        rnum0 = r0.number
        rins0 = r0.insertion_code
        if not rins0:
            rins0 = '?'
        cid1 = r1.chain_id
        if cid1 == ' ':
            cid1 = '.'
        rnum1 = r1.number
        rins1 = r1.insertion_code
        if not rins1:
            rins1 = '?'
        # find all alt_loc pairings
        alt_pairs = []
        for alt_loc in a0.alt_locs:
            if a1.has_alt_loc(alt_loc):
                alt_pairs.append((alt_loc, alt_loc))
            elif not a1.alt_locs:
                alt_pairs.append((alt_loc, ' '))
        else:
            for alt_loc in a1.alt_locs:
                alt_pairs.append((' ', alt_loc))
            else:
                alt_pairs.append((' ', ' '))
        original_alt_loc0 = a0.alt_loc
        original_alt_loc1 = a1.alt_loc
        for alt_loc0, alt_loc1 in alt_pairs:
            if alt_loc0 == ' ':
                alt_loc0 = '.'
            else:
                a0.set_alt_loc(alt_loc0, False)
            if alt_loc1 == ' ':
                alt_loc1 = '.'
            else:
                a1.set_alt_loc(alt_loc1, False)
            dist = "%.3f" % b.length
            count += 1
            struct_conn_data.append((
                '%s%d' % (tag, count), tag,
                a0.name, alt_loc0, r0_asym, r0_seq, cid0, rnum0, rins0, r0.name, "1_555",
                a1.name, alt_loc1, r1_asym, r1_seq, cid1, rnum1, rins1, r1.name, "1_555",
                dist))
        if original_alt_loc0 is not ' ':
            a0.set_alt_loc(original_alt_loc0, False)
        if original_alt_loc1 is not ' ':
            a1.set_alt_loc(original_alt_loc1, False)

    # disulfide bonds
    from chimerax.core.atomic import Sequence
    rname3to1 = Sequence.rname3to1
    count = 0
    atoms = best_m.atoms
    bonds = best_m.bonds
    has_disulf = False
    covalent = []
    sg = atoms.filter(atoms.names == "SG")
    for b, a0, a1 in zip(bonds, *bonds.atoms):
        if a0 in sg and a1 in sg:
            has_disulf = True
            struct_conn_bond('disulf', b, a0, a1)
            continue
        r0 = a0.residue
        r1 = a1.residue
        if r0 == r1:
            continue
        if r0.chain is None or r0.chain != r1.chain:
            covalent.append((b, a0, a1))
        elif rname3to1(r0.name) == 'X' or rname3to1(r1.name) == 'X':
            covalent.append((b, a0, a1))
    if has_disulf:
        struct_conn_type_data.append('disulf')

    # metal coordination bonds
    # assume intra-residue metal coordination bonds are handled by residue template
    count = 0
    pbg = best_m.pseudobond_group(best_m.PBG_METAL_COORDINATION, create_type=None)
    if pbg:
        bonds = pbg.pseudobonds
        if len(bonds) > 0:
            struct_conn_type_data.append('metalc')
        for b, a0, a1 in zip(bonds, *bonds.atoms):
            if a0.residue == a1.residue:
                continue
            struct_conn_bond('metalc', b, a0, a1)

    # hydrogen bonds
    count = 0
    pbg = best_m.pseudobond_group(best_m.PBG_HYDROGEN_BONDS, create_type=None)
    if pbg:
        bonds = pbg.pseudobonds
        if len(bonds) > 0:
            struct_conn_type_data.append('hydrog')
        for b, a0, a1 in zip(bonds, *bonds.atoms):
            struct_conn_bond('hydrog', b, a0, a1)

    # extra/other covalent bonds
    # TODO: covalent bonds not in resdiue template
    count = 0
    if len(covalent) > 0:
        struct_conn_type_data.append('covale')
    for b, a0, a1 in covalent:
        struct_conn_bond('covale', b, a0, a1)

    struct_conn_data[:] = flattened(struct_conn_data)
    struct_conn.print(file, fixed_width=True)
    # struct_conn_type_data[:] = flattened(struct_conn_type_data)
    struct_conn_type.print(file, fixed_width=True)
    del struct_conn_data, struct_conn, struct_conn_type_data, struct_conn_type

    # struct_conf
    struct_conf_data = []
    struct_conf = mmcif.MMCIFTable("struct_conf", [
        "id", "conf_type_id",
        "beg_label_comp_id",
        "beg_label_asym_id",
        "beg_label_seq_id",
        "end_label_comp_id",
        "end_label_asym_id",
        "end_label_seq_id",
        "beg_auth_asym_id",
        "beg_auth_seq_id",
        "pdbx_beg_PDB_ins_code",
        "end_auth_asym_id",
        "end_auth_seq_id",
        "pdbx_end_PDB_ins_code",
    ], struct_conf_data)

    struct_conf_type_data = []
    struct_conf_type = mmcif.MMCIFTable("struct_conf_type", [
        "id"
    ], struct_conf_type_data)

    def struct_conf_entry(id, ctype, beg_res, end_res):
        beg_asym, beg_seq = residue_info[beg_res]
        end_asym, end_seq = residue_info[end_res]
        beg_cid = beg_res.chain_id
        if beg_cid == ' ':
            beg_cid = '.'
        beg_rnum = beg_res.number
        beg_rins = beg_res.insertion_code
        if not beg_rins:
            beg_rins = '?'
        end_cid = end_res.chain_id
        if end_cid == ' ':
            end_cid = '.'
        end_rnum = end_res.number
        end_rins = end_res.insertion_code
        if not end_rins:
            end_rins = '?'
        struct_conf_data.append((
            id, ctype,
            beg_res.name, beg_asym, beg_seq,
            end_res.name, end_asym, end_seq,
            beg_cid, beg_rnum, beg_rins,
            end_cid, end_rnum, end_rins))

    sheet_range_data = []
    sheet_range = mmcif.MMCIFTable("struct_sheet_range", [
        "sheet_id", "id",
        "beg_label_comp_id",
        "beg_label_asym_id",
        "beg_label_seq_id",
        "end_label_comp_id",
        "end_label_asym_id",
        "end_label_seq_id",
        "symmetry",
        "beg_auth_asym_id",
        "beg_auth_seq_id",
        "pdbx_beg_PDB_ins_code",
        "end_auth_asym_id",
        "end_auth_seq_id",
        "pdbx_end_PDB_ins_code",
    ], sheet_range_data)

    def sheet_range_entry(sheet_id, count, beg_res, end_res, symmetry="1_555"):
        beg_asym, beg_seq = residue_info[beg_res]
        end_asym, end_seq = residue_info[end_res]
        beg_cid = beg_res.chain_id
        if beg_cid == ' ':
            beg_cid = '.'
        beg_rnum = beg_res.number
        beg_rins = beg_res.insertion_code
        if not beg_rins:
            beg_rins = '?'
        end_cid = end_res.chain_id
        if end_cid == ' ':
            end_cid = '.'
        end_rnum = end_res.number
        end_rins = end_res.insertion_code
        if not end_rins:
            end_rins = '?'
        sheet_range_data.append((
            sheet_id, count,
            beg_res.name, beg_asym, beg_seq,
            end_res.name, end_asym, end_seq,
            symmetry,
            beg_cid, beg_rnum, beg_rins,
            end_cid, end_rnum, end_rins))

    helix_count = 0
    strand_count = 0
    residues = best_m.residues
    ssids = residues.secondary_structure_ids
    last_ssid = 0
    beg_res = None
    end_res = None
    for r, ssid in zip(residues, ssids):
        if last_ssid == 0:
            beg_res = end_res = r
            last_ssid = ssid
        elif ssid == last_ssid:
            end_res = r
        else:
            if beg_res.is_helix:
                helix_count += 1
                struct_conf_entry('HELX%d' % helix_count, "HELX_P", beg_res, end_res)
            elif beg_res.is_strand:
                strand_count += 1
                sheet_range_entry('?', strand_count, beg_res, end_res)
            beg_res = end_res = r
            last_ssid = ssid
    if last_ssid:
        if beg_res.is_helix:
            helix_count += 1
            struct_conf_entry('HELX%d' % helix_count, "HELX_P", beg_res, end_res)
        elif beg_res.is_strand:
            strand_count += 1
            struct_conf_entry('STRN%d' % strand_count, "STRN_P", beg_res, end_res)

    if helix_count:
        struct_conf_type_data.append("HELX_P")

    struct_conf_data[:] = flattened(struct_conf_data)
    struct_conf.print(file, fixed_width=True)
    # struct_conf_type_data[:] = flattened(struct_conf_type_data)
    struct_conf_type.print(file, fixed_width=True)
    del struct_conf_data, struct_conf, struct_conf_type_data, struct_conf_type
    sheet_range_data[:] = flattened(sheet_range_data)
    sheet_range.print(file, fixed_width=True)
    del sheet_range_data, sheet_range

    _save_metadata(best_m, ['entity_src_gen', 'entity_src_nat'], file)
    _save_metadata(best_m, ['cell', 'symmetry'], file)
    _save_metadata(best_m, ['pdbx_struct_assembly', 'pdbx_struct_assembly_gen', 'pdbx_sruct_oper_list'], file)


def save_components(model, file):
    residues = model.residues
    unique_names = residues.unique_names
    names = None
    chem_comp_fields = ['id', 'type']

    old_chem_comp, = mmcif.get_mmcif_tables_from_metadata(model, ['chem_comp'])
    if not old_chem_comp:
        has_name = False
        existing_info = {}
    else:
        has_name = old_chem_comp.has_field('name')
        if has_name:
            chem_comp_fields.append('name')
        existing_info = old_chem_comp.mapping('id', chem_comp_fields[1:])

    new_values = []
    for n in unique_names:
        new_values.append(n)
        fields = existing_info.get(n, None)
        if fields is not None:
            new_values.extend(fields)
            continue
        # TODO: get type and description from residue template
        # fallback: fake values if no match above
        if names is None:
            names = residues.names.tolist()
        r = residues[names.index(n)]
        if r.polymer_type == r.PT_AMINO:
            rtype = 'peptide linking'
        elif r.polymer_type == r.PT_NUCLEIC:
            rtype = 'rna linking'
        else:
            rtype = 'non-polymer'
        new_values.append(rtype)
        if has_name:
            new_values.append('?')

    new_chem_comp = mmcif.MMCIFTable('chem_comp', chem_comp_fields, new_values)
    new_chem_comp.print(file, fixed_width=True)

    # TODO: chem_comp_atom
    # TODO: chem_comp_bond


if 0:
    # testing
    from chimerax.core.commands.open import open as open_cmd
    # pdb_id = '1a0m'  # alternate atom locations
    # pdb_id = '3fx2'  # old standby
    # pdb_id = '1mtx'  # NMR ensemble
    # pdb_id = '1ejg'  # atom_site_anisotrop
    # pdb_id = '5cd4'  # struct_conn: disulf, covale, metalc, hydrog
    pdb_id = '2adw'  # struct_conn: bond to alternate atom
    models = open_cmd(session, pdb_id)  # noqa
    # models = open_cmd(session, pdb_id, format='pdb')
    # save_mmcif(session, '%s.cif' % pdb_id, models)
    with open('%s.cif' % pdb_id, 'w') as file:
        save_structure(session, file, models, set())  # noqa
    raise SystemExit(-1)
