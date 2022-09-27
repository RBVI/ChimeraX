# distutils: language=c++
# cython: language_level=3, boundscheck=False, auto_pickle=False
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
from . import mmcif
from chimerax.core.utils import flattened
import platform
from chimerax import app_dirs
import sys

WRITER_VERSION = 'v9'  # TODO: update after any change

MMCIF_PREAMBLE = "#\\#CIF_1.1\n" "# mmCIF\n"

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
    'X': "UNK",
    'Y': "TYR",
    'Z': "GLX",
}
_standard_residues = set()  # filled in once at runtime


def _set_standard_residues():
    _standard_residues.update(_rna1to3.values())
    _standard_residues.update(_dna1to3.values())
    _standard_residues.update(_protein1to3.values())


def _same_chains(chain0, chain1):
    c0 = {c.chain_id: c.characters for c in chain0}
    c1 = {c.chain_id: c.characters for c in chain1}
    return c0 == c1


def write_mmcif(session, path, *, models=None, rel_model=None, selected_only=False, displayed_only=False, fixed_width=True, best_guess=False, all_coordsets=False, computed_sheets=False):
    from chimerax.atomic import Structure
    if models is None:
        models = session.models.list(type=Structure)
    else:
        models = [m for m in models if isinstance(m, Structure)]

    if not models:
        session.logger.info("no structures to save")
        return

    xforms = {}
    if rel_model is None:
        for m in models:
            p = m.scene_position
            if p.is_identity():
                xforms[m] = None
            else:
                xforms[m] = p
    else:
        inv = rel_model.scene_position.inverse()
        for m in models:
            if m.scene_position == rel_model.scene_position:
                xforms[m] = None
            else:
                xforms[m] = m.scene_position * inv

    if not _standard_residues:
        _set_standard_residues()

    file_per_model = "[NAME]" in path or "[ID]" in path
    if file_per_model:
        for m in models:
            used_data_names = set()
            file_name = path.replace("[ID]", m.id_string).replace("[NAME]", m.name)
            with open(file_name, 'w', encoding='utf-8', newline='\r\n') as f:
                f.write(MMCIF_PREAMBLE)
                save_structure(session, f, [m], [xforms[m]], used_data_names, selected_only, displayed_only, fixed_width, best_guess, all_coordsets, computed_sheets)
        return

    # Need to figure out which ChimeraX models should be grouped together
    # as mmCIF models.  Start with assumption that all models with the same
    # "parent" id (other than blank) are a nmr ensemble.
    grouped = {}
    for m in models:
        if m.id is None:
            group = None
        else:
            group = m.id[:-1]
        grouped.setdefault(group, []).append(m)

    # Go through grouped models and confirm they look like an NMR ensemble
    # This should catch fix for docking models and hierarchical models (IHM)
    is_ensemble = {}
    for g, models in grouped.items():
        if len(models) == 1 or g is None:
            is_ensemble[g] = False
            continue
        chains = models[0].chains
        is_ensemble[g] = all(_same_chains(chains, m.chains) for m in models[1:])

    used_data_names = set()
    with open(path, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(MMCIF_PREAMBLE)
        for g, models in grouped.items():
            if is_ensemble[g]:
                save_structure(session, f, models, [xforms[m] for m in models], used_data_names, selected_only, displayed_only, fixed_width, best_guess, all_coordsets, computed_sheets)
            else:
                for m in models:
                    save_structure(session, f, [m], [xforms[m]], used_data_names, selected_only, displayed_only, fixed_width, best_guess, all_coordsets, computed_sheets)


ChimeraX_audit_conform = mmcif.CIFTable(
    "audit_conform",
    [
        "dict_name",
        "dict_version",
        "dict_location",
    ], [
        "mmcif_pdbx.dic",
        "4.007",
        "http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic",
    ]
)

ChimeraX_audit_syntax_info = {
    "case_sensitive_flag": "Y",
    "fixed_width": "atom_site atom_site_anisotrop"
}

ChimeraX_citation_id = "chimerax"
ChimeraX_citation_info = {
    'title': "UCSF ChimeraX: Structure visualization for researchers, educators, and developers",
    'journal_abbrev': "Protein Sci.",
    'journal_volume': '30',
    'year': '2021',
    'page_first': '70',
    'page_last': '82',
    'journal_issue': '1',
    'pdbx_database_id_PubMed': '28710774',
    'pdbx_database_id_DOI': '10.1002/pro.3943',
}

ChimeraX_authors = (
    'Pettersen EF',
    'Goddard TD',
    'Huang CC',
    'Meng EC',
    'Couch GS',
    'Croll TI',
    'Morris JH',
    'Ferrin TE',
)

_CHAIN_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

system = platform.system()
if system == 'Darwin':
    system = 'macOS'  # TODO? Mac OS X (thru 10.7)/OS X thru 10.11/macOS
ChimeraX_software_info = {
    'name': '%s %s' % (app_dirs.appauthor, app_dirs.appname),
    'version': "%s/%s" % (app_dirs.version, WRITER_VERSION),
    'location': 'https://www.rbvi.ucsf.edu/chimerax/',
    'classification': 'model building',
    'os': system,
    'type': 'package',
    'citation_id': ChimeraX_citation_id,
}
del system


def _mmcif_chain_id(i):
    # want A..9, AA..99
    assert i > 0
    i -= 1
    num_chars = len(_CHAIN_CHARS)
    max = num_chars
    num_digits = 1
    while i >= max:
        i -= max
        num_digits += 1
        max = max * num_chars
    if i == 0:
        return _CHAIN_CHARS[0] * num_digits
    output = []
    num_chars = len(_CHAIN_CHARS)
    for d in range(num_digits):
        output.append(_CHAIN_CHARS[i % num_chars])
        i //= num_chars
    output.reverse()
    return ''.join(output)


def _save_metadata(model, categories, file, metadata):
    tables = mmcif.get_mmcif_tables_from_metadata(model, categories, metadata=metadata)
    printed = False
    for t in tables:
        if t is None:
            continue
        t.print(file, fixed_width=True)
        printed = True
    return printed


def save_structure(session, file, models, xforms, used_data_names, selected_only, displayed_only, fixed_width, best_guess, all_coordsets, computed_sheets):
    # save mmCIF data section for a structure
    # 'models' should only have more than one model if NMR ensemble
    # All 'models' should have the same metadata.
    # All 'models' should have the same number of atoms, but in PDB files
    # then often don't, so pick the model with the most atoms.
    #
    from chimerax.atomic import concatenate
    if len(models) == 1:
        best_m = models[0]
    else:
        # TODO: validate that the models are actually similar, ie.,
        # same number of chains which same chain ids and sequences,
        # and same kinds of HET residues
        tmp = list(models)
        tmp.sort(key=lambda m: m.num_atoms)
        best_m = tmp[-1]
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

    restricted = selected_only or displayed_only

    def nonblank_chars(name):
        return ''.join(ch for ch in name if not ch.isspace())

    print('data_%s' % nonblank_chars(name), file=file)
    print('#', file=file)

    best_metadata = best_m.metadata  # get once from C++ layer

    _save_metadata(best_m, ['entry'], file, best_metadata)

    ChimeraX_audit_conform.print(file=file, fixed_width=fixed_width)

    audit_syntax_info = ChimeraX_audit_syntax_info.copy()
    if not fixed_width:
        audit_syntax_info["fixed_width"] = ""
    tags, data = zip(*audit_syntax_info.items())
    audit_syntax = mmcif.CIFTable("audit_syntax", tags, list(data))
    audit_syntax.print(file=file, fixed_width=fixed_width)

    citation, citation_author, citation_editor = mmcif._add_citation(
            best_m, ChimeraX_citation_id, ChimeraX_citation_info,
            ChimeraX_authors, metadata=best_metadata, return_existing=True)
    software = mmcif._add_software(
            best_m, ChimeraX_software_info['name'], ChimeraX_software_info,
            metadata=best_metadata, return_existing=True)
    citation.print(file, fixed_width=fixed_width)
    citation_author.print(file, fixed_width=fixed_width)
    if citation_editor is not None:
        citation_editor.print(file, fixed_width=fixed_width)
    software.print(file, fixed_width=fixed_width)
    del citation, citation_author, citation_editor, software

    save_components(best_m, file, best_metadata, fixed_width)

    _save_metadata(best_m, ['exptl'], file, best_metadata)

    from chimerax.atomic import Residue
    old_entity, old_asym = mmcif.get_mmcif_tables_from_metadata(
        best_m, ['entity', 'struct_asym'], metadata=best_metadata)
    try:
        if not old_entity or not old_asym:
            raise ValueError
        old_mmcif_chain_to_entity = old_asym.mapping('id', 'entity_id')
        old_entity_to_description = old_entity.mapping('id', 'pdbx_description')
    except ValueError:
        old_mmcif_chain_to_entity = {}
        old_entity_to_description = {}

    from collections import OrderedDict
    entity_info = {}     # { entity_id: (type, pdbx_description) }
    asym_info = {}       # { auth_chain_id: (entity_id, label_asym_id) }
    het_asym_info = {}   # { mmcif_chain_id: (entity_id, label_asym_id) }
    poly_info = []       # [(entity_id, type, one-letter-seq)]
    poly_seq_info = []   # [(entity_id, num, mon_id)]
    pdbx_poly_info = []  # [(entity_id, asym_id, mon_id, seq_id, pdb_strand_id, auth_seq_num, pdb_ins_code)]
    residue_info = {}    # { residue: (label_asym_id, label_seq_id) }

    skipped_sequence_info = False

    seq_entities = OrderedDict()   # { chain.characters : (entity_id, _1to3, [chains]) }
    for c in best_m.chains:
        chars = c.characters
        if chars in seq_entities:
            eid, _1to3, chains = seq_entities[chars]
            chains.append(c)
        else:
            mcid = c.existing_residues[0].mmcif_chain_id
            try:
                descrip = old_entity_to_description[old_mmcif_chain_to_entity[mcid]]
            except KeyError:
                descrip = '?'
            eid = len(entity_info) + 1
            entity_info[eid] = ('polymer', descrip)
            names = set(c.existing_residues.names)
            nstd = 'yes' if names.difference(_standard_residues) else 'no'
            # _1to3 is reverse map to handle missing residues
            if not best_guess and not c.from_seqres:
                skipped_sequence_info = True
                _1to3 = None
            else:
                if c.polymer_type == Residue.PT_AMINO:
                    _1to3 = _protein1to3
                    poly_info.append((eid, nstd, 'polypeptide(L)', chars))  # TODO: or polypeptide(D)
                elif names.isdisjoint(set(_rna1to3)):
                    # must be DNA
                    _1to3 = _dna1to3
                    poly_info.append((eid, nstd, 'polyribonucleotide', chars))
                else:
                    # must be RNA
                    _1to3 = _rna1to3
                    poly_info.append((eid, nstd, 'polydeoxyribonucleotide', chars))
            seq_entities[chars] = (eid, _1to3, [c])

    if skipped_sequence_info:
        session.logger.warning("Not saving entity_poly_seq for non-authoritative sequences")

    # use all chains of the same entity to figure out what the sequence's residues are named
    pdbx_poly_tmp = {}
    for chars, (eid, _1to3, chains) in seq_entities.items():
        chains = [c for c in chains if c.from_seqres]
        pdbx_poly_tmp[eid] = []
        for seq_id, ch, residues in zip(range(1, sys.maxsize), chars, zip(*(c.residues for c in chains))):
            label_seq_id = str(seq_id)
            for r in residues:
                if r is not None:
                    name = r.name
                    seq_num = r.number
                    ins_code = r.insertion_code
                    if not ins_code:
                        ins_code = '.'
                    break
            else:
                name = _1to3.get(ch, 'UNK')
                seq_num = '?'
                ins_code = '.'
            poly_seq_info.append((eid, label_seq_id, name))
            pdbx_poly_tmp[eid].append((name, label_seq_id, seq_num, ins_code))

    existing_mmcif_chain_ids = set(best_m.residues.mmcif_chain_ids)
    used_mmcif_chain_ids = set()
    last_asym_id = 0

    def get_asym_id(want_id):
        nonlocal existing_mmcif_chain_ids, used_mmcif_chain_ids, last_asym_id
        if want_id not in used_mmcif_chain_ids:
            used_mmcif_chain_ids.add(want_id)
            return want_id
        while True:
            last_asym_id += 1
            asym_id = _mmcif_chain_id(last_asym_id)
            if asym_id in existing_mmcif_chain_ids:
                continue
            used_mmcif_chain_ids.add(asym_id)
            return asym_id

    # assign label_asym_id's to each chain
    for c in best_m.chains:
        mcid = c.existing_residues[0].mmcif_chain_id
        label_asym_id = get_asym_id(mcid)
        chars = c.characters
        chain_id = c.chain_id
        eid, _1to3, _ = seq_entities[chars]
        asym_info[(chain_id, chars)] = (label_asym_id, eid)

        tmp = pdbx_poly_tmp[eid]
        for name, label_seq_id, seq_num, ins_code in tmp:
            pdbx_poly_info.append((eid, label_asym_id, name, label_seq_id, chain_id, seq_num, ins_code))
    del pdbx_poly_tmp

    het_entities = {}   # { het_name: { 'entity': entity_id, chain: (label_entity_id, label_asym_id) } }
    het_residues = concatenate(
        [m.residues.filter(m.residues.polymer_types == Residue.PT_NONE) for m in models])
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
        label_asym_id = get_asym_id(mcid)
        het_asym_info[mcid] = (label_asym_id, eid)
        het_entities[n][mcid] = (eid, label_asym_id)

    entity = mmcif.CIFTable('entity', ['id', 'type', 'pdbx_description'], flattened(entity_info.items()))
    entity.print(file, fixed_width=fixed_width)
    entity_poly = mmcif.CIFTable('entity_poly', ['entity_id', 'nstd_monomer', 'type', 'pdbx_seq_one_letter_code_can'], flattened(poly_info))
    entity_poly.print(file, fixed_width=fixed_width)
    entity_poly_seq = mmcif.CIFTable('entity_poly_seq', ['entity_id', 'num', 'mon_id'], flattened(poly_seq_info))
    entity_poly_seq.print(file, fixed_width=fixed_width)
    import itertools
    struct_asym = mmcif.CIFTable(
        'struct_asym', ['id', 'entity_id'],
        flattened(itertools.chain(asym_info.values(), het_asym_info.values())))
    struct_asym.print(file, fixed_width=fixed_width)
    pdbx_poly_seq = mmcif.CIFTable('pdbx_poly_seq_scheme', ['entity_id', 'asym_id', 'mon_id', 'seq_id', 'pdb_strand_id', 'pdb_seq_num', 'pdb_ins_code'], flattened(pdbx_poly_info))
    pdbx_poly_seq.print(file, fixed_width=fixed_width)
    del entity, entity_poly_seq, pdbx_poly_seq, struct_asym

    elements = list(set(best_m.atoms.elements))
    elements.sort(key=lambda e: e.number)
    atom_type_data = [e.name for e in elements]
    atom_type = mmcif.CIFTable("atom_type", ["symbol"], atom_type_data)
    atom_type.print(file, fixed_width=fixed_width)
    del atom_type_data, atom_type

    atom_site_data = []
    atom_site = mmcif.CIFTable("atom_site", [
        'group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_alt_id',
        'label_comp_id', 'label_asym_id', 'label_entity_id', 'label_seq_id',
        'Cartn_x', 'Cartn_y', 'Cartn_z',
        'auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code',
        'occupancy', 'B_iso_or_equiv', 'pdbx_PDB_model_num'
    ], atom_site_data)
    atom_site_anisotrop_data = []
    atom_site_anisotrop = mmcif.CIFTable("atom_site_anisotrop", [
        'id', 'type_symbol',
        'U[1][1]', 'U[2][2]', 'U[3][3]',
        'U[1][2]', 'U[1][3]', 'U[2][3]',
    ], atom_site_anisotrop_data)
    serial_num = 0

    def atom_site_residue(residue, seq_id, asym_id, entity_id, model_num, xform):
        nonlocal serial_num, residue_info, atom_site_data, atom_site_anisotrop_data
        residue_info[residue] = (asym_id, seq_id)
        atoms = residue.atoms
        rname = residue.name
        cid = residue.chain_id
        if cid == ' ':
            cid = '.'
        rnum = residue.number
        rins = residue.insertion_code
        if not rins:
            rins = '?'
        if rname in _standard_residues:
            group = 'ATOM'
        else:
            group = 'HETATM'
        for atom in atoms:
            if restricted:
                if selected_only and not atom.selected:
                    continue
                if displayed_only and not atom.display:
                    continue
            elem = atom.element.name
            aname = atom.name
            original_alt_loc = atom.alt_loc
            for alt_loc in atom.alt_locs or '.':
                if alt_loc != '.':
                    atom.set_alt_loc(alt_loc, False)
                coord = atom.coord
                if xform is not None:
                    coord = xform * coord
                xyz = ['%.3f' % f for f in coord]
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
            if alt_loc != '.':
                atom.set_alt_loc(original_alt_loc, False)

    def do_atom_site_model(m, xform, model_num):
        residues = m.residues
        het_residues = residues.filter(residues.polymer_types == Residue.PT_NONE)
        for c in m.chains:
            chain_id = c.chain_id
            chars = c.characters
            asym_id, entity_id = asym_info[(chain_id, chars)]
            for seq_id, r in zip(range(1, sys.maxsize), c.residues):
                if r is None:
                    continue
                atom_site_residue(r, seq_id, asym_id, entity_id, model_num, xform)
            chain_het = het_residues.filter(het_residues.chain_ids == chain_id)
            het_residues -= chain_het
            for r in chain_het:
                asym_id, entity_id = het_asym_info[r.mmcif_chain_id]
                atom_site_residue(r, '.', asym_id, entity_id, model_num, xform)
        for r in het_residues:
            asym_id, entity_id = het_asym_info[r.mmcif_chain_id]
            atom_site_residue(r, '.', asym_id, entity_id, model_num, xform)

    if all_coordsets and len(models) == 1 and best_m.num_coordsets > 1:
        xform = xforms[0]
        for model_num in range(1, best_m.num_coordsets + 1):
            do_atom_site_model(best_m, xform, model_num)
    else:
        for m, xform, model_num in zip(models, xforms, range(1, sys.maxsize)):
            do_atom_site_model(m, xform, model_num)

    atom_site_data[:] = flattened(atom_site_data)
    atom_site.print(file, fixed_width=fixed_width)
    atom_site_anisotrop_data[:] = flattened(atom_site_anisotrop_data)
    atom_site_anisotrop.print(file, fixed_width=fixed_width)
    # del atom_site_data, atom_site, atom_site_anisotrop_data, atom_site_anisotrop  # not in cython

    struct_conn_data = []
    struct_conn = mmcif.CIFTable("struct_conn", [
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
    struct_conn_type = mmcif.CIFTable("struct_conn_type", [
        "id",
    ], struct_conn_type_data)

    def struct_conn_bond(tag, count, b, a0, a1):
        nonlocal struct_conn_data
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
            struct_conn_data.append((
                f"{tag}{count}", tag,
                a0.name, alt_loc0, r0_asym, r0_seq, cid0, rnum0, rins0, r0.name, "1_555",
                a1.name, alt_loc1, r1_asym, r1_seq, cid1, rnum1, rins1, r1.name, "1_555",
                dist))
        if original_alt_loc0 != ' ':
            a0.set_alt_loc(original_alt_loc0, False)
        if original_alt_loc1 != ' ':
            a1.set_alt_loc(original_alt_loc1, False)

    disulfide, covalent = mmcif.non_standard_bonds(best_m.bonds, selected_only, displayed_only)

    if disulfide:
        struct_conn_type_data.append('disulf')
        for count, (b, a0, a1) in enumerate(zip(disulfide, *disulfide.atoms), start=1):
            struct_conn_bond('disulf', count, b, a0, a1)

    # metal coordination bonds
    # assume intra-residue metal coordination bonds are handled by residue template
    pbg = best_m.pseudobond_group(best_m.PBG_METAL_COORDINATION, create_type=None)
    if pbg:
        bonds = pbg.pseudobonds
        if len(bonds) > 0:
            count = 0
            for b, a0, a1 in zip(bonds, *bonds.atoms):
                r0 = a0.residue
                r1 = a1.residue
                if r0 == r1:
                    continue
                if restricted:
                    if selected_only and (not a0.selected or not a1.selected):
                        continue
                    if displayed_only and (not a0.display or not a1.display):
                        continue
                count += 1
                struct_conn_bond('metalc', count, b, a0, a1)
            if count > 0:
                struct_conn_type_data.append('metalc')

    # hydrogen bonds
    pbg = best_m.pseudobond_group(best_m.PBG_HYDROGEN_BONDS, create_type=None)
    if pbg:
        bonds = pbg.pseudobonds
        if len(bonds) > 0:
            count = 0
            for b, a0, a1 in zip(bonds, *bonds.atoms):
                if restricted:
                    if selected_only and (not a0.selected or not a1.selected):
                        continue
                    if displayed_only and (not a0.display or not a1.display):
                        continue
                count += 1
                struct_conn_bond('hydrog', count, b, a0, a1)
            if count > 0:
                struct_conn_type_data.append('hydrog')

    if covalent:
        struct_conn_type_data.append('covale')
        for count, (b, a0, a1) in enumerate(zip(covalent, *covalent.atoms), start=1):
            struct_conn_bond('covale', count, b, a0, a1)

    struct_conn_data[:] = flattened(struct_conn_data)
    struct_conn.print(file, fixed_width=fixed_width)
    # struct_conn_type_data[:] = flattened(struct_conn_type_data)
    struct_conn_type.print(file, fixed_width=fixed_width)
    # del struct_conn_data, struct_conn, struct_conn_type_data, struct_conn_type  # not in cython

    # struct_conf
    struct_conf_data = []
    struct_conf = mmcif.CIFTable("struct_conf", [
        "id", "conf_type_id",
        "beg_label_comp_id",
        "beg_label_asym_id",
        "beg_label_seq_id",
        "end_label_comp_id",
        "end_label_asym_id",
        "end_label_seq_id",
        "beg_auth_comp_id",
        "beg_auth_asym_id",
        "beg_auth_seq_id",
        "pdbx_beg_PDB_ins_code",
        "end_auth_comp_id",
        "end_auth_asym_id",
        "end_auth_seq_id",
        "pdbx_end_PDB_ins_code",
    ], struct_conf_data)

    struct_conf_type_data = []
    struct_conf_type = mmcif.CIFTable("struct_conf_type", [
        "id"
    ], struct_conf_type_data)

    def struct_conf_entry(id, ctype, beg_res, end_res):
        nonlocal struct_conf_data
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
            beg_res.name, beg_cid, beg_rnum, beg_rins,
            end_res.name, end_cid, end_rnum, end_rins))

    sheet_data = []
    sheet = mmcif.CIFTable("struct_sheet", [
        "id",
        "number_strands"
    ], sheet_data)

    def sheet_entry(id, count):
        sheet_data.append((id, count))

    sheet_order_data = []
    sheet_order = mmcif.CIFTable("struct_sheet_order", [
        "sheet_id",
        "range_id_1",
        "range_id_2",
        "sense",
    ], sheet_order_data)

    def sheet_order_entry(sheet_id, first, second, sense):
        sheet_order_data.append((sheet_id, first, second, sense))

    sheet_range_data = []
    sheet_range = mmcif.CIFTable("struct_sheet_range", [
        "sheet_id", "id",
        "beg_label_comp_id",
        "beg_label_asym_id",
        "beg_label_seq_id",
        "end_label_comp_id",
        "end_label_asym_id",
        "end_label_seq_id",
        "beg_auth_comp_id",
        "beg_auth_asym_id",
        "beg_auth_seq_id",
        "pdbx_beg_PDB_ins_code",
        "end_auth_comp_id",
        "end_auth_asym_id",
        "end_auth_seq_id",
        "pdbx_end_PDB_ins_code",
    ], sheet_range_data)

    def sheet_range_entry(sheet_id, strand_num, beg_res, end_res):
        nonlocal sheet_range_data
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
            sheet_id, strand_num,
            beg_res.name, beg_asym, beg_seq,
            end_res.name, end_asym, end_seq,
            beg_res.name, beg_cid, beg_rnum, beg_rins,
            end_res.name, end_cid, end_rnum, end_rins))

    helix_count = 0
    strand_count = 0
    residues = best_m.residues
    # use ChimeraX's secondary structure information
    # which does not include sheets
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
            skip = False
            if skip:
                pass
            elif beg_res.is_helix:
                helix_count += 1
                struct_conf_entry('HELX%d' % helix_count, "HELX_P", beg_res, end_res)
            elif not computed_sheets and beg_res.is_strand:
                strand_count += 1
                sheet_range_entry('?', strand_count, beg_res, end_res)
            beg_res = end_res = r
            last_ssid = ssid
    if last_ssid:
        skip = False
        if skip:
            pass
        elif beg_res.is_helix:
            helix_count += 1
            struct_conf_entry('HELX%d' % helix_count, "HELX_P", beg_res, end_res)
        elif not computed_sheets and beg_res.is_strand:
            strand_count += 1
            sheet_range_entry('?', strand_count, beg_res, end_res)
    if computed_sheets:
        # "best guess" case - guess at sheet information by running DSSP
        with best_m.suppress_ss_change_notifications():
            from chimerax.dssp import compute_ss
            ss_data = compute_ss(best_m, return_values=True)
            # helix_info = ss_data["helix_info"]
            # if helix_info:
            #     for (beg_res, end_res), htype in helix_info:
            #         # Helix type is always HELX_P in current PDB entries
            #         helix_count += 1
            #         struct_conf_entry('HELX%d' % helix_count, "HELX_P", beg_res, end_res)
            ss_sheets = ss_data["sheets"]
            if ss_sheets:
                ss_strands = ss_data["strands"]
                strand_map = {}
                for i, strands in enumerate(ss_sheets, start=1):
                    sheet_id = f'S{i}'
                    sheet_entry(sheet_id, len(strands))
                    for j, strand in enumerate(strands, start=1):
                        beg_res, end_res = ss_strands[strand]
                        sheet_range_entry(sheet_id, j, beg_res, end_res)
                        strand_map[strand] = (sheet_id, j)
                ss_parallel = ss_data["strands_parallel"]
                for (first, second) in ss_parallel:
                    parallel = 'parallel' if ss_parallel[(first, second)] else 'anti-parallel'
                    st1 = strand_map[first]
                    st2 = strand_map[second]
                    if st1[0] != st2[0]:
                        print("old strand order:", st1, st2)
                        continue
                    sheet_order_entry(st1[0], st1[1], st2[1], parallel)

    struct_conf_data[:] = flattened(struct_conf_data)
    struct_conf.print(file, fixed_width=fixed_width)
    # struct_conf_type_data[:] = flattened(struct_conf_type_data)
    struct_conf_type.print(file, fixed_width=fixed_width)
    # del struct_conf_data, struct_conf, struct_conf_type_data, struct_conf_type  # not in cython
    sheet_data[:] = flattened(sheet_data)
    sheet.print(file, fixed_width=fixed_width)
    sheet_order_data[:] = flattened(sheet_order_data)
    sheet_order.print(file, fixed_width=fixed_width)
    sheet_range_data[:] = flattened(sheet_range_data)
    sheet_range.print(file, fixed_width=fixed_width)
    # del sheet_range_data, sheet_range  # not in cython

    _save_metadata(best_m, ['entity_src_gen', 'entity_src_nat'], file, best_metadata)
    _save_metadata(best_m, ['cell', 'symmetry'], file, best_metadata)
    _save_metadata(best_m, ['pdbx_struct_assembly', 'pdbx_struct_assembly_gen', 'pdbx_struct_oper_list'], file, best_metadata)


def save_components(model, file, metadata, fixed_width):
    residues = model.residues
    unique_names = residues.unique_names
    names = None
    chem_comp_fields = ['id', 'type']

    old_chem_comp, = mmcif.get_mmcif_tables_from_metadata(model, ['chem_comp'], metadata=metadata)
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

    new_chem_comp = mmcif.CIFTable('chem_comp', chem_comp_fields, new_values)
    new_chem_comp.print(file, fixed_width=fixed_width)

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
        save_structure(session, file, models, set(), False, False, False)  # noqa
    raise SystemExit(-1)
