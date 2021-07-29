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

# -----------------------------------------------------------------------------
# Fetch additional information about PDB BLAST hits using the PDB REST web
# services.  Offer descriptions of all chains, names of molecular components
# and ligands, publication date, literature reference, number of residues...
#

query_template = """{
  entries(entry_ids: [%s]) {
    rcsb_id
    audit_author {
        name
    }
    exptl {
        method
    }
    nonpolymer_entities {
        nonpolymer_comp {
            rcsb_id
            rcsb_chem_comp_descriptor {
                SMILES
            }
            chem_comp {
                formula
                formula_weight
                name
            }
        }
    }
    pdbx_database_status {
        status_code
    }
    pdbx_database_PDB_obs_spr {
        replace_pdb_id
    }
    polymer_entities {
        entity_poly {
            pdbx_strand_id
            rcsb_sample_sequence_length
        }
        entity_src_gen {
            pdbx_gene_src_scientific_name
        }
        entity_src_nat {
            pdbx_organism_scientific
            species
        }
        rcsb_polymer_entity {
            formula_weight
            pdbx_description
        }
        rcsb_polymer_entity_container_identifiers {
            asym_ids
            auth_asym_ids
            reference_sequence_identifiers {
                database_name
                database_accession
            }
        }
    }
    rcsb_accession_info {
        deposit_date
        initial_release_date
        revision_date
    }
    rcsb_entry_info {
        deposited_atom_count
        deposited_polymer_monomer_count
        polymer_entity_count
        resolution_combined
    }
    rcsb_primary_citation {
        pdbx_database_id_PubMed
        rcsb_authors
    }
    struct {
        title
    }
    struct_keywords {
        pdbx_keywords
    }
  }
}"""

entry_attr_name_mapping = [
    ("structureId", ['rcsb_id']),
    ("title", ['struct', 'title']),
    ("expMethod", ['exptl', 'method']),
    ("resolution", ['rcsb_entry_info', 'resolution_combined']),
    ("keywords", ['struct_keywords', 'pdbx_keywords']),
    ("nr_entities", ['rcsb_entry_info', 'polymer_entity_count']),
    ("nr_residues", ['rcsb_entry_info', 'deposited_polymer_monomer_count']),
    ("nr_atoms", ['rcsb_entry_info', 'deposited_atom_count']),
    ("deposition_date", ['rcsb_accession_info', 'deposit_date']),
    ("publish_date", ['rcsb_accession_info', 'initial_release_date']),
    ("revision_date", ['rcsb_accession_info', 'revision_date']),
    ("structure_authors", ['audit_author', 'name']),
    ("pubmedId", ['rcsb_primary_citation', 'pdbx_database_id_PubMed']),
    ("citation_authors", ['rcsb_primary_citation', 'rcsb_authors']),
    ("status", ['pdbx_database_status', 'status_code']),
    ("replaces", ['pdbx_database_PDB_obs_spr', 'replace_pdb_id']),
    ('ligandFormulas', ['nonpolymer_entities', 'nonpolymer_comp', 'chem_comp', 'formula']),
    ('ligandNames', ['nonpolymer_entities', 'nonpolymer_comp', 'chem_comp', 'name']),
    ('ligandSmiles', ['nonpolymer_entities', 'nonpolymer_comp', 'rcsb_chem_comp_descriptor', 'SMILES']),
    ('ligandSymbols', ['nonpolymer_entities', 'nonpolymer_comp', 'rcsb_id']),
    ('ligandWeights', ['nonpolymer_entities', 'nonpolymer_comp', 'chem_comp', 'formula_weight'])
]

chain_attr_name_mapping = [
    ('chain_names', False, ['rcsb_polymer_entity', 'pdbx_description']),
    ('chain_copies', True, ["entity_poly", "pdbx_strand_id", lambda v: len(v.split(','))]),
    ('npolymers', False, [len]),
    ('chain_residues', True, ["entity_poly", "rcsb_sample_sequence_length"]),
    ('chain_species', True, [(["entity_src_gen", "pdbx_gene_src_scientific_name"],
        ["entity_src_nat", (["pdbx_organism_scientific"], ["species"])])]),
    ('chain_sequence_id', True, ["rcsb_polymer_entity_container_identifiers",
        "reference_sequence_identifiers", {("database_name", "UniProt"): ["database_accession"]}]),
    ('chain_weight',  True, ['rcsb_polymer_entity', 'formula_weight'])
]

def fetch_pdb_info(session, entry_chain_list):
    # Can't just sub in a list, since Python uses single quotes around strings by default
    query = query_template % ",".join(['"%s"' % entry_chain.split('_')[0]
        for entry_chain in entry_chain_list])
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError
    try:
        req = Request("https://data.rcsb.org/graphql", data=query.encode('utf-8'), headers={
             "Content-Type": "application/graphql"
        })
        f = urlopen(req)
        data = f.read()
        f.close()
    except (URLError, HTTPError) as e:
        session.logger.warning("Fetching BLAST PDB info failed: %s" % str(e))
        return {}
    else:
        data = data.decode('utf-8').replace(':null', ':None')
        info = eval(data)
    if 'errors' in info:
        session.logger.warning("Fetching BLAST PDB info had errors: %s" % info['errors'])
    by_entry = {}
    for entry_data in info['data']['entries']:
        by_entry[entry_data['rcsb_id']] = entry_data
    pdb_info = {}
    for info_key in entry_chain_list:
        entry, chain = info_key.split('_')
        pdb_info[info_key] = hits = {}
        if entry not in by_entry:
            continue
        for attr_name, mmcif_keys in entry_attr_name_mapping:
            val = get_val(by_entry[entry], mmcif_keys)
            hits[attr_name] = val
        all_polys = by_entry[entry]['polymer_entities']
        for poly in all_polys:
            try:
                ids = poly['rcsb_polymer_entity_container_identifiers']
            except KeyError:
                continue
            if ids is None:
                continue
            try:
                auth_ids = ids['auth_asym_ids']
            except KeyError:
                continue
            if auth_ids and chain in auth_ids:
                break
        else:
            poly = None
        for attr_name, per_chain, mmcif_keys in chain_attr_name_mapping:
            if per_chain:
                val = None if poly is None else get_val(poly, mmcif_keys)
            else:
                # value across all chains, not just our own
                if all_polys is None:
                    val = None
                elif len(mmcif_keys) == 1 and callable(mmcif_keys[0]):
                    val = mmcif_keys[0](all_polys)
                else:
                    val = []
                    for p in all_polys:
                        pval = get_val(p, mmcif_keys)
                        auth_ids = get_val(p, ['rcsb_polymer_entity_container_identifiers', 'auth_asym_ids'])
                        if pval is not None and auth_ids is not None:
                            val.append("%s: %s" % (",".join(auth_ids), pval))
                    if not val:
                        val = None
            hits[attr_name] = val
    return pdb_info

def get_val(init_val, mmcif_keys):
    val = init_val
    for mmcif_key in mmcif_keys:
        if callable(mmcif_key):
            val = mmcif_key(val)
            break
        if type(mmcif_key) == tuple:
            for subkey in mmcif_key:
                subval = get_val(val, subkey)
                if subval is not None:
                    val = subval
                    break
            else:
                val = None
        elif type(mmcif_key) == dict:
            vals = []
            for condition, val_key in mmcif_key.items():
                cond_key, match_val = condition
                for subval in val:
                    cond_val = get_val(subval, [cond_key])
                    if cond_val == match_val:
                        final_val = get_val(subval, val_key)
                        if final_val is not None:
                            vals.append(final_val)
            if vals:
                val = vals
            else:
                val = None
        elif type(val) == list:
            vals = []
            for subval in val:
                try:
                    subsubval = subval[mmcif_key]
                    if subsubval is not None:
                        vals.append(subsubval)
                except KeyError:
                    continue
            if vals:
                val = vals
            else:
                val = None
        else:
            try:
                val = val[mmcif_key]
            except KeyError:
                val = None
        if val is None:
            break
    return val

