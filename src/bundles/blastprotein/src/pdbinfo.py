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
_polymer_info = {}
_non_polymer_info = {}
class PDB_Entry_Info:

    columns = (
        ("Authors", 'citation_authors', {'anchor':"nw"},
         'PDB authors of deposited structure'),
        ("Date", 'deposition_date', {'anchor':"n"},
         'PDB structure publication date'),
        ("Method", 'expMethod', {'anchor':"n"},
         'PDB experiment type used\nfor structure determination'),
        ("PubMed", 'pubmedId', {'anchor':"n"},
         'PDB PubMed literature reference for structure'),
        ("Resolution", 'resolution', {'anchor':"n"},
         'PDB resolution in Angstroms'),
        ("Title", 'title', {'anchor':"nw"}, 'PDB entry title'),
        ("Total atoms", 'nr_atoms', {'anchor':"n"},
         'PDB total number of atoms\nin structure (all chains)'),
        ("Total residues", 'nr_residues', {'anchor':"n"},
         'PDB total number of residues\nin structure (all chains)'),
        )

    def fetch_info(self, session, pdb_chain_ids):

        global _polymer_info, _non_polymer_info
        session.logger.status("Fetching PDB entry info for hits")
        _polymer_info.clear()
        _non_polymer_info.clear()
        if len(pdb_chain_ids) == 0:
            return {}

        pdb_ids = [id.split('_')[0].upper() for id in pdb_chain_ids]

        # Fetch PDB entry info
        mmcif_data = fetch_from_pdb(session, 'entry/%s', set(pdb_ids))
        if mmcif_data is None:
            session.logger.warning("Could not fetch overall PDB entry info corresponding to BLAST results")
            return {}
        for id, info in mmcif_data.items():
            try:
                id_info = info["rcsb_entry_container_identifiers"]
                _polymer_info[id] = id_info["polymer_entity_ids"]
                _non_polymer_info[id] = id_info.get("non_polymer_entity_ids", [])
            except KeyError:
                _polymer_info[id] = []
                _non_polymer_info[id] = []
        pis = [self.pdb_info(session, id, info) for id, info in mmcif_data.items()]
        da = dict((pi.structureId.lower(), self.entry_properties(pi))
                  for pi in pis)
        pimap = dict((pcid, da.get(pdb_id.lower(),{}))
                     for pdb_id, pcid in zip(pdb_ids, pdb_chain_ids))
        session.logger.status("Done fetching PDB entry info for hits")
        return pimap

    def pdb_info(self, session, pdb_id, info):
        attr_name_mapping = [
            ("structureId", ['entry', 'id']),
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
            ("pubmedId", ['rcsb_primary_citation', 'pdbx_database_id_pub_med']),
            ("citation_authors", ['rcsb_primary_citation', 'rcsb_authors']),
            ("status", ['pdbx_database_status', 'status_code']),
            ("replaces", ['pdbx_database_pdbobs_spr', 'replace_pdb_id'])
        ]

        pe = PDB_Entry()
        for attr_name, lookup_info in attr_name_mapping:
            key1, key2 = lookup_info
            try:
                table_info = info[key1]
            except KeyError:
                #session.logger.warning("Could not find '%s' table in %s entry information" % (key1, pdb_id))
                continue
            if type(table_info) == dict:
                try:
                    value = table_info[key2]
                except KeyError:
                    #session.logger.info("%s item missing from '%s' table in %s entry information"
                    #    % (key2, key1, pdb_id))
                    continue
                if value and isinstance(value, list) and isinstance(value[0], str):
                    value = ", ".join(value)
            else:
                values = []
                for table in table_info:
                    try:
                        values.append(table[key2])
                    except KeyError:
                        continue
                if not values:
                    #session.logger.info("%s item missing from all '%s' tables in %s entry information"
                    #    % (key2, key1, pdb_id))
                    continue
                value = ", ".join(values)
            setattr(pe, attr_name, value)
        if pe.structureId is None:
            session.logger.warning("Could not fetch entry information for %s" % pdb_id)
            pe.structureId = pdb_id
        return pe

    def entry_properties(self, pe):

        cnames = tuple(c[1] for c in self.columns)
        amap = dict((n,getattr(pe,n)) for n in cnames if hasattr(pe,n))
        return amap

# -----------------------------------------------------------------------------
#
class PDB_Entry:

    def __init__(self):

        self.structureId = None         # "4HHB"
        self.title = None               # "THE CRYSTAL STRUCTURE OF HUMAN DEOXYHAEMOGLOBIN AT 1.74 ANGSTROMS RESOLUTION"
        self.expMethod = None           # "X-RAY DIFFRACTION"
        self.resolution = None          # "1.74"
        self.keywords = None            # "OXYGEN TRANSPORT"
        self.nr_entities = None         # "4"
        self.nr_residues = None         # "574"
        self.nr_atoms = None            # "4779"
        self.publish_date = None	# "1984-03-07"
        self.revision_date = None	# "1984-07-17"
        self.structure_authors = None	# "Fermi, G., Perutz, M.F."
        self.pubmedId = None            # "6726807"
        self.citation_authors = None	# "Fermi, G., Perutz, M.F., Shaanan, B., Fourme, R."
        self.status = None		# "CURRENT"
        self.replaces = None            # "1HHB"

# -----------------------------------------------------------------------------
#
class PDB_Chain_Info:

    columns = (
        ('Chain names', 'chain_names', {'anchor':"nw"}, 'PDB chain names'),
        ('Copies', 'chain_copies', {'anchor':"n"},
         'PDB number of copies\nof matched chain'),
        ("Polymers", 'npolymers', {'anchor':"n"},
         'PDB number of different\npolymers in structure'),
        ('Residues', 'chain_residues', {'anchor':"n"},
         'PDB number of residues\nin matched chain'),
        ('Species', 'chain_species', {'anchor':"n"}, 'PDB chain taxonomy'),
        ('UniProt', 'chain_sequence_id', {'anchor':"n"},
         'Uniprot sequence identifier\nof matched PDB chain'),
        ('Weight', 'chain_weight',  {'anchor':"n"},
         'PDB molecular weight (Daltons)\nof matched chain'),
        )

    def fetch_info(self, session, pdb_chain_ids):

        if len(pdb_chain_ids) == 0:
            return {}

        # The tool presents some data that is based on all chains of the structure,
        # so look up all polymers in the entries, leveraging data cached from the
        # previous entry lookup
        # Also, Blast uses author chain ID whereas RSCB web service uses 'canonical'
        # chain ID, so make a mapping from one to the other
        session.logger.status("Fetching PDB chain mapping info for hits")
        polymer_data = {}
        chainID_mapping = {}
        global _polymer_info
        for i, item in enumerate(_polymer_info.items()):
            session.logger.status("%d/%d" % (i, len(_polymer_info)), secondary=True)
            entry_code, entity_ids = item
            polymer_data[entry_code] = per_entity_data = {}
            chainID_mapping[entry_code] = cid_mapping = {}
            for entity_id in entity_ids:
                per_entity_data[entity_id] = edata = fetch_from_pdb(session,
                    'polymer_entity/%s/%s', [(entry_code, entity_id)])[(entry_code, entity_id)]
                id_data = edata['rcsb_polymer_entity_container_identifiers']
                canon = id_data['asym_ids']
                auth = id_data['auth_asym_ids']
                cid_mapping.update(dict(zip(auth, canon)))

        session.logger.status("Fetching PDB chain entity info for hits")
        entry_chain_data = {}
        for i, pcid in enumerate(pdb_chain_ids):
            session.logger.status("%d/%d" % (i, len(pdb_chain_ids)), secondary=True)
            entry, auth_cid = pcid.split('_')
            try:
                key = (entry, chainID_mapping[entry][auth_cid])
            except KeyError:
                session.logger.warning("Chain ID mapping failed for %s" % pcid)
                chain_data = {}
            else:
                chain_data = fetch_from_pdb(session, 'polymer_entity_instance/%s/%s', [key])[key]
            entry_chain_data[pcid] = chain_data
        session.logger.status("", secondary=True)

        if entry_chain_data is None:
            # TODO: Warn that fetch failed.
            return {}

        # Compute derived chain attributes
        pimap = {}
        for pcid in pdb_chain_ids:
            entry, auth_id = pcid.split('_')
            pimap[pcid] = self.chain_properties(entry_chain_data[pcid], polymer_data[entry])

        session.logger.status("Done fetching PDB chain entity info for hits")
        return pimap

    def chain_properties(self, author_data, polymers):

        pr = {}
        try:
            entity_id = author_data["rcsb_polymer_entity_instance_container_identifiers"]["entity_id"]
        except KeyError:
            entity_data = {}
            entity_id = None
        else:
            entity_data = polymers[entity_id]
        try:
            chain_ids = entity_data["entity_poly"]["pdbx_strand_id"]
            num_chains = len(chain_ids.split(','))
        except KeyError:
            chain_ids = None
            num_chains = 0
        pr['chain_copies'] = num_chains
        try:
            num_residues = entity_data["entity_poly"]["rcsb_sample_sequence_length"]
        except KeyError:
            num_residues = 0
        pr['chain_residues'] = num_residues
        pr['npolymers'] = len(polymers)
        entity_ids = list(polymers.keys())
        entity_ids.sort(key= lambda eid, prime=entity_id: -1 if eid == prime else int(eid))
        cdesc = []
        if chain_ids:
            for eid in entity_ids:
                entity_info = polymers[eid]
                entity_chain_ids = entity_info["entity_poly"]["pdbx_strand_id"]
                cdesc.append("%s: %s" % (entity_chain_ids,
                    entity_info['rcsb_polymer_entity']['pdbx_description']))
        pr['chain_names'] = cdesc
        try:
            gene_srcs = set()
            for src_data in entity_data["entity_src_gen"]:
                gene_srcs.add(src_data["pdbx_gene_src_scientific_name"])
            src = ", ".join(gene_srcs)
        except KeyError:
            try:
                nat_srcs = set()
                for src_data in entity_data["entity_src_nat"]:
                    try:
                        nat_srcs.add(src_data["pdbx_organism_scientific"])
                    except KeyError:
                        nat_srcs.add(src_data["species"])
                src = ", ".join(nat_srcs)
            except KeyError:
                src = ""
        pr['chain_species'] = src
        try:
            accessions = entity_data["rcsb_polymer_entity_container_identifiers"]["reference_sequence_identifiers"]
            for accession_info in accessions:
                if accession_info["database_name"] == "UniProt":
                    accession_id = accession_info["database_accession"]
                    break
            else:
                accession_id = ""
        except KeyError:
            accession_id = ""
        pr['chain_sequence_id'] = accession_id
        try:
            weight = entity_data['rcsb_polymer_entity']['formula_weight']
        except KeyError:
            weight = 0.0
        pr['chain_weight'] = weight
        return pr

# -----------------------------------------------------------------------------
#
class PDB_Ligand_Info:
		
    columns = (
        ("Ligand formulas", 'ligandFormulas',
         {'anchor':"nw"}, 'PDB ligand chemical formulas.'),
        ("Ligand names", 'ligandNames',
         {'anchor':"nw"}, 'PDB ligand chemical names.'),
        ("Ligand smiles", 'ligandSmiles',
         {'anchor':"nw"}, 'PDB ligand smile strings.'),
        ("Ligand symbols", 'ligandSymbols',
         {'anchor':"nw"}, 'PDB ligand symbols.'),
        ("Ligand weights", 'ligandWeights',
         {'anchor':"nw"}, 'PDB ligand weights (Daltons).'),
        )

    def fetch_info(self, session, pdb_chain_ids):

        if len(pdb_chain_ids) == 0:
            return {}

        session.logger.status("Fetching PDB ligand info for hits")
        global _non_polymer_info
        by_name = {}
        by_entry = {}
        for i, code_chain in enumerate(pdb_chain_ids):
            session.logger.status("%d/%d" % (i, len(pdb_chain_ids)), secondary=True)
            entry = code_chain.split('_')[0]
            if entry in by_entry:
                continue
            by_entry[entry] = all_ligands = []
            for entity_id in _non_polymer_info[entry]:
                key = (entry, entity_id)
                entity_data = fetch_from_pdb(session, 'nonpolymer_entity/%s/%s', [key])[key]
                if not entity_data:
                    continue
                name = entity_data["pdbx_entity_nonpoly"]["comp_id"]
                all_ligands.append(name)
                if name in by_name:
                    continue
                comp_data = fetch_from_pdb(session, 'chemcomp/%s', [name])[name]
                by_name[name] = lig_info = {}
                try:
                    lig_info["Name"] = comp_data["chem_comp"]["name"]
                except KeyError:
                    lig_info["Name"] = None
                try:
                    lig_info["Formula"] = comp_data["chem_comp"]["formula"]
                except KeyError:
                    lig_info["Formula"] = None
                try:
                    lig_info["Weight"] = comp_data["chem_comp"]["formula_weight"]
                except KeyError:
                    lig_info["Weight"] = None
                try:
                    lig_info["Smile"] = comp_data["rcsb_chem_comp_descriptor"]["smiles"]
                except KeyError:
                    lig_info["Smile"] = None
        session.logger.status("", secondary=True)

        entry_ligands = {}
        for entry, names in by_entry.items():
            entry_ligands[entry] = lig_dict = {}
            lig_dict['ligandSymbols'] = names
            comp_dicts = [by_name[n] for n in names]
            for key in ["Name", "Formula", "Weight", "Smile"]:
                lig_dict["ligand"+key+"s"] = [cd[key] for cd in comp_dicts]
        session.logger.status("Done fetching PDB ligand info for hits")
        return dict([(pcid, entry_ligands[pcid.split('_')[0]]) for pcid in pdb_chain_ids])


def fetch_from_pdb(session, query_template, query_args):

    info = {}
    for i, arg in enumerate(query_args):
        if len(query_args) > 1:
            session.logger.status("%d/%d" % (i, len(query_args)), secondary=True)
        url = 'https://data.rcsb.org/rest/v1/core/%s' % (query_template % arg)
        from urllib.request import urlopen
        from urllib.error import URLError, HTTPError
        import sys
        try:
            f = urlopen(url)
            data = f.read()
            f.close()
        except (URLError, HTTPError) as e:
            if session:
                session.logger.warning('Fetching BLAST PDB info using URL:\n'
                                       '%s\nFailed %s\n' % (url, str(e)))
            info[arg] = {}
            continue
        info[arg] = eval(data)
    if len(query_args) > 1:
        session.logger.status("", secondary=True)
    return info

# -----------------------------------------------------------------------------
#
entry_info = PDB_Entry_Info()
chain_info = PDB_Chain_Info()
ligand_info = PDB_Ligand_Info()

if __name__ == "__main__":
    chain_ids = ["1GCN_A", "3FX2_A"]
    from pprint import pprint
    print("Entry info:")
    pprint(entry_info.fetch_info(None, chain_ids))
    print("Chain info:")
    pprint(chain_info.fetch_info(None, chain_ids))
    print("Ligand info:")
    pprint(ligand_info.fetch_info(None, chain_ids))
