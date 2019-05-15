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

        if len(pdb_chain_ids) == 0:
            return {}

        pdb_ids = [id.split('_')[0].lower() for id in pdb_chain_ids]

        # Fetch PDB entry info
        xml = fetch_from_pdb(session, 'describePDB', set(pdb_ids))
        if xml is None:
            # TODO: Warn that fetch failed.
            return {}
        pis = [self.pdb_info(e) for e in xml.getElementsByTagName('PDB')]
        da = dict((pi.structureId.lower(), self.entry_properties(pi))
                  for pi in pis)
        pimap = dict((pcid, da.get(pdb_id.lower(),{}))
                     for pdb_id, pcid in zip(pdb_ids, pdb_chain_ids))
        return pimap

    def pdb_info(self, xml_element):

        pe = PDB_Entry()
        attrs = xml_element.attributes
        for i in range(attrs.length):
            a = attrs.item(i)
            setattr(pe, a.name, a.value)
        strings_to_numeric_values(pe)
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

        self.integer_fields = ('nr_entities', 'nr_residues', 'nr_atoms',
                               'pubmedId')
        self.float_fields = ('resolution',)

# -----------------------------------------------------------------------------
#
def strings_to_numeric_values(o):

    for fname, fcast in (('integer', int), ('float', float)):
        for aname in getattr(o, fname+'_fields', ()):
            try:
                setattr(o, aname, fcast(getattr(o, aname)))
            except:
                pass

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

        pdb_ids = [id.split('_')[0].lower() for id in pdb_chain_ids]

        # Fetch chain info.
        xml = fetch_from_pdb(session, 'describeMol', set(pdb_ids))
        if xml is None:
            # TODO: Warn that fetch failed.
            return {}
        pos = [self.polymers_info(e)
               for e in xml.getElementsByTagName('structureId')]

        # Compute derived chain attributes
        pomap = dict((pdb_id.lower(), polys) for pdb_id, polys in pos)
        pimap = {}
        for pcid in pdb_chain_ids:
            pdb_id, cid = pcid.split('_')
            polys = pomap.get(pdb_id.lower(), [])
            pimap[pcid] = self.chain_properties(polys, cid)

        return pimap

    def polymers_info(self, e):

        pdb_id = e.getAttribute('id')
        polymers = [self.polymer_info(p)
                    for p in e.getElementsByTagName('polymer')]
        return (pdb_id, polymers)

    def polymer_info(self, e):

        p = Polymer()
        for a in ('entityNr', 'length', 'type', 'weight'):
            if e.hasAttribute(a):
                setattr(p, a, e.getAttribute(a))
        p.chainIds = [c.getAttribute('id')
                      for c in e.getElementsByTagName('chain')
                      if c.hasAttribute('id')]
        p.macroMoleculeName = tag_attribute(e, 'macroMolecule', 'name')
        mm = e.getElementsByTagName('macroMolecule')
        if mm.length >= 1:
            p.macroMoleculeId = tag_attribute(mm.item(0), 'accession', 'id')
        p.polymerDescription = tag_attribute(e, 'polymerDescription',
                                                'description')
        p.fragment = tag_attribute(e, 'fragment', 'desc')
        p.details = tag_attribute(e, 'details', 'desc')
        p.taxonomyName = tag_attribute(e, 'Taxonomy', 'name')
        p.taxonomyId = tag_attribute(e, 'Taxonomy', 'id')
        strings_to_numeric_values(p)
        return p

    def chain_properties(self, polymers, chain_id):

        pr = {}
        cp = self.chain_polymer(polymers, chain_id)
        pr['chain_copies'] = len(cp.chainIds) if cp else 0
        pr['chain_residues'] = cp.length if cp else 0
        pr['npolymers'] = len(polymers)
        polys = list(polymers)
        if cp:
            polys.remove(cp)
            polys.insert(0,cp)
        cdesc = ['%s: %s %s' % (''.join(p.chainIds), p.polymerDescription, p.fragment) for p in polys]
        pr['chain_names'] = cdesc
        pr['chain_species'] = cp.taxonomyName if cp else ''
        pr['chain_sequence_id'] = cp.macroMoleculeId if cp and cp.macroMoleculeId else ''
        pr['chain_weight'] = cp.weight if cp else ''
        return pr

    def chain_polymer(self, polymers, chain_id):

        for p in polymers:
            if chain_id in p.chainIds:
                return p
        return None

# -----------------------------------------------------------------------------
#
class Polymer:

    def __init__(self):

        self.entityNr = None            # "1"
        self.length = None              # "141"
        self.type = None                # "protein"
        self.weight = None              # "15150.5"
        self.chainIds = []              # ["A", "C"]
        self.macroMoleculeName = ''     # "Hemoglobin subunit alpha"
        self.macroMoleculeId = None     # "P69905"
        self.polymerDescription = ''    # "HEMOGLOBIN (DEOXY) (ALPHA CHAIN)"
        self.fragment = ''              # "HEAVY CHAIN 1-219"
        self.details = ''               # "OBTAINED BY PAPAIN CLEAVAGE (FAB)"
        self.taxonomyName = ''          # "Homo sapiens"
        self.taxonomyId = None          # "9606"

        self.integer_fields = ('entityNr', 'length')
        self.float_fields = ('weight',)

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

        pdb_ids = [id.split('_')[0].lower() for id in pdb_chain_ids]

        xml = fetch_from_pdb(session, 'ligandInfo', set(pdb_ids))
        if xml is None:
            # TODO: Warn that fetch failed.
            return {}
        los = [self.ligands_info(e)
               for e in xml.getElementsByTagName('structureId')]

        lmap = dict((pdb_id.lower(), ligands) for pdb_id, ligands in los)
        pmap = dict((pcid, self.ligand_properties(lmap.get(pdb_id, {})))
                    for pcid, pdb_id in zip(pdb_chain_ids, pdb_ids))
        return pmap

    def ligands_info(self, e):

        pdb_id = e.getAttribute('id')
        ligands = [self.ligand_info(p)
                   for p in e.getElementsByTagName('ligand')]
        return (pdb_id, ligands)

    def ligand_info(self, e):

        lig = Ligand()
        for a in ('chemicalID', 'type', 'molecularWeight'):
            if e.hasAttribute(a):
                setattr(lig, a, e.getAttribute(a))
        for t in ('chemicalName', 'formula', 'smiles', 'InChi', 'InChiKey'):
            te = e.getElementsByTagName(t)
            if te.length >= 1:
                setattr(lig, t, tag_text(te.item(0)))
        strings_to_numeric_values(lig)
        return lig

    def ligand_properties(self, ligands):

        d = {
            'ligandNames': [l.chemicalName for l in ligands],
            'ligandSmiles': [l.smiles for l in ligands],
            'ligandSymbols': [l.chemicalID for l in ligands],
            'ligandWeights': [l.molecularWeight for l in ligands],
            'ligandFormulas': [l.formula for l in ligands],
            }
        return d

# -----------------------------------------------------------------------------
#
class Ligand:

    def __init__(self):

        self.chemicalID = None          # "1B0"
        self.type = None                # "non-polymer"
        self.molecularWeight = None     # "425.522"
        self.chemicalName = ''          # "PHENYLALANINAMIDE"
        self.formula = ''               # "C27 H27 N3 O2"
        self.smiles = ''                # "Cc1c(c2ccccc2[nH]1)CC(=O)N[C@@H](Cc3ccccc3)C(=O)N(C)c4ccccc4</smiles>"
        self.InChi = None
        self.InChiKey = None

        self.float_fields = ('molecularWeight',)

# -----------------------------------------------------------------------------
#
def fetch_from_pdb(session, query, pdb_ids):

    ids = ','.join([pdb_id.lower() for pdb_id in pdb_ids])
    url = 'https://www.rcsb.org/pdb/rest/%s?structureId=%s' % (query, ids)
    from urllib.request import urlopen
    from urllib.error import URLError, HTTPError
    try:
        f = urlopen(url)
        xml_string = f.read()
        f.close()
    except (URLError, HTTPError) as e:
        if session:
            session.logger.warning('Fetching BLAST PDB info using URL:\n'
                                   '%s\nFailed %s\n' % (url, str(e)))
        return None
    from xml.dom.minidom import parseString
    xml = parseString(xml_string)
    return xml

# -----------------------------------------------------------------------------
#
def tag_attribute(element, tag_name, attr_name, default = ''):

    for e in element.getElementsByTagName(tag_name):
        if e.hasAttribute(attr_name):
            return e.getAttribute(attr_name)
    return default

# -----------------------------------------------------------------------------
#
def tag_text(element, default = ''):

    t = ''.join(n.data for n in element.childNodes if n.nodeType == n.TEXT_NODE)
    return t

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
