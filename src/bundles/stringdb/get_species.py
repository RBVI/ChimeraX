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


def get_species(m):
    """Return list of species names for given model."""
    # Extract the _entity_src_gen.pdbx_gene_src_scientific_name
    # field from an AtomicStructure instance created from an
    # mmCIF file.  No error checking is done, so caller needs
    # to catch any exceptions raised.  (No, I don't have a
    # comprehensive list of exceptions that may be raised.)
    # The _entity_src_gen table may have a single or multiple
    # rows, and the _pdbx_gene_src_scientific_name field may
    # list a single or multiple (comma-separated) organisms.
    # The code tries to normalize all that into a single set
    # of scientific names of organisms as they appear in the
    # mmCIF file.  Note that while the vocabulary for organism
    # names is constrained, the names appear to be case-independent.
    # (I've seen "Homo sapiens" and "HOMO SAPIENS" in different
    # entries.)  (No, I do not have a comprehensive list of
    # species names that appear in the PDB.)  The names are
    # presented "as-is" in case the caller cares, but I expect
    # most will just down/upcase the entire set.
    from chimerax.core.atomic.mmcif import get_mmcif_tables
    ep, esg, esn = get_mmcif_tables(m.filename, ["entity_poly", "entity_src_gen", "entity_src_nat"])
    # First, get our entity_id chain association
    rows = ep.fields(["entity_id", "pdbx_strand_id"])
    chain_ids = {}
    for row in rows:
        chain_ids[row[0]] = [chain.strip() for chain in row[1].split(',')]
    if esg is None:
        esg_rows = []
    else:
        esg_rows = esg.fields(["entity_id", "pdbx_gene_src_scientific_name"])
    if esn is None:
        esn_rows = []
    else:
        esn_rows = esn.fields(["entity_id", "pdbx_organism_scientific"])
    import itertools
    species = {}
    for row in itertools.chain(esg_rows, esn_rows):
        entity_id = row[0]
        # names = [name.strip() for name in row[1].split(',')]
        if entity_id in chain_ids:
            for chain in chain_ids[entity_id]:
                species[chain] = row[1]
        # for name in names:
        #     if name in species:
        #         species[name] += 1
        #     else:
        #         species[name] = 1
    return species


# Test with "ChimeraX.exe 3fx2 get_species.py" to see species
# list on standard output.
if __name__.startswith("ChimeraX_sandbox"):
    session = session # noqa
    from chimerax.core.atomic import AtomicStructure
    for m in session.models.list(type=AtomicStructure):
        session.logger.info("species %s %s" % (m, get_species(m)))
