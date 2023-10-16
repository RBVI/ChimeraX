# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
# Sequence searches for the AlphaFold database.
#
def fetch_alphafold_sequence_databases(directory, log):
    metadata_url = 'http://ftp.ebi.ac.uk/pub/databases/alphafold/download_metadata.json'
    from os.path import join
    dbinfo = join(directory, 'databases.json')
    from chimerax.core.fetch import retrieve_url
    retrieve_url(metadata_url, dbinfo, logger=log)

#    _fetch_proteomes(directory, dbinfo, log)
    _fetch_reference_proteomes(directory, dbinfo, log)

# -----------------------------------------------------------------------------
# Appears AlphaFold DB does uses reference proteomes, one sequence per gene,
# judging by the number of structures listed in databases.json.
#
# https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/README
#
# https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000006548/UP000006548_3702.fasta.gz
#
def _fetch_reference_proteomes(directory, dbinfo, log):
    proteome_ids = _alphafold_uniprot_ids(dbinfo)
    kingdoms = ['Eukaryota', 'Bacteria', 'Archaea', 'Viruses']
    proteome_url = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/%s/%s/%s_%s.fasta.gz'
    from chimerax.core.fetch import retrieve_url
    from os.path import join
    for proteome_id, taxon_id, species in proteome_ids:
        filename = '%s_%s.fasta' % (species.replace(' ', '_'), proteome_id)
        for kingdom in kingdoms:
            url = proteome_url % (kingdom, proteome_id, proteome_id, taxon_id)
            path = join(directory, filename)
            try:
                retrieve_url(url, path, logger=log)
            except Exception:
                continue
            print (species, kingdom)
            break
        

# -----------------------------------------------------------------------------
# Appears AlphaFold DB does not use the proteomes, instead uses reference proteomes
# which maybe exclude isoforms.
#
def _fetch_proteomes(directory, dbinfo, log):
    proteome_url = 'https://www.uniprot.org/uniprot/?query=proteome:%s&format=fasta&compress=yes'
    proteome_ids = _uniprot_proteomes(dbinfo)
    from chimerax.core.fetch import retrieve_url
    for proteome_id, taxon_id, species in proteome_ids:
        url = proteome_url % proteome_id
        filename = '%s_%s.fasta' % (species.replace(' ', '_'), proteome_id)
        path = join(directory, filename)
        retrieve_url(url, path, logger=log)
        
# -----------------------------------------------------------------------------
#
def _alphafold_uniprot_ids(dbinfo):
    '''Returns list of (proteome_id, taxon_id, species_name).'''
    f = open(dbinfo, 'r')
    import json
    d = json.load(f)
    f.close()
    # Example
    #
    # {"archive_name": "UP000006548_3702_ARATH.tar",
    #  "species": "Arabidopsis thaliana",
    #  "common_name": "Arabidopsis",
    #  "latin_common_name": true,
    #  "reference_proteome": "UP000006548",
    #  "num_predicted_structures": 27434,
    #  "size_bytes": 3818536960}
    #
    ids = [(db['reference_proteome'], db['archive_name'].split('_')[1], db['species']) for db in d]
    return ids

if 'session' in globals():
    fetch_alphafold_sequence_databases(directory = 'sequences', log = session.logger)

