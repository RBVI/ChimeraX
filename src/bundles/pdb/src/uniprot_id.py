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

def uniprot_ids(structure):
    '''
    Extract uniprot database identifiers for each chain from PDB meta data.
    '''
    if not 'DBREF' in structure.metadata:
        return []

    useqs = []
    dbrefs = structure.metadata['DBREF']
    for dbref in dbrefs:
        try:
            if dbref[26:32].strip() != 'UNP':
                continue
            chain_id = dbref[12]
            uniprot_id = dbref[33:41].strip()
            uniprot_name = dbref[42:54].strip()
            chain_seq_start = int(dbref[14:18].strip())
            chain_seq_end = int(dbref[20:24].strip())
            db_seq_start = int(dbref[55:60].strip())
            db_seq_end = int(dbref[62:67].strip())
            useq = UniprotSequence(chain_id, uniprot_id, uniprot_name,
                                   (db_seq_start, db_seq_end),
                                   (chain_seq_start, chain_seq_end))
            useqs.append(useq)
        except Exception:
            pass

    return useqs

class UniprotSequence:
    def __init__(self, chain_id, uniprot_id, uniprot_name,
                 database_sequence_range, chain_sequence_range):
        self.chain_id = chain_id
        self.uniprot_id = uniprot_id        
        self.uniprot_name = uniprot_name
        self.database_sequence_range = database_sequence_range
        self.chain_sequence_range = chain_sequence_range
