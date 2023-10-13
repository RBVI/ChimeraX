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
    Extract uniprot database identifiers for each chain from mmCIF meta data.
    '''
    table_names = ('struct_ref',
                   'struct_ref_seq')
    from chimerax import mmcif
    database_ref, database_ref_seq = mmcif.get_mmcif_tables_from_metadata(structure, table_names)
    if not database_ref or not database_ref_seq:
        return []

    db_refs = database_ref.mapping('id', ['db_name', 'db_code', 'pdbx_db_accession'])
    db_codes = {id: (db_code, db_id) for id, (db_name, db_code, db_id) in db_refs.items()
                if db_name == 'UNP'}

    chains = database_ref_seq.fields(('ref_id', 'pdbx_strand_id',
                                      'db_align_beg', 'db_align_end',
                                      'pdbx_auth_seq_align_beg', 'pdbx_auth_seq_align_end'),
                                     allow_missing_fields=True)
    useqs = []
    for ref_id, chain_id, db_seq_start, db_seq_end, chain_seq_start, chain_seq_end in chains:
        if ref_id not in db_codes:
            continue
        uniprot_name, uniprot_id = db_codes.get(ref_id)
        if None in (uniprot_name, uniprot_id, db_seq_start, db_seq_end, chain_seq_start, chain_seq_end):
            continue
        try:
            db_seq_range = (int(db_seq_start), int(db_seq_end))
        except Exception:
            db_seq_range = None
        try:
            chain_seq_range = (int(chain_seq_start), int(chain_seq_end))
        except Exception:
            chain_seq_range = None

        useq = UniprotSequence(chain_id, uniprot_id, uniprot_name,
                               db_seq_range, chain_seq_range)
        useqs.append(useq)

    return useqs


class UniprotSequence:
    def __init__(self, chain_id, uniprot_id, uniprot_name,
                 database_sequence_range, chain_sequence_range):
        self.chain_id = chain_id
        self.uniprot_id = uniprot_id
        self.uniprot_name = uniprot_name
        self.database_sequence_range = database_sequence_range
        self.chain_sequence_range = chain_sequence_range
