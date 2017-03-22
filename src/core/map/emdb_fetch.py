# vim: set expandtab ts=4 sw=4:

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
# Fetch density maps from the Electron Microscopy Data Bank
#
#       ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-5582/map/emd_5582.map.gz
#	https://files.rcsb.org/pub/emdb/structures/EMD-1013/map/emd_1013.map.gz
#       ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-5680/map/emd_5680.map.gz
#
def fetch_emdb(session, emdb_id, ignore_cache=False, **kw):
    from ..errors import UserError
    if len(emdb_id) != 4:
        raise UserError("EMDB identifiers are 4 characters long")

    import socket
    hname = socket.gethostname()
    if hname.endswith('.edu') or hname.endswith('.gov'):
        # TODO: RCSB https is 20x slower than ftp. Cole Christie looking into it.
        #    url_pattern = ('https://files.rcsb.org/pub/emdb/structures/EMD-%s/map/%s.gz'
        # The RCSB ftp does not report file size so progress messages don't indicate how long it will take.
        url_pattern = 'ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-%s/map/%s.gz'
    elif hname.endswith('.cn'):
        url_pattern = 'ftp://ftp.emdb-china.org/structures/EMD-%s/map/%s.gz'
    else:
        url_pattern = 'ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/map/%s.gz'
        
    map_name = 'emd_%s.map' % emdb_id
    map_url = url_pattern % (emdb_id, map_name)

    from ..fetch import fetch_file
    filename = fetch_file(session, map_url, 'map %s' % emdb_id, map_name, 'EMDB',
                          uncompress = True, ignore_cache=ignore_cache)

    from .. import io
    models, status = io.open_data(session, filename, format = 'ccp4', name = emdb_id, **kw)
    return models, status

# -----------------------------------------------------------------------------
#
def register_emdb_fetch():
    from .. import fetch
    fetch.register_fetch('emdb', fetch_emdb, 'ccp4', prefixes = ['emdb'])
