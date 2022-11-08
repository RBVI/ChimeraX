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
def fetch_emdb(session, emdb_id, mirror = None, transfer_method = None, ignore_cache=False, **kw):
    from chimerax.core.errors import UserError
    if len(emdb_id) < 4:
        raise UserError("EMDB identifiers are at least 4 characters long")

    if mirror is None:
        import socket
        hname = socket.gethostname()
        if hname.endswith('.edu') or hname.endswith('.gov'):
            mirror = 'united states'
        elif hname.endswith('.cn'):
            mirror = 'china'
        elif hname.endswith('.jp'):
            mirror = 'japan'
        else:
            mirror = 'europe'

    # Choice of ftp vs https based on speed tests.  Ticket #5448
    if mirror == 'united states':
        # The RCSB ftp does not report file size so progress messages don't indicate how long it will take.
        if transfer_method == 'ftp':
            url_pattern = 'ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-%s/map/%s.gz'
        else:
            url_pattern = 'https://files.wwpdb.org/pub/emdb/structures/EMD-%s/map/%s.gz'
    elif mirror == 'china':
        if transfer_method == 'https':
            url_pattern = 'https://ftp.emdb-china.org/structures/EMD-%s/map/%s.gz'
        else:
            url_pattern = 'ftp://ftp.emdb-china.org/structures/EMD-%s/map/%s.gz'
    elif mirror == 'japan':
        if transfer_method == 'ftp':
            url_pattern = 'ftp://ftp.pdbj.org/pub/emdb/structures/EMD-%s/map/%s.gz'
        else:
            url_pattern = 'https://ftp.pdbj.org/pub/emdb/structures/EMD-%s/map/%s.gz'
    else:
        if transfer_method == 'https':
            url_pattern = 'https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/map/%s.gz'
        else:
            url_pattern = 'ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/map/%s.gz'
        
    map_name = 'emd_%s.map' % emdb_id
    map_url = url_pattern % (emdb_id, map_name)

    from chimerax.core.fetch import fetch_file
    filename = fetch_file(session, map_url, 'map %s' % emdb_id, map_name, 'EMDB',
                          uncompress = True, ignore_cache=ignore_cache)

    model_name = 'emdb %s' % emdb_id
    models, status = session.open_command.open_data(filename, format = 'ccp4',
        name = model_name, **kw)
    return models, status

def fetch_emdb_europe(session, emdb_id, transfer_method = None, ignore_cache=False, **kw):
    return fetch_emdb(session, emdb_id, mirror = 'europe', transfer_method = transfer_method,
                      ignore_cache = ignore_cache, **kw)

def fetch_emdb_japan(session, emdb_id, transfer_method = None, ignore_cache=False, **kw):
    return fetch_emdb(session, emdb_id, mirror = 'japan', transfer_method = transfer_method,
                      ignore_cache = ignore_cache, **kw)

def fetch_emdb_china(session, emdb_id, transfer_method = None, ignore_cache=False, **kw):
    return fetch_emdb(session, emdb_id, mirror = 'china', transfer_method = transfer_method,
                      ignore_cache = ignore_cache, **kw)

def fetch_emdb_us(session, emdb_id, transfer_method = None, ignore_cache=False, **kw):
    return fetch_emdb(session, emdb_id, mirror = 'united states', transfer_method = transfer_method,
                      ignore_cache = ignore_cache, **kw)
