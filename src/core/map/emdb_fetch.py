# vim: set expandtab ts=4 sw=4:
# -----------------------------------------------------------------------------
# Fetch density maps from the Electron Microscopy Data Bank
#
#       ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-5582/map/emd_5582.map.gz
#       ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-5680/map/emd_5680.map.gz
#
def fetch_emdb(session, emdb_id):
    from ..errors import UserError
    if len(emdb_id) != 4:
        raise UserError("EMDB identifiers are 4 characters long")

    import socket
    hname = socket.gethostname()
    url_pattern = ('ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-%s/map/%s.gz'
                   if hname.endswith('.edu') or hname.endswith('.gov') else
                   'ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/map/%s.gz')
    map_name = 'emd_%s.map' % emdb_id
    map_url = url_pattern % (emdb_id, map_name)

    from ..fetch import fetch_file
    filename = fetch_file(session, map_url, 'map %s' % emdb_id, map_name, 'EMDB',
                          uncompress = True)

    return filename, emdb_id

# -----------------------------------------------------------------------------
#
def register_emdb_fetch():
    # TODO: The io module doesn't support the concept database name, instead requiring a format name.
    from .. import io
    from .volume import open_map
    io.register_format('emdb', io.VOLUME, [".map"], ["emdb"], open_func = open_map)
    io.register_fetch('emdb', fetch_emdb)
