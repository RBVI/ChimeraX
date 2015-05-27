# -----------------------------------------------------------------------------
# Fetch density maps from the Electron Microscopy Data Bank
#
#       ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-5582/map/emd_5582.map.gz
#       ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-5680/map/emd_5680.map.gz
#
def fetch_emdb(session, emdb_id):
    if len(emdb_id) != 4:
        raise UserError("EMDB identifiers are 4 characters long")

    filename = "~/Downloads/Chimera/EMDB/emd_%s.map" % emdb_id
    import os
    filename = os.path.expanduser(filename)

    if os.path.exists(filename):
        return filename, emdb_id

    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)

    import socket
    hname = socket.gethostname()
    url_pattern = ('ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-%s/map/%s'
                   if hname.endswith('.edu') or hname.endswith('.gov') else
                   'ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/map/%s')
    map_name = 'emd_%s.map' % emdb_id
    map_gz_name = map_name + '.gz'
    map_url = url_pattern % (emdb_id, map_gz_name)
    filename_gz = filename + '.gz'

    from urllib.request import URLError, Request
    from .. import utils
    request = Request(map_url, unverifiable=True, headers={
        "User-Agent": utils.html_user_agent(session.app_dirs),
    })
    try:
        utils.retrieve_cached_url(request, filename_gz, session.logger)
    except URLError as e:
        raise UserError(str(e))

    from .. import io
    io.gunzip(filename_gz, filename)

    return filename, emdb_id

# -----------------------------------------------------------------------------
#
def register_emdb_fetch():
    # TODO: The io module doesn't support the concept database name, instead requiring a format name.
    from .. import io
    from .volume import open_map
    io.register_format('emdb', io.VOLUME, [".map"], ["emdb"], open_func = open_map)
    io.register_fetch('emdb', fetch_emdb)
