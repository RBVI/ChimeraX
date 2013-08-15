def fetch_pdb(id, ignore_cache=False):

    site = 'www.rcsb.org'
    url_pattern = 'pdb/files/%s.pdb'
    idu = id.upper()
    url = "http://%s/%s" % (site, url_pattern % idu)
    save_dir = 'PDB'
    save_name = '%s.pdb' % idu
    min_file_size = 80*20
    
    from .fetch import fetch_file
    try:
        path, headers = fetch_file(url, id, min_file_size,
                                   save_dir, save_name,
                                   ignore_cache=ignore_cache)
    except IOError as e:
      raise

    from .pdb import open_pdb_file
    m = open_pdb_file(path)
    return [m]

# -----------------------------------------------------------------------------
# Register to fetch PDB maps with open command.
#
def register_pdb_fetch():

  from .fetch import register_fetch_database as reg
  reg('PDB', fetch_pdb, '1a0m', 'www.rcsb.org/pdb/home',
      'http://www.rcsb.org/pdb/explore/explore.do?structureId=%s')
