def fetch_pdb(id, session, ignore_cache=False):
    '''
    Fetch molecular models from the Protein Data Bank, www.rcsb.org
    '''

    site = 'www.rcsb.org'
    url_pattern = 'pdb/files/%s.pdb.gz'
    idu = id.upper()
    url = "http://%s/%s" % (site, url_pattern % idu)
    save_dir = 'PDB'
    save_name = '%s.pdb' % idu
    min_file_size = 80*20
    
    from .fetch import fetch_file
    try:
        path, headers = fetch_file(url, id, session, min_file_size,
                                   save_dir, save_name, uncompress = 'always',
                                   ignore_cache = ignore_cache)
    except IOError as e:
      raise

    from .pdb import open_pdb_file
    m = open_pdb_file(path, session)
    m.database_fetch = (id, 'pdb')
    from ..molecule import Molecule
    return [m] if isinstance(m, Molecule) else m

# -----------------------------------------------------------------------------
# Register to fetch PDB maps with open command.
#
def register_pdb_fetch(session):

  from .fetch import register_fetch_database as reg
  reg('PDB', fetch_pdb, '1a0m', 'www.rcsb.org/pdb/home',
      'http://www.rcsb.org/pdb/explore/explore.do?structureId=%s', session)
