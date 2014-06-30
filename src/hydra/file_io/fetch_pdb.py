def fetch_pdb(id, session, ignore_cache = False, file_format = 'pdb'):
    '''
    Fetch molecular models from the Protein Data Bank, www.rcsb.org
    '''

    site = 'www.rcsb.org'
    url_pattern = 'pdb/files/%s.%s.gz'
    idu = id.upper()
    file_suffix = {'pdb':'pdb', 'mmcif':'cif'}[file_format]
    url = "http://%s/%s" % (site, url_pattern % (idu, file_suffix))
    save_dir = 'PDB'
    save_name = '%s.%s' % (idu, file_suffix)
    min_file_size = 80*20
    
    from .fetch import fetch_file
    try:
        path, headers = fetch_file(url, id, session, min_file_size,
                                   save_dir, save_name, uncompress = 'always',
                                   ignore_cache = ignore_cache)
    except IOError as e:
      raise

    if file_format == 'pdb':
        from .pdb import open_pdb_file
        mols = open_pdb_file(path, session)
    elif file_format == 'mmcif':
        from .mmcif import open_mmcif_file
        mols = open_mmcif_file(path, session)
    mols[0].database_fetch = (id, file_format)
    return mols

# -----------------------------------------------------------------------------
#
def fetch_mmcif(id, session, ignore_cache = False):
    '''
    Fetch molecular models in mmCIF format from the Protein Data Bank, www.rcsb.org
    '''
    return fetch_pdb(id, session, ignore_cache, file_format = 'mmcif')

# -----------------------------------------------------------------------------
# Register to fetch PDB maps with open command.
#
def register_pdb_fetch(session):

  from .fetch import register_fetch_database as reg
  reg('PDB', fetch_pdb, '1a0m', 'www.rcsb.org/pdb/home',
      'http://www.rcsb.org/pdb/explore/explore.do?structureId=%s', session)
  reg('PDBmmCIF', fetch_mmcif, '1a0m', 'www.rcsb.org/pdb/home',
      'http://www.rcsb.org/pdb/explore/explore.do?structureId=%s', session)
