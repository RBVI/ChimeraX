# ---------------------------------------------------------------------------------------
# Fetch fasta file from uniprot, e.g. http://www.uniprot.org/uniprot/P12345.fasta
#
def fetch_uniprot(id, session, ignore_cache = False):
    '''
    Fetch protein sequences from UniProt in fasta format.
    '''
    site = 'www.uniprot.org'
    url_pattern = 'uniprot/%s.fasta'
    idu = id.upper()
    url = "http://%s/%s" % (site, url_pattern % (idu,))
    save_dir = 'UniProt'
    save_name = '%s.fasta' % (idu,)
    min_file_size = None
    
    from .fetch import fetch_file
    try:
        path, headers = fetch_file(url, id, session, min_file_size,
                                   save_dir, save_name, ignore_cache = ignore_cache)
    except IOError as e:
      raise

    return path
