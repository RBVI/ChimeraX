# -----------------------------------------------------------------------------
# Fetch crystallographic density maps from the Upsalla Electron Density Server.
#
# ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-1535/map/emd_1535.map.gz
# ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-1535/header/emd-1535.xml
#

# -----------------------------------------------------------------------------
#
def fetch_emdb_map(id, open_fit_pdbs = False, ignore_cache=False):

  map_path = emdb_map_path(id, ignore_cache)

  # Display map.
  from os.path import basename
  map_name = basename(map_path)
  from ..ui.gui import show_status, show_info
  show_status('Opening map %s...' % map_name)
  from ..map import open_volume_file
  models = open_volume_file(map_path, 'ccp4', map_name, 'surface',
                            open_models = False)
  for m in models:
    m.data.database_fetch = (id, 'emdb')

  if open_fit_pdbs:
    # Find fit pdb ids.
    show_status('EMDB %s: looking for fits PDBs' % id)
    pdb_ids = fit_pdb_ids_from_web_service(id)
    msg = ('EMDB %s has %d fit PDB models: %s\n'
           % (id, len(pdb_ids), ','.join(pdb_ids)))
    show_status(msg)
    show_info(msg)
    if pdb_ids:
      mlist = []
      from .fetch_pdb import fetch_pdb
      for pdb_id in pdb_ids:
        show_status('Opening %s' % pdb_id)
        m = fetch_pdb(pdb_id, ignore_cache=ignore_cache)
        mlist.extend(m)
      models.extend(mlist)

  return models

# -----------------------------------------------------------------------------
#
def emdb_map_path(id, ignore_cache = False):

  site = 'ftp.wwpdb.org'
  url_pattern = 'ftp://%s/pub/emdb/structures/EMD-%s/map/%s'
  xml_url_pattern = 'ftp://%s/pub/emdb/structures/EMD-%s/header/%s'

  from ..ui.gui import show_status
  show_status('Fetching %s from %s...' % (id,site))

  # Fetch map.
  map_name = 'emd_%s.map' % id
  map_gz_name = map_name + '.gz'
  map_url = url_pattern % (site, id, map_gz_name)
  name = 'EMDB %s' % id
  minimum_map_size = 8192       # bytes
  from .fetch import fetch_file
  try:
    map_path, headers = fetch_file(map_url, name, minimum_map_size,
                 'EMDB', map_name, uncompress = 'always', ignore_cache=ignore_cache)
  except IOError as e:
    if 'Failed to change directory' in str(e):
      raise IOError('EMDB ID %s does not exist or map has not been released.' % id)
    else:
      raise

  return map_path
  
# -----------------------------------------------------------------------------
#
def fit_pdb_ids_from_xml(xml_file):

  # ---------------------------------------------------------------------------
  # Handler for use with Simple API for XML (SAX2).
  #
  from xml.sax import ContentHandler
  class EMDB_SAX_Handler(ContentHandler):

    def __init__(self):
      self.pdbEntryId = False
      self.ids = []

    def startElement(self, name, attrs):
      if name == 'pdbEntryId':
        self.pdbEntryId = True

    def characters(self, s):
      if self.pdbEntryId:
        self.ids.append(s)

    def endElement(self, name):
      if name == 'pdbEntryId':
        self.pdbEntryId = False

    def pdb_ids(self):
      return (' '.join(self.ids)).split()

  from xml.sax import make_parser
  xml_parser = make_parser()

  from xml.sax.handler import feature_namespaces
  xml_parser.setFeature(feature_namespaces, 0)

  h = EMDB_SAX_Handler()
  xml_parser.setContentHandler(h)
  xml_parser.parse(xml_file)

  return h.pdb_ids()

# -----------------------------------------------------------------------------
#
def fit_pdb_ids_from_web_service(id):

  from WebServices.emdb_client import EMDB_WS
  ws = EMDB_WS()
  import socket
  try:
    results = ws.findFittedPDBidsByAccessionCode(id)
  except socket.gaierror:
    from chimera import replyobj
    replyobj.error('Could not connect to EMDB web service\nto determine fit PDB entries.')
    pdb_ids = []
  else:
    pdb_ids = [t['fittedPDBid'] for t in ws.rowValues(results)
               if t['fittedPDBid']]
  return pdb_ids

# -----------------------------------------------------------------------------
# Register to fetch EMDB maps with open command.
#
def register_emdb_fetch():

  from .fetch import register_fetch_database as reg
  reg('EMDB', fetch_emdb_map, '1535', 'www.ebi.ac.uk/pdbe/emdb',
      'http://www.ebi.ac.uk/msd-srv/emsearch/atlas/%s_summary.html')
