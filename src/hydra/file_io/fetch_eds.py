# -----------------------------------------------------------------------------
#
def fetch_eds_map(id, session, type = '2fofc', ignore_cache=False):
  '''
  Fetch crystallographic density maps from the Upsalla Electron Density Server.

   2fofc:    http://eds.bmc.uu.se/eds/sfd/1cbs/1cbs.omap
    fofc:    http://eds.bmc.uu.se/eds/sfd/1cbs/1cbs_diff.omap
    Info:    http://eds.bmc.uu.se/cgi-bin/eds/uusfs?pdbCode=1cbs
  Holdings:  http://eds.bmc.uu.se/eds/eds_holdings.txt
  '''

  site = 'eds.bmc.uu.se'
  url_pattern = 'http://%s/eds/dfs/%s/%s/%s'

  # Fetch map.
  s = session
  s.show_status('Fetching %s from web site %s...' % (id,site))
  if type == 'fofc':
    map_name = id + '_diff.omap'
  elif type == '2fofc':
    map_name = id + '.omap'
  map_url = url_pattern % (site, id[1:3], id, map_name)
  name = 'map %s' % id
  minimum_map_size = 8192       # bytes
  from .fetch import fetch_file
  map_path, headers = fetch_file(map_url, name, session, minimum_map_size,
                                 'EDS', map_name, ignore_cache=ignore_cache)
    
  # Display map.
  s.show_status('Opening map %s...' % map_name)
  from ..map import open_volume_file
  models = open_volume_file(map_path, session, 'dsn6', map_name, 'mesh',
                            open_models = False)
  for m in models:
    m.data.database_fetch = (id, 'eds')

  return models

# -----------------------------------------------------------------------------
# Register to fetch EMDB maps with open command.
#
def register_eds_fetch(session):

  from .fetch import register_fetch_database as reg
  reg('EDS', fetch_eds_map, '1A0M', 'eds.bmc.uu.se/eds',
      'http://eds.bmc.uu.se/cgi-bin/eds/uusfs?pdbCode=%s', session)
