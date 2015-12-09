# -----------------------------------------------------------------------------
#
def fetch_eds_map(session, id, type = '2fofc', ignore_cache=False):
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
  log = session.logger
  log.status('Fetching %s from web site %s...' % (id,site))

  if type == 'fofc':
    map_name = id + '_diff.omap'
  elif type == '2fofc':
    map_name = id + '.omap'
  map_url = url_pattern % (site, id[1:3], id, map_name)

  from ..fetch import fetch_file
  filename = fetch_file(session, map_url, 'map %s' % id, map_name, 'EDS')

  from .. import io
  models, status = io.open_data(session, filename, format = 'dsn6', name = id)
  return models, status

# -----------------------------------------------------------------------------
# Register to fetch EMDB maps with open command.
#
def register_eds_fetch(session):
    from .. import fetch
    fetch.register_fetch(session, 'eds', fetch_eds_map, 'dsn6',
                         prefixes = ['eds'])
#    reg('EDS', fetch_eds_map, '1A0M', 'eds.bmc.uu.se/eds',
#        'http://eds.bmc.uu.se/cgi-bin/eds/uusfs?pdbCode=%s', session)
