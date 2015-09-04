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

  filename = "~/Downloads/Chimera/EDS/%s" % map_name
  import os
  filename = os.path.expanduser(filename)

  if os.path.exists(filename):
    return filename, id

  dirname = os.path.dirname(filename)
  os.makedirs(dirname, exist_ok=True)

  from urllib.request import URLError, Request
  from .. import utils
  request = Request(map_url, unverifiable=True, headers={
      "User-Agent": utils.html_user_agent(session.app_dirs),
  })
  try:
    utils.retrieve_cached_url(request, filename, log)
  except URLError as e:
    raise UserError(str(e))

  return filename, id

# -----------------------------------------------------------------------------
# Register to fetch EMDB maps with open command.
#
def register_eds_fetch():
    # TODO: The io module doesn't support the concept database name, instead requiring a format name.
    from .. import io
    from .volume import open_map
    io.register_format('eds', io.VOLUME, [".omap"], ["eds"], open_func = open_map)
    io.register_fetch('eds', fetch_eds_map)
#    reg('EDS', fetch_eds_map, '1A0M', 'eds.bmc.uu.se/eds',
#        'http://eds.bmc.uu.se/cgi-bin/eds/uusfs?pdbCode=%s', session)
