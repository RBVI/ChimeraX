# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
def fetch_ihm(session, id, ignore_cache=False, **kw):
  '''
  Fetch IHM models from PDB-Dev.

  https://pdb-dev.wwpdb.org/cif/PDBDEV_00000012.cif
  '''

  url_pattern = 'https://pdb-dev.wwpdb.org/cif/%s'
  
  if len(id) < 8:
      zero_pad = '0'*(8-len(id))
      full_id = zero_pad + id
  else:
      full_id = id
      
  log = session.logger
  log.status('Fetching %s from PDB-Dev...' % (full_id,))

  name = 'PDBDEV_%s.cif' % full_id
  url = url_pattern % name

  from chimerax.core.fetch import fetch_file
  filename = fetch_file(session, url, 'IHM %s' % full_id, name, 'PDBDev',
                        ignore_cache=ignore_cache)

  log.status('Opening %s' % name)
  models, status = session.open_command.open_data(filename, format = 'ihm',
  	name = name, **kw)
    
  return models, status
