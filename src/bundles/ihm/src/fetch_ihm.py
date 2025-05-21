# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
def fetch_ihm(session, id, ignore_cache=False, **kw):
  '''
  Fetch IHM models from PDB-IHM.

  https://pdb-ihm.org/cif/8zzi.cif
  https://pdb-ihm.org/cif/PDBDEV_00000012.cif
  '''

  url_pattern = 'https://pdb-ihm.org/cif/%s'

  if len(id) == 4 and [c for c in id if c.isalpha()]:
    full_id = id
    prefix = ''
  elif len(id) < 8:
    zero_pad = '0'*(8-len(id))
    full_id = zero_pad + id
    prefix = 'PDBDEV_'
  else:
    full_id = id
    prefix = 'PDBDEV_'
      
  log = session.logger
  log.status('Fetching %s from PDB-IHM...' % (full_id,))

  name = f'{prefix}{full_id}.cif'
  url = url_pattern % name

  from chimerax.core.fetch import fetch_file
  filename = fetch_file(session, url, 'IHM %s' % full_id, name, 'PDB-IHM',
                        ignore_cache=ignore_cache)

  log.status('Opening %s' % name)
  models, status = session.open_command.open_data(filename, format = 'ihm',
  	                                          name = name, **kw)
    
  return models, status
