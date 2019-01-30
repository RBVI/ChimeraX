# vim: set expandtab shiftwidth=4 softtabstop=4:

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
def open_dicom(session, stream, name = None, format = 'dicom', **kw):
  if isinstance(stream, (str, list)):
    map_path = stream         # Batched paths
  else:
    map_path = stream.name
    stream.close()
  from os.path import basename
  name = basename(map_path if isinstance(map_path, str) else map_path[0])

  from chimerax.map import data
  grids = data.open_file(map_path, file_type = format, log = session.logger,
                         verbose = kw.get('verbose'))

  models = []
  msg_lines = []
  sgrids = []
  from chimerax.map.volume import open_grids
  for grid_group in grids:
    if isinstance(grid_group, (tuple, list)):
      # Handle multiple channels or time series
      from os.path import commonprefix
      gname = commonprefix([g.name for g in grid_group])
      if len(gname) == 0:
        gname = name
      gmodels, gmsg = open_grids(session, grid_group, gname, **kw)
      models.extend(gmodels)
      msg_lines.append(gmsg)
    else:
      sgrids.append(grid_group)

  if sgrids:
    smodels, smsg = open_grids(session, sgrids, name, **kw)
    models.extend(smodels)
    msg_lines.append(smsg)

  msg = '\n'.join(msg_lines)

  return models, msg

# -----------------------------------------------------------------------------
#
from chimerax.map.data import MapFileFormat
class DICOMMapFormat(MapFileFormat):
    def __init__(self):
        MapFileFormat.__init__(self, 'DICOM image', 'dicom', ['dicom'], ['dcm'],
                               batch = True, allow_directory = True)

    @property
    def open_func(self):
        return self.open_dicom_grids
    
    def open_dicom_grids(self, paths, log = None, verbose = False):

        if isinstance(paths, str):
            paths = [paths]
        from .dicom_grid import dicom_grids
        grids = dicom_grids(paths, log = log, verbose = verbose)
        return grids

# -----------------------------------------------------------------------------
#
def register_dicom_format(session):
    fmt = DICOMMapFormat()

    # Register file suffix
    # TODO: Do this in bundle_info.xml once it has allow_directory option.
    suf = tuple('.' + s for s in fmt.suffixes)
    from chimerax.core import io, toolshed
    io.register_format(fmt.description, toolshed.VOLUME, suf, nicknames=fmt.prefixes,
                       open_func=open_dicom, batch=fmt.batch, allow_directory=fmt.allow_directory)
    
    # Add map grid format reader
    from chimerax.map import add_map_format
    add_map_format(session, fmt)
