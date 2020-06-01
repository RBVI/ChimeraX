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
# Read a SITUS ascii density map file.
#

# -----------------------------------------------------------------------------
#
class SITUS_Density_Map:

  def __init__(self, path):

    self.path = path

    f = open(path, 'r')
    params = f.readline()
    self.data_offset = f.tell()
    f.close()

    fields = params.split()
    if len(fields) < 7:
      raise SyntaxError('First line of SITUS map must have 7 parameters: voxel size, origin xyz, grid size xyz, got\n' + params)

    try:
      self.voxel_size = float(fields[0])
      self.origin = [float(x) for x in fields[1:4]]
      self.grid_size = [int(s) for s in fields[4:7]]
    except Exception:
      raise SyntaxError('Error parsing first line of SITUS map: voxel size, origin xyz, grid size xyz, got\n' + params)

    if self.voxel_size == 0:
      raise SyntaxError('Error parsing first line of SITUS map: first value is voxel size and must be non-zero, got line\n' + params)

  # ---------------------------------------------------------------------------
  #
  def matrix(self, progress):

    from ..readarray import read_text_floats
    data = read_text_floats(self.path, self.data_offset, self.grid_size,
                              progress = progress)
    return data
