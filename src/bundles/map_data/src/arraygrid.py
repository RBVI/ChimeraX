# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# ArrayGridData objects wraps a NumPy array for use by volume viewer.
#
from . import GridData

# -----------------------------------------------------------------------------
# Constructor requires NumPy arrays with indices in z, y, x order.
# Origin and step parameters are in x, y, z order.
#
class ArrayGridData(GridData):
  '''
  Supported API.  Create a GridData from a 3-dimensional numpy array.

  Attributes
  ----------
  array : 3D numpy array
      Data array can be any scalar numpy type.
  origin : 3 floats
      Position of grid index (0,0,0) in physical coordinates (x,y,z) (usually Angstroms).
      Default (0,0,0).
  step : 3 floats
      Grid plane spacing along x,y,z axes.  Default (1,1,1)
  cell_angles : 3 floats
      Cell angles for skewed crystallography maps.
      Angles (alpha,beta,gamma) between yz, xz, and xy axes in degrees.
      Default (90,90,90).
  rotation : 3x3 matrix
      Rotation matrix in physical coordinates. Default ((1,0,0),(0,1,0),(0,0,1))
  symmetries : :class:`~.chimerax.geometry.Places` or None
      Symmetry transformations that map the to itself in physical coordinates.
      Default None means no symmetries.
  name : string
      Descriptive name.  Default ''
  '''

  def __init__(self,
               array,
               origin = (0,0,0),
               step = (1,1,1),
               cell_angles = (90,90,90),
               rotation = ((1,0,0),(0,1,0),(0,0,1)),
               symmetries = (),
               name = ''):
      
      self.array = array
      
      path = ''
      file_type = ''
      component_name = ''

      grid_size = list(array.shape)
      grid_size.reverse()

      value_type = array.dtype

      GridData.__init__(self, grid_size, value_type,
                        origin, step, cell_angles = cell_angles,
                        rotation = rotation, symmetries = symmetries,
                        name = name, path = path, file_type = file_type)

      self.writable = True
  
  # ---------------------------------------------------------------------------
  #
  def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
  
    return self.cached_data(ijk_origin, ijk_size, ijk_step)

  # ---------------------------------------------------------------------------
  #
  def cached_data(self, ijk_origin, ijk_size, ijk_step):

      m = self.matrix_slice(self.array, ijk_origin, ijk_size, ijk_step)
      return m
