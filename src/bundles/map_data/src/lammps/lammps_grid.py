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
# Wrap LAMMPS grid3d data as grid data for displaying as surfaces, meshes, and volumes.
#
from .. import GridData

# -----------------------------------------------------------------------------
#
class LammpsGrid(GridData):
    """
    Reads LAMMPS grid3d format files produced by the 'dump grid' command in LAMMPS.
    """
    def __init__(self, path):

        from . import lammps_format
        d = lammps_format.Lammps_Grid_Data(path)
        
        self.lammps_data = d
        
        GridData.__init__(self, d.matrix_size, d.element_type, 
                         origin=d.origin, step=d.step,
                         path=path, file_type='grid3d',
                         name=d.name)
    
    # ---------------------------------------------------------------------------
    #
    def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
        return self.lammps_data.read_matrix(ijk_origin, ijk_size, ijk_step, progress)
