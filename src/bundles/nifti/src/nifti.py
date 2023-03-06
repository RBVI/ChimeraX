# vim: set expandtab shiftwidth=4 softtabstop=4:

#  === UCSF ChimeraX Copyright ===
#  Copyright 2023 Regents of the University of California.
#  All rights reserved.  This software provided pursuant to a
#  license agreement containing restrictions on its disclosure,
#  duplication and use.  For details see:
#  https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
#  This notice must be embedded in or attached to all copies,
#  including partial copies, of the software or any revisions
#  or derivations thereof.
#  === UCSF ChimeraX Copyright ===

import nibabel
from chimerax.map.volume import open_grids
from chimerax.map_data import GridData, allocate_array

class NifTI:
    def __init__(self, session, data):
        self.session = session
        self.paths = data
        self.nifti_images = [nibabel.load(file) for file in data]

    @classmethod
    def from_paths(cls, session, data):
        return cls(session, data)

    def open(self):
        grids = []
        for image in self.nifti_images:
            d = NiftiData(image)
            g = NiftiGrid(d)
            grids.append(g)
        models, message = open_grids(self.session, grids, name="FOO")
        return models, message

class NiftiData:
    def __init__(self, data):
        self._raw_data = data
        self.data_size = data.shape
        self.images = data.get_fdata()
        # TODO: Get the rotation from the NifTI file
        affine = data.affine
        self.data_rotation = [
            [affine[0,0], 0, 0]
            , [0, affine[1,1], 0]
            , [0, 0, affine[2,2]]
        ]
        self.center = [affine[0,3], affine[1,3], affine[2,3]]
        # Affine matrix is a 4 x 4 matrix
        # --          --
        # | a  0  0  x |
        # | 0  b  0  y |
        # | 0  0  c  z |
        # | 0  0  0  1 |
        # --          --
        # Where x, y, and z are translations
        # the sign of the 3x3 diagonal determines direction
        # and the magnitudes of the 3x3 diagonal are scaling factors
        self.data_type = data.dataobj.dtype


class NiftiGrid(GridData):
    def __init__(self, nifti, time = None, channel = None):
        self.nifti_data = nifti
        GridData.__init__(
            self, nifti.data_size, nifti.data_type
            , nifti.center, rotation=nifti.data_rotation #, data_step = ???, rotation = ???
            # , path = ???, name = ???
            , file_type = 'nifti' #, time, channel ???
        )

    def read_matrix(self, ijk_origin = (0,0,0), ijk_size = None,
                  ijk_step = (1,1,1), progress = None):
        return self.nifti_data.images
        # self.initial_plane_display = True ?
