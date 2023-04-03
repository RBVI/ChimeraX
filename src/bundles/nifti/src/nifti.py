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
import math

import nibabel

from chimerax.map.volume import open_grids
from chimerax.map_data import GridData, allocate_array
from chimerax.dicom.coordinates import get_coordinate_system

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
        # Data is often in x-y-z but we need z-y-x
        self.data_size = data.shape[::-1]
        self.images = data.get_fdata()
        # TODO: Get the rotation from the NifTI file
        affine = data.affine
        self.coordinate_system = get_coordinate_system("".join(nibabel.orientations.aff2axcodes(affine)))
        self.scale = [
              math.sqrt(affine[0][0]**2 + affine[1][0]**2 + affine[2][0]**2)
            , math.sqrt(affine[0][1] ** 2 + affine[1][1] ** 2 + affine[2][1] ** 2)
            , math.sqrt(affine[0][2] ** 2 + affine[1][2] ** 2 + affine[2][2] ** 2)
        ]
        self.center = [affine[0][3], affine[1][3], affine[2][3]]
        self.data_rotation =[
                    [affine[0][0] / self.scale[0], affine[0][1] / self.scale[1], affine[0][2] / self.scale[2]]
                , [affine[1][0] / self.scale[0], affine[1][1] / self.scale[1], affine[1][2] / self.scale[2]]
                , [affine[2][0] / self.scale[0], affine[2][1] / self.scale[1], affine[2][2] / self.scale[2]]
            ]
        self.data_type = data.dataobj.dtype


class NiftiGrid(GridData):
    def __init__(self, nifti, time = None, channel = None):
        self.nifti_data = nifti
        GridData.__init__(
            self, nifti.data_size, nifti.data_type
            , nifti.center, rotation=nifti.data_rotation, step = nifti.scale
            # , path = ???, name = ???
            , file_type = 'nifti' #, time, channel ???
        )

    def read_matrix(self, ijk_origin = (0,0,0), ijk_size = None,
                  ijk_step = (1,1,1), progress = None):
        return self.nifti_data.images
        # self.initial_plane_display = True ?
