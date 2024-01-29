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
from chimerax.map_data import GridData
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
            d = NiftiData(self.session, image)
            g = NiftiGrid(d)
            grids.append(g)
        models, message = open_grids(self.session, grids, name="FOO")
        return models, message


class NiftiData:
    def __init__(self, session, data):
        self.session = session
        self._raw_data = data
        self.data_size = data.shape
        # Data is in x-y-z but we need z-y-x
        darray = data.get_fdata().transpose()
        self.images = darray
        # TODO: Get the rotation from the NifTI file
        affine = data.affine
        self.coordinate_system = get_coordinate_system(
            "".join(nibabel.orientations.aff2axcodes(affine))
        )
        self.scale = [
            math.sqrt(affine[0][0] ** 2 + affine[1][0] ** 2 + affine[2][0] ** 2),
            math.sqrt(affine[0][1] ** 2 + affine[1][1] ** 2 + affine[2][1] ** 2),
            math.sqrt(affine[0][2] ** 2 + affine[1][2] ** 2 + affine[2][2] ** 2),
        ]
        self.center = [affine[0][3], affine[1][3], affine[2][3]]
        # self.data_rotation =[
        #            [affine[0][0] / self.scale[0], affine[0][1] / self.scale[1], affine[0][2] / self.scale[2]]
        #        , [affine[1][0] / self.scale[0], affine[1][1] / self.scale[1], affine[1][2] / self.scale[2]]
        #        , [affine[2][0] / self.scale[0], affine[2][1] / self.scale[1], affine[2][2] / self.scale[2]]
        #    ]
        # We previously respected the rotation and that code is commented out above, but
        # respecting the rotation breaks the segmentation viewer and causes the 3D cursors
        # to appear in the wrong place.  So we just use the identity matrix for the rotation.
        self.data_rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.data_type = self.images.dtype
        self.slope, self.intercept = data.header.get_slope_inter()
        if self.slope is None:
            self.slope = 1
        if self.intercept is None:
            self.intercept = 0


class NiftiGrid(GridData):
    def __init__(self, nifti, time=None, channel=None):
        self.nifti_data = nifti
        GridData.__init__(
            self,
            nifti.data_size,
            nifti.data_type,
            nifti.center,
            rotation=nifti.data_rotation,
            step=nifti.scale,
            file_type="nifti",
        )

    def read_matrix(
        self, ijk_origin=(0, 0, 0), ijk_size=None, ijk_step=(1, 1, 1), progress=None
    ):
        array = self.nifti_data.images[
            ijk_origin[2] : ijk_origin[2] + ijk_size[2] : ijk_step[2],
            ijk_origin[1] : ijk_origin[1] + ijk_size[1] : ijk_step[1],
            ijk_origin[0] : ijk_origin[0] + ijk_size[0] : ijk_step[0],
        ]
        return array
        # if self.nifti_data.slope != 1:
        #    array *= self.nifti_data.slope
        # if self.nifti_data.intercept != 0:
        #    array += self.nifti_data.intercept

    def pixel_spacing(self) -> tuple[float, float, float]:
        return self.nifti_data.scale
    
    def inferior_to_superior(self) -> bool:
        return False

    def segment(self, number) -> 'NiftiSegmentation':
        return NiftiSegmentation(self.nifti_data, number = number)

class NiftiSegmentation(GridData, Segmentation):
    def __init__(self, nifti, time = None, channel = None, number = 1):
        self.reference_data = nifti
        GridData.__init__(
            self, nifti.data_size, nifti.data_type
            , nifti.center, rotation=nifti.data_rotation, step = nifti.scale
            # , path = ???
            , name = "segmentation %d" % number
            , file_type = 'nifti' #, time, channel ???
        )
        self.segment_array = zeros(self.reference_data.data_size[::-1], dtype=uint8)
        #self.initial_plane_display = True

    def pixel_spacing(self) -> tuple[float, float, float]:
        return self.reference_data.scale
 
    def inferior_to_superior(self) -> bool:
        return False

    def read_matrix(self, ijk_origin = (0,0,0), ijk_size = None,
                  ijk_step = (1,1,1), progress = None):
        array = self.segment_array[::ijk_step[0], ::ijk_step[1], ::ijk_step[2]]
        return array
        #if self.nifti_data.slope != 1:
        #    array *= self.nifti_data.slope
        #if self.nifti_data.intercept != 0:
        #    array += self.nifti_data.intercept
        #return array

    def save(filename):
        pass