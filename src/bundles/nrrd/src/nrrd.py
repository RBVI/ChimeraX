# vim: set expandtab shiftwidth=4 softtabstop=4:
import chimerax.core.session
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

import nrrd
import numpy as np
import os
from chimerax.core.session import Session
from chimerax.map.volume import open_grids
from chimerax.map_data import GridData
from chimerax.image_formats.open_image import ImageSurface
from chimerax.dicom.coordinates import get_coordinate_system
from chimerax.dicom.types import Segmentation

class NRRD:
    def __init__(self, session, data):
        self.session = session
        self.paths = data
        self.nrrds = []
        for path in data:
            # nrrd reads elements in Fortran order by default, while numpy and other
            # Python libraries typically use C-order
            # What this means in English is that the data is in x-y-z order and
            # we needed it in z-y-x order.
            # See https://pynrrd.readthedocs.io/en/stable/background/how-to-use.html
            img, hdr = nrrd.read(path, index_order='C')
            self.nrrds.append(NRRDData(self.session, hdr, img, path))

    @classmethod
    def from_paths(cls, session, data):
        return cls(session, data)

    def open(self):
        models, message = [], ""
        for nrd in self.nrrds:
            if len(nrd.shape) == 2:
                models.append(ImageSurface(self.session, nrd.name, nrd.image, nrd.shape[0], nrd.shape[1]))
            else:
                grids = []
                for nrd in self.nrrds:
                    g = NRRDGrid(nrd)
                    grids.append(g)
                    mods, msg = open_grids(self.session, grids, name=nrd.name)
                    models.extend(mods)
                    message += msg
        return models, message

class NRRDData:
    """A wrapper over nrrd."""
    def __init__(self, session: Session, header, data: np.ndarray, path = None):
        self.session = session
        self._path = path
        self._raw_header = header
        self._raw_data = data
        self._transformed_data = None
        self._spacings = None
        self._name = None
        self._image = None
        self._coordinate_system = None

    @property
    def name(self):
        if not self._name:
            name = self._raw_header.get("content", None)
            if not name:
                name = os.path.split(self._path)[1]
            if not name:
                self._name = "Unknown"
        return self._name

    @property
    def image(self):
        if self._image is None:
            self._image = self._raw_data
        return self._image

    @property
    def dimension(self):
        """Number of dimensions in this NRRD image. 1 for univariate histograms,
        2 for grayscale images, 3 for volumes and color images, 4 for
        time-varied images"""
        return self._raw_header.get("dimension", None)

    @property
    def shape(self):
        # _raw_data.shape and _raw_header.sizes should be the same data
        return self._raw_header['sizes']

    @property
    def data_type(self):
        return self._raw_data.dtype

    @property
    def origin(self):
        return self._raw_header.get("space origin", (0,0,0))

    @property
    def rotation(self):
        return self.coordinate_system.rotation_matrix

    # @property
    # def columns(self):
    #     ...

    # @property
    # def rows(self):
    #     ...

    @property
    def pixel_spacing(self):
        if not self._spacings:
            x, y, z = self.coordinate_system.space_ordering
            if 'spacings' in self._raw_header:
                spacing_vector = self._raw_header['spacings']
                spacings = [
                    spacing_vector[x]
                    , spacing_vector[y]
                    , spacing_vector[z]
                ]
            elif 'space directions' in self._raw_header:
                space_and_direction_matrix = self._raw_header['space directions']
                spacings = [
                    space_and_direction_matrix[x][x]
                    , space_and_direction_matrix[y][y]
                    , space_and_direction_matrix[z][z]
                ]
            else:
                spacings = [1] * self.dimension
            self._spacings = spacings
        return self._spacings

    @property
    def coordinate_system(self):
        if not self._coordinate_system:
            space = self._raw_header.get('space', None)
            if space:
                self._coordinate_system = get_coordinate_system(space)
            else:
                # if self.dimension...
                self._coordinate_system = get_coordinate_system("LAS")
        return self._coordinate_system


class NRRDGrid(GridData):
    def __init__(self, nrrd, time = None, channel = None):
        self.nrrd_data = nrrd
        GridData.__init__(
            self, nrrd.shape, nrrd.data_type, origin = nrrd.origin
            , rotation = nrrd.rotation
            , step = nrrd.pixel_spacing, file_type = 'nrrd'
        )

    def read_matrix(self, ijk_origin = (0,0,0), ijk_size = None,
                  ijk_step = (1,1,1), progress = None):
        return self.nrrd_data.image[::ijk_step[0], ::ijk_step[1], ::ijk_step[2]]

    def pixel_spacing(self) -> tuple[float, float, float]:
        return self.nrrd_data.pixel_spacing
 
    def inferior_to_superior(self) -> bool:
        return False
    
    def segment(self, number) -> 'NRRDSegmentation':
        return NRRDSegmentation(self.nrrd_data, number)


class NRRDSegmentation(GridData, Segmentation):
    def __init__(self, nrrd, time = None, channel = None, number = 0):
        self.nrrd_data = nrrd
        GridData.__init__(
            self, nrrd.shape, nrrd.data_type, origin = nrrd.origin
            , rotation = nrrd.rotation
            , name = "segmentation %d" % number
            , step = nrrd.pixel_spacing, file_type = 'nrrd'
        )
        self.segment_array = np.zeros(nrrd.shape, dtype = np.uint8)

    def read_matrix(self, ijk_origin = (0,0,0), ijk_size = None,
                  ijk_step = (1,1,1), progress = None):
        return self.segment_array[::ijk_step[0], ::ijk_step[1], ::ijk_step[2]]
    
    def pixel_spacing(self) -> tuple[float, float, float]:
        return self.reference_data.pixel_spacing
    
    def inferior_to_superior(self) -> bool:
        return False

    def save(filename) -> None:
        pass
