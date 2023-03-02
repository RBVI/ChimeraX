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

import nrrd
import numpy as np
import os
from scipy.ndimage import rotate
from enum import Enum
from chimerax.map.volume import open_grids
from chimerax.map_data import GridData
from chimerax.image_formats.open_image import ImageSurface

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
            self.nrrds.append(NRRDData(hdr, img, path))

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
    def __init__(self, header: nrrd.NRRDHeader, data: np.ndarray, path = None):
        self._path = path
        self._raw_header = header
        self._raw_data = data
        self._transformed_data = None
        self._spacings = None
        self._name = None

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
        return self._raw_data

    @property
    def dimension(self):
        """Number of dimensions in this NRRD image. 1 for univariate histograms,
        2 for grayscale images, 3 for volumes and color images, 4 for
        time-varied images"""
        return self._raw_header.get("dimension", None)

    @property
    def shape(self):
        # _raw_data.shape and _raw_header.sizes should be the same data
        return self._raw_data.shape

    @property
    def data_type(self):
        return self._raw_data.dtype

    @property
    def origin(self):
        return self._raw_header.get("space origin", (0,0,0))

    @property
    def pixel_spacing(self):
        if not self._spacings:
            if 'spacings' in self._raw_header:
                spacings = self._raw_header['spacings']
            elif 'space directions' in self._raw_header:
                space_and_direction_matrix = self._raw_header['space directions']
                spacings = [
                    space_and_direction_matrix[1][1]
                    , space_and_direction_matrix[0][0]
                    , space_and_direction_matrix[2][2]
                ]
            else:
                spacings = [1]*self.dimension
            self._spacings = spacings
        return self._spacings

    def _axis_corrected_spacing(self):
        pass


class NRRDGrid(GridData):
    def __init__(self, nrrd, time = None, channel = None):
        self.nrrd_data = nrrd
        GridData.__init__(
            self, nrrd.shape, nrrd.data_type, origin = nrrd.origin
            , step = nrrd.pixel_spacing, file_type = 'nrrd'
        )

    def read_matrix(self, ijk_origin = (0,0,0), ijk_size = None,
                  ijk_step = (1,1,1), progress = None):
        return self.nrrd_data.image
