from numpy import (
    array, concatenate, empty, float32, int32
)

from chimerax.core.decorators import requires_gui
from chimerax.core.models import Model, Surface

from chimerax.map_data import GridData
from chimerax.map_data.readarray import allocate_array

class DicomContours(Model):
    def __init__(self, session, data, name):
        def rgb_255(cs):
            return tuple(int(c) for c in cs)

        def xyz_list(xs):
            a = tuple(float(x) for x in xs)
            xyz = array(a, float32).reshape(len(a) // 3, 3)
            return xyz

        if type(data) is not list:
            data = [data]
        self.dicom_series = data[0]

        if len(data) > 1:
            raise ValueError(
                'DICOM series has %d files, can only handle one file for "RT Structure Set Storage", '
                'file %s' % (len(series.paths), path)
            )

        Model.__init__(self, name, session)

        el = self.dicom_elements(
            self.dicom_series
            , {
                'StructureSetROISequence': {
                    'ROINumber': int
                    , 'ROIName': str
                }
                , 'ROIContourSequence': {
                    'ROIDisplayColor': rgb_255
                    , 'ContourSequence': {
                        'ContourGeometricType':  str
                        , 'NumberOfContourPoints': int
                        , 'ContourData': xyz_list
                    }
                }
            }
        )
        regions = []
        for rs, rcs in zip(el['StructureSetROISequence'], el['ROIContourSequence']):
            r = ROIContourModel(session, rs['ROIName'], rs['ROINumber'], rcs['ROIDisplayColor'], rcs['ContourSequence'])
            regions.append(r)

        self.add(regions)

    def dicom_elements(self, data, fields):
        values = {}
        for name, v in fields.items():
            d = data.get(name)
            if d is None:
                raise ValueError('Did not find %s' % name)
            if isinstance(v, dict):
                values[name] = [self.dicom_elements(e, v) for e in d]
            else:
                values[name] = v(d)
        return values


class ROIContourModel(Surface):
    def __init__(self, session, name, number, color, contour_info):
        Model.__init__(self, name, session)
        self.roi_number = number
        opacity = 255
        self.color = tuple(color) + (opacity,)
        va, ta = self._contour_lines(contour_info)
        self.set_geometry(va, None, ta)
        self.display_style = self.Mesh
        self.use_lighting = False

    def _contour_lines(self, contour_info):
        points = []
        triangles = []
        nv = 0
        for ci in contour_info:
            ctype = ci['ContourGeometricType']
            if ctype != 'CLOSED_PLANAR':
                # TODO: handle other contour types
                continue
            np = ci['NumberOfContourPoints']  # noqa unused var
            pts = ci['ContourData']
            points.append(pts)
            n = len(pts)
            tri = empty((n, 2), int32)
            tri[:, 0] = tri[:, 1] = range(n)
            tri[:, 1] += 1
            tri[n - 1, 1] = 0
            tri += nv
            nv += n
            triangles.append(tri)
        va = concatenate(points)
        ta = concatenate(triangles)
        return va, ta


class DicomGrid(GridData):
    initial_rendering_options = {
        'projection_mode':    '3d',
        'colormap_on_gpu':    True,
        'full_region_on_gpu': True
    }

    def __init__(self, d, time=None, channel=None):
        self.dicom_data = d
        GridData.__init__(
            self, d.data_size, d.value_type,
            d.data_origin, d.data_step, rotation=d.data_rotation,
            path=d.paths, name=d.name,
            file_type='dicom', time=time, channel=channel
            )
        if d.files_are_3d:
            # For fast access of multiple planes this reading method avoids
            # opening same dicom file many times to read planes.  50x faster in tests.
            self.read_matrix = self.dicom_read_matrix
        else:
            # For better caching if we read the whole plane, this method caches the
            # whole plane even if only part of the plane is needed.
            self.read_xy_plane = self.dicom_read_xy_plane
        self.multichannel = (channel is not None)
        self.initial_plane_display = True
        s = d.dicom_series
        if s.bits_allocated == 1 or s.dicom_class == 'Segmentation Storage':
            self.binary = True  # Use initial thresholds for binary segmentation
            self.initial_image_thresholds = [(0.5, 0), (1.5, 1)]
        else:
            self.initial_image_thresholds = [(-1000, 0.0), (300, 0.9), (3000, 1.0)]
        self.ignore_pad_value = d.pad_value

    # ---------------------------------------------------------------------------
    # If GridData.read_xy_plane() uses this method then whole planes are cached
    # even when a partial plane is requested.  The whole DICOM planes are always
    # read.  Caching them helps performance when say an xz-plane is being read.
    def dicom_read_xy_plane(self, k):
        c = self.channel if self.multichannel else None
        m = self.dicom_data.read_plane(k, self.time, c)
        return m

    # ---------------------------------------------------------------------------
    # If GridData.read_matrix() uses this method it only caches the actual requested
    # data.  For multiframe DICOM files this is much faster than reading separate
    # planes where the dicom data has to be opened once for each plane.
    # In a 512 x 512 x 235 test with 3d dicom segmentations this is 50x faster
    # than reading xy planes one at a time.
    def dicom_read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
        m = allocate_array(ijk_size, self.value_type, ijk_step, progress)
        c = self.channel if self.multichannel else None
        self.dicom_data.read_matrix(
            ijk_origin, ijk_size, ijk_step,
            self.time, c, m, progress
            )
        return m

    @requires_gui
    def show_info(self):
        from .ui import DICOMMetadata
        return DICOMMetadata.from_series(self.dicom_data.dicom_series.session, self.dicom_data.dicom_series)
