import datetime
import io

from numpy import (
    array, zeros, concatenate, empty, float32, int32, uint8
)
import pydicom.uid

from pydicom.dataset import FileMetaDataset, Dataset
from pydicom.sequence import Sequence

from chimerax.core import version as chimerax_version
from chimerax.core.decorators import requires_gui
from chimerax.core.models import Model, Surface

from chimerax.map_data import GridData
from chimerax.map_data.readarray import allocate_array

from .. import __version__ as dicom_bundle_version
from ..types import Segmentation

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

    def pixel_spacing(self) -> tuple[float, float, float]:
        return self.dicom_data.pixel_spacing()
    
    def inferior_to_superior(self) -> bool:
        return self.dicom_data.inferior_to_superior()

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
    
    def segment(self, number) -> 'DicomSegmentation':
        return DicomSegmentation(self.dicom_data, self.time, self.channel, number)

    @requires_gui
    def show_info(self):
        from .ui import DICOMMetadata
        return DICOMMetadata.from_series(self.dicom_data.dicom_series.session, self.dicom_data.dicom_series)

class DicomSegmentation(GridData, Segmentation):
    initial_rendering_options = {
        'projection_mode':    '3d',
        'colormap_on_gpu':    True,
        'full_region_on_gpu': True
    }

    def __init__(self, d, time=None, channel=None, number = 1):
        self.reference_data = d
        name = " ".join(["segmentation", str(number)])
        GridData.__init__(
            self, d.data_size, d.value_type,
            d.data_origin, d.data_step, rotation=d.data_rotation,
            name = name,
            file_type='dicom', time=time, channel=channel
        )
        self.segment_array = zeros(d.data_size[::-1], dtype=uint8)
        #if d.files_are_3d:
        #    # For fast access of multiple planes this reading method avoids
        #    # opening same dicom file many times to read planes.  50x faster in tests.
        #    self.read_matrix = self.dicom_read_matrix
        #else:
        #    # For better caching if we read the whole plane, this method caches the
        #    # whole plane even if only part of the plane is needed.
        #    self.read_xy_plane = self.dicom_read_xy_plane
        self.multichannel = (channel is not None)
        self.initial_plane_display = False
        s = d.dicom_series
        if s.bits_allocated == 1 or s.dicom_class == 'Segmentation Storage':
            self.binary = True  # Use initial thresholds for binary segmentation
            self.initial_image_thresholds = [(0.5, 0), (1.5, 1)]
        else:
            self.initial_image_thresholds = [(-1000, 0.0), (300, 0.9), (3000, 1.0)]
        self.ignore_pad_value = d.pad_value

    def read_matrix(self, ijk_origin=(0, 0, 0), ijk_size=None, ijk_step=(1, 1, 1), progress=None):
        array = self.segment_array[::ijk_step[0], ::ijk_step[1], ::ijk_step[2]]
        return array
    
    def pixel_spacing(self) -> tuple[float, float, float]:
        return self.reference_data.pixel_spacing()
  
    def inferior_to_superior(self) -> bool:
        return self.reference_data.inferior_to_superior()
  
    def save(self, filename = None) -> None:
        header = FileMetaDataset()

        header.FileMetaInformationGroupLength = 182
        header.FileMetaInformationVersion = b'\x00\x01'
        # header.MediaStorageSOPClassUID 
        # header.MediaStorageSOPInstanceUID
        # header.TransferSyntaxUID
        # header.ImplementationClassUID
        # header.ImplementationVersionName

        ds = Dataset()
        ds.ImageType = ["DERIVED", "PRIMARY"]
        # ds.InstanceCreatorUID = ??
        ds.SOPClassUID = pydicom.uid.SegmentationStorage
        # ds.SOPInstanceUID = ??
        dt = datetime.datetime.now()
        date = dt.strftime('%Y%m%d')
        time = dt.strftime('%H%M%S.%f')
        ds.StudyDate = self.reference_data.dicom_series.get("StudyDate", "")
        ds.SeriesDate = self.reference_data.dicom_series.get("SeriesDate", "")
        ds.AcquisitionDate = date
        ds.ContentDate = date
        ds.StudyTime = self.reference_data.dicom_series.get("StudyTime", "")
        ds.SeriesTime = self.reference_data.dicom_series.get("SeriesTime", "")
        ds.AcquisitionTime = time
        ds.ContentTime = time
        # ds.AccessionNumber = ??
        ds.Modality = "SEG"
        ds.Manufacturer = "UCSF ChimeraX"
        ds.ReferringPhysicianName = ''
        ds.ManufacturerModelName = "https://www.github.com/RBVI/ChimeraX"
        # Patient's Name
        ds.PatientName = self.reference_data.dicom_series.get("PatientName", "")
        # Patient ID
        ds.PatientID = self.reference_data.dicom_series.get("PatientID", "")
        # Patient's Birth Date
        ds.PatientBirthDate = self.reference_data.dicom_series.get("PatientBirthDate", "")

        # Autogenerated code also tried to set:
        # Patient Sex, Patient Identity Removed, Device Serial Number, Software Versions, Study Instance UID,
        # Series Instance UID, Study ID, Series Number, Instance Number, Frame of Reference UID, 
        # Position Reference Indicator,
        # Number of Frames
        # Rows
        # Columns
        # Bits Stored
        # High Bit
        # Pixel Representation
        # Segmentation Type
        # Lossy Image Compression

        # Transfer Syntax
        ds.is_little_endian = self.reference_data.dicom_series.is_little_endian
        ds.is_implicit_VR = self.reference_data.dicom_series.is_implicit_VR

        # Series Instance UID -- ??
        # Study Instance UID -- get from parent data
        # Series Number -- ??
        # Study Number -- ??
        ds.StudyInstanceUID = self.reference_data.dicom_series.get("StudyInstanceUID", "")

        # Software Versions -- UCSF ChimeraX / DICOM bundle ver
        ds.SoftwareVersions = f"UCSF ChimeraX {chimerax_version}, DICOM bundle version {dicom_bundle_version}"
        
        # Content Creator's Name -- always UCSF ChimeraX
        ds.ContentCreatorName = "UCSF ChimeraX"
        
       
        # Always MONOCHROME2 for a 0-1 segmentation
        ds.PhotometricInterpretation = "MONOCHROME2"
        
        # Samples per Pixel -- always 1
        ds.SamplesPerPixel = 1

        # Always SegmentationStorage

        ds.SliceThickness = self.reference_data.dicom_series.get("SliceThickness", 1.0)
        ds.SliceSpacing = self.reference_data.dicom_series.get("SliceSpacing", 1.0) 
        ds.PixelSpacing = self.reference_data.dicom_series.get("PixelSpacing", [1.0, 1.0])
        ds.ImageOrientationPatient = self.reference_data.dicom_series.get("ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        refd_series_sequence = Sequence()
        ds.ReferencedSeriesSequence = refd_series_sequence

        refd_series = Dataset()
        refd_series.SeriesInstanceUID = self.reference_data.dicom_series.get("SeriesInstanceUID", "")
        refd_instance_sequence = Sequence()

        refd_series.ReferencedInstanceSequence = refd_instance_sequence

        per_frame_functional_groups_sequence = Sequence()

        ds.PerFrameFunctionalGroupsSequence = per_frame_functional_groups_sequence

        origin = self.reference_data.data_origin
        step = self.reference_data.data_step
        for index, slice in enumerate(self.segment_array):
            refd_instance = Dataset()
            refd_instance.ReferencedSOPClassUID = self.reference_data.files[index].get("SOPClassUID", "") 
            refd_instance.ReferencedSOPInstanceUID = self.reference_data.files[index].get("SOPInstanceUID", "")
            refd_instance_sequence.append(refd_instance)
            
            per_frame_functional_group = Dataset()
            frame_content_sequence = Sequence()
            plane_position_sequence = Sequence()
            per_frame_functional_group.FrameContentSequence = frame_content_sequence
            per_frame_functional_group.PlanePositionSequence = plane_position_sequence

            frame_content = Dataset()
            frame_content.StackID = "1"
            frame_content.InStackPositionNumber = index + 1
            frame_content.DimensionIndexValues = [1, index + 1]
            frame_content_sequence.append(frame_content)

            plane_position = Dataset()
            plane_position.ImagePositionPatient = [origin[0], origin[1], origin[2] + index * step[2]]
            plane_position_sequence.append(plane_position)
            per_frame_functional_groups_sequence.append(per_frame_functional_group)
        
        ds.PixelData = self.segment_array.tobytes()
        ds.file_meta = header 
        # ds.is_implicit_VR = ??
        # ds.is_little_endian = ??
    # List of fields we need to save a segmentation:
    # Series Description -- ask the user
    # Frame of Reference UID -- get from parent data