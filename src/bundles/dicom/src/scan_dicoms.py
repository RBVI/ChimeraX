#
# Scan subdirectories finding all *.dcm files and compile information about each series
# such as the number of images, rows, columns, modality, bits, z-spacing, date, image type...
#
import pydicom
import sys

from pydicom.multival import MultiValue
from os import listdir
from os.path import dirname, isdir, isfile, join

dicom_attrs = ['AccessionNumber', 'BitsAllocated', 'BitsStored', 'Columns', 'HighBit', 'ImageOrientationPatient', 'ImagePositionPatient', 'ImageType', 'InstanceCreationDate', 'InstanceCreationTime', 'InstanceNumber', 'Modality', 'PatientBirthDate', 'PatientID', 'PatientName', 'PatientSex', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing', 'ReferringPhysicianName', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows', 'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SeriesDate', 'SeriesDescription', 'SeriesInstanceUID', 'SeriesNumber', 'SeriesTime', 'StudyDate', 'StudyID', 'StudyInstanceUID', 'StudyTime']

def find_dicom_attributes(path, file_attributes = None):
    if file_attributes is None:
        return_attributes = True
        file_attributes = []
    else:
        return_attributes = False

    if isdir(path):
        for p in listdir(path):
            find_dicom_attributes(join(path, p), file_attributes)
    elif isfile(path) and path.endswith('.dcm'):
        d = pydicom.dcmread(path)
        attr_values = {a: getattr(d, a) for a in dicom_attrs if hasattr(d, a)}
        attr_values['path'] = path
        file_attributes.append(attr_values)

    if return_attributes:
        series_attributes = collate_series_files(file_attributes)
        return series_attributes

def collate_series_files(file_attributes):
    aggregate_attrs = ['BitsStored', 'Columns', 'ImagePositionPatient',
                       'ImageType', 'Modality', 'PixelSpacing', 'PhotometricInterpretation', 'Rows',
                       'SamplesPerPixel', 'SeriesDescription', 'SeriesInstanceUID',
                       'StudyDate', 'StudyInstanceUID']
    series_values = {}
    for attrs in file_attributes:
        path = attrs.get('path')
        study_uid = attrs.get('StudyInstanceUID')
        series_uid = attrs.get('SeriesInstanceUID')
        if study_uid is not None and series_uid is not None:
            dir = dirname(path)
            key = (dir, study_uid, series_uid)
            if key not in series_values:
                series_values[key] = {'paths': set()}
            values = series_values[key]
            values['paths'].add(path)
            for aa in aggregate_attrs:
                desc = attrs.get(aa)
                if desc is not None:
                    if isinstance(desc, MultiValue):
                        desc = tuple(desc)
                    if aa not in values:
                        values[aa] = set()
                    values[aa].add(desc)
    return series_values

def series_report(series_values, top_directory, show_paths = True):
    # Group copies of same series
    sg = {}
    for attrs in series_values.values():
        sg.setdefault(one_value(attrs, 'SeriesInstanceUID'), []).append(attrs)

    series_counts = '%d unique series, %d total in %s' % (len(sg), len(series_values), top_directory)
    lines = [series_counts]
    for series_id, attrs in sg.items():
        a0 = attrs[0]
        lines.append('%s series %s date %s' % (one_value(a0, 'Modality'), series_id, one_value(a0, 'StudyDate')))
        for a in attrs:
            lines.append('  ' + formatted_series_info(a, top_directory, show_paths))
    return '\n'.join(lines)

def formatted_series_info(series_attrs, top_directory, show_paths = True):
    # return ['%s: %s' % (a, value.pop()) for a, value in series_attrs.items() if len(value) == 1]
    a = series_attrs
    x_spacing, y_spacing = [float(x) for x in one_value(a, 'PixelSpacing')]
    nz = len(a['paths'])
    z = [z for x, y, z in a.get('ImagePositionPatient', [])]
    if len(z) >= 2:
        z.sort()
        z_spacing = z[1] - z[0]
    else:
        z_spacing = 0
    fields = [
        one_value(a, 'SeriesDescription'),
        '%.3f x %.3f x %.3f mm' % (x_spacing, y_spacing, z_spacing),
        one_value(a, 'PhotometricInterpretation'),
        '%d x %d x %d' % (one_value(a, 'Columns'), one_value(a, 'Rows'), nz),
        '%d bits' % one_value(a, 'BitsStored'),
        # one_value(a,'Modality'),
        # one_value(a,'SeriesInstanceUID'),
    ]
    if show_paths:
        paths = tuple(a['paths'])
        prefix, suffix = common_prefix_and_suffix(paths)
        # print(prefix, suffix, paths)
        if prefix.startswith(top_directory):
            prefix = prefix[len(top_directory) + 1:]
        fields.append('%s*%s' % (prefix, suffix))

    return ', '.join(fields)

def one_value(attrs, name):
    if name in attrs:
        for s in attrs[name]:
            return s
        return '*'
    return '.'

# -----------------------------------------------------------------------------
#
def common_prefix_and_suffix(strings):
    prefix = suffix = strings[0]
    for s in strings[1:]:
        if not s.startswith(prefix):
            for i in range(min(len(prefix), len(s))):
                if s[i] != prefix[i]:
                    prefix = prefix[:i]
                    break
        if not s.endswith(suffix):
            for i in range(min(len(suffix), len(s))):
                if s[-1 - i] != suffix[-1 - i]:
                    suffix = suffix[len(suffix) - i:]
                    break
    return prefix, suffix


top_directory = sys.argv[1]
show_paths = (len(sys.argv) >= 3 and sys.argv[2] == 'paths')
series_attributes = find_dicom_attributes(top_directory)
print(series_report(series_attributes, top_directory, show_paths))
