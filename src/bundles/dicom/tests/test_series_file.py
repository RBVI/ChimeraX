import os
import pytest
from pydicom import dcmread
from chimerax.dicom import SeriesFile

test_dicom_filename = os.path.join(os.path.dirname(__file__), "data", "img0001--67.1456.dcm")


@pytest.fixture
def series_file():
    return SeriesFile(dcmread(test_dicom_filename))


@pytest.fixture
def raw_dicom():
    return dcmread(test_dicom_filename)


def test_path(series_file, raw_dicom):
    assert series_file.path == test_dicom_filename
    assert series_file.path == raw_dicom.filename


def test_position(series_file, raw_dicom):
    assert series_file.position == raw_dicom.get('ImagePositionPatient')


def test_orientation(series_file, raw_dicom):
    assert series_file.class_uid == raw_dicom.get('SOPClassUID')


def test_num_frames(series_file):
    assert series_file.multiframe is False
