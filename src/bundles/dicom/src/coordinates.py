"""
This file contains matrices necessary to transform medical imaging coordinate systems
such that the images are rendered in ChimeraX as if standing in front of the subject.

ChimeraX's coordinate system is a right-handed one. In the default view, the
Z-axis comes out at the viewer, the X axis increases towards the viewer's right,
and the Y axis increases towards the top of the viewer's screen. According to
http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm this would be
described in medical imaging terms as "LSA".

This can be the case for scanners. Take an MRI: by convention, Z
increases towards the back of the scanner. X increases, when facing the inlet,
towards the right. Y increases towards the earth. Rotate ChimeraX's X-Z plane
180 degrees, and indeed,

LPS -- Left-Posterior-Superior: x increases to the patient left, y increases to the earth,
z increases towards the head. Imagine yourself looking into an MRI from the inlet port.
"""
from typing import Optional
from enum import Enum
from scipy.ndimage.interpolation import rotate

def get_coordinate_system(coords: Optional[str] = None) -> 'CoordinateSystem':
    """Given a string representing a coordinate system, return a CoordinateSystem object.
    Possible values are RAS, RAST, LAS, LAST, LPS, LPST, scanner-xyz, scanner-xyz-time,
    3d-right-handed, 3d-right-handed-time, 3d-left-handed, or 3d-left-handed-time.
    If no parameter is given a default coordinate system, 3DRightHanded, will be returned.
    """
    if not coords:
        return RightHanded3D()
    if coords == "right-anterior-superior" or coords == "RAS":
        sys = RAS()
    elif coords == "right-anterior-superior-time" or coords == "RAST":
        sys = RAS()
        sys.have_time_axis = True
    elif coords == "left-anterior-superior" or coords == "LAS":
        sys = LAS()
    elif coords == "left-anterior-superior-time" or coords == "LAST":
        sys = LAS()
        sys.have_time_axis = True
    elif coords == "left-posterior-superior" or coords == "LPS":
        sys = LPS()
    elif coords == "left-posterior-superior-time" or coords == "LPST":
        sys = LPS()
        sys.have_time_axis = True
    elif coords == "scanner-xyz":
        sys = ScannerXYZ()
    elif coords == "scanner-xyz-time":
        sys = ScannerXYZ()
        sys.have_time_axis = True
    elif coords == "3D-right-handed":
        sys = RightHanded3D()
    elif coords == "3D-right-handed":
        sys = RightHanded3D()
        sys.have_time_axis = True
    elif coords == "3D-left-handed":
        sys = LeftHanded3D()
    elif coords == "3D-left-handed-time":
        sys = LeftHanded3D()
        sys.have_time_axis = True
    else:
        raise ValueError(f"Unknown coordinate system: {coords}")
    return sys

class Handedness(Enum):
    RIGHT = 0
    LEFT = 1

class CoordinateSystem:
    def __init__(self):
        self.have_time_axis = False
        self.handedness = Handedness.RIGHT
        self.patient_based = True

    @staticmethod
    def to_xyz(array):
        return array

    @property
    def space_ordering(self):
        return 1, 2, 3

    @property
    def dimension(self):
        # 3 or 4
        return 3 + int(self.have_time_axis)

class LPS(CoordinateSystem):
    @staticmethod
    def to_xyz(array):
        return rotate(array, 90, axes=(0,1))

    @property
    def space_ordering(self):
        # We swap the Y and Z axes when we do the rotation, so
        # we need to apply the Z spacing to Y and vice versa
        return 0, 2, 1

class RAS(CoordinateSystem):
    pass

class LAS(CoordinateSystem):
    def __init__(self):
        super().__init__()
        self.handedness = Handedness.LEFT


class ScannerXYZ(CoordinateSystem):
    def __init__(self):
        super().__init__()
        self.patient_based = False

class RightHanded3D(CoordinateSystem):
    @staticmethod
    def to_xyz(array):
        return rotate(array, 90, axes=(0,1))

    @property
    def space_ordering(self):
        return 0, 2, 1

class LeftHanded3D(CoordinateSystem):
    pass
