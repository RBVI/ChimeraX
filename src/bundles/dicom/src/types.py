from enum import IntEnum
from typing import Protocol

class Direction(IntEnum):
    """Used by the SliceOverlay class to control whether to draw
    horizontally or vertically"""
    HORIZONTAL = 0
    VERTICAL = 1

class Axis(IntEnum):
    AXIAL = 2
    CORONAL = 1
    SAGITTAL = 0

    def __str__(self):
        return self.name.lower()

    @property
    def transform(self):
        if self.value == 2:
            return [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        elif self.value == 1:
            return [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        else:
            return [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

    @property
    def positive_direction(self):
        return [-1, 1][self.value != 1]

    @property
    def vertical(self):
        if self.value == 2:
            return 0
        else:
            return 2
        
    @property
    def horizontal(self):
        if self.value == 2:
            return 1
        else:
            return 0
        
    @property
    def cartesian(self):
        return ["x", "y", "z"][self.value]


class Segmentation(Protocol):
    def save(filename) -> None:
        raise NotImplementedError

