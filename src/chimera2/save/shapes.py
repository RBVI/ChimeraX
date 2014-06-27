"""
shapes: Abstract Base Classes for Common Geometrical Shapes in the Universe
===========================================================================

Provide abstract base classes for common geometrical shapes
to allow for alternate implementations to share a common API.
"""
__all__ = ['Sphere', 'Cylinder', 'Mesh']

from abc import ABCMeta, abstractproperty
from .universe import Item

class Sphere(Item):
    """A sphere"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def radius(self):
        raise NotImplemented

    @abstractproperty
    def center(self):
        raise NotImplemented

class Cylinder(Item):
    """A cylinder"""
    __metaclass__ = ABCMeta

    #TODO: independent top and bottom caps?

    @abstractproperty
    def radius(self):
        raise NotImplemented

    @abstractproperty
    def end_points(self):
        raise NotImplemented

    @abstractproperty
    def cap(self):
        raise NotImplemented

class Mesh(Item):
    """A triangle mesh"""
    __metaclass__ = ABCMeta

    INDEPENDENT = 0	# Independent triangles
    STRIP = 1		# Strip of triangles
    FAN = 2		# Triangle fan

    @abstractproperty
    def mode(self):
        raise NotImplemented

    @abstractproperty
    def vertices(self):
        raise NotImplemented

    @abstractproperty
    def indices(self):
        raise NotImplemented
