"""
Fancy containing of geometrical objects
---------------------------------------
"""
from chimera2 import universe, shapes
import numpy
from chimera2.math3d import Identity

from chimera2.arrayattr import AttributeItem, Group

class Sphere(object):
    def __init__(self, radius = 0.0, center = None):
        self.radius = radius
        if center is None:
            center = Point(0.0, 0.0, 0.0)
        self.center = center
shapes.Sphere.register(Sphere)
universe.Item.register(Sphere)

class Cylinder(object):
    def __init__(self, radius = 0.0, end_points = None):
        self.radius = radius
        if end_points is None:
            end_points = (Point(0.0, 0.0, 0.0), Point(0.0, 0.0, 0.0))
        self.end_points = end_points
shapes.Cylinder.register(Cylinder)
universe.Item.register(Cylinder)

Sphere_info = {
    'radius': AttributeItem('d', True),
    'center': AttributeItem('3d', True),
    'color': AttributeItem('3d', False),
}

Cylinder_info = {
    'radius': AttributeItem('d', True),
    'end_points': AttributeItem('(2, 3)d', True),
    'color': AttributeItem('3d', False),
}

class Transform(Group):

    attribute_info = {
            shapes.Sphere: Sphere_info,
            shapes.Cylinder: Cylinder_info,
    }

    def __init__(self, *args, **kw):
        Group.__init__(self, *args, **kw)
        self.xform = self.path_xform = Identity()

    def append(self, item, item_type=None):
        p = Group.append(self, item, item_type)
        if isinstance(p, Transform):
            p.path_xform = self.path_xform * p.xform
        return p
