"""
simple: Basic geometrical objects
---------------------------------
"""
from chimera2.math3d import Point, Identity
from chimera2 import universe, shapes

class Sphere(object):
    __slots__ = ['container', 'radius', 'center', 'color']
    def __init__(self, radius = 0.0, center = None):
        self.radius = radius
        if center is None:
            center = Point(0.0, 0.0, 0.0)
        self.center = center
        self.color = None

    def xformed_center(self):
        return container.path_xform * self.center
shapes.Sphere.register(Sphere)
universe.Item.register(Sphere)

class Cylinder(object):
    """A cylinder"""

    __slots__ = ['container', 'radius', 'end_points', 'color']
    def __init__(self, radius = 0.0, end_points = None, cap = False):
        self.radius = radius
        if end_points is None:
            end_points = (Point(0.0, 0.0, 0.0), Point(0.0, 0.0, 0.0))
        self.end_points = end_points
        # TODO: self.cap = cap
        self.color = None

    def xformed_endpoints(self):
        return container.path_xform * self.end_points
shapes.Cylinder.register(Cylinder)
universe.Item.register(Cylinder)

class Mesh(object):
    """A triangle mesh"""

    def __init__(self):
        self.mode = TRIANGLES
        self.vertices = []
        self.indices = []	# optional list of triplets of indices
shapes.Mesh.register(Mesh)
universe.Item.register(Mesh)

class Transform(universe.Group):

    def __init__(self):
        universe.Group.__init__(self)
        self.xform = self.path_xform = Identity()

    def append(self, item):
        """Place an item in the group"""
        p = universe.Group.append(self, item)
        if isinstance(p, Transform):
            p.path_xform = self.path_xform * p.xform
        return p
