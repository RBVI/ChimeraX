"""
Compare filling a registered attribute versus an unregistered one
"""
import sys
sys.path.insert(0, '..')
from chimera2 import universe
import fancy as geom
from chimera2 import shapes

#import pdbmtx_atoms as data
#import pdb3fx2_atoms as data
#import pdb3k9f_atoms as data
import pdb3cc4_atoms as data
# populate universe with Spheres, Cylinders, and Meshes
def build_group():
    # Store directly into Aggregator arrays
    group = geom.Transform()

    sphere_count = 0
    cylinder_count = 0
    for item in data.data:
        if item[0] == 's':
            sphere_count += 1
        elif item[0] == 'c':
            cylinder_count += 1
    group.reserve(shapes.Sphere, sphere_count)
    group.reserve(shapes.Cylinder, cylinder_count)

    spheres = group.items(shapes.Sphere)
    sphere_radii = spheres.arrays['radius']
    sphere_center = spheres.arrays['center']
    sphere_color = spheres.arrays['color']
    s = spheres.populate_proxies(sphere_count)

    cylinders = group.items(shapes.Cylinder)
    cylinder_radii = cylinders.arrays['radius']
    cylinder_ends = cylinders.arrays['end_points']
    c = cylinders.populate_proxies(cylinder_count)

    from chimera2.math3d import Point, Xform
    for item in data.data:
        if item[0] == 's':
            # sphere: 's', radius, [x, y, z], [r, g, b, a]
            sphere_radii[s] = item[1]
            sphere_center[s] = item[2]
            #sphere_color[s] = (0, 0, 0)
            s += 1
        elif item[0] == 'c':
            # cylinder: 'c', radius, height, mat4x3, [r, g, b, a]
            xf = Xform(item[3])
            ep0 = xf * Point([0, -item[2], 0])
            ep1 = xf * Point([0, item[2], 0])
            cylinder_radii[c] = item[1]
            cylinder_ends[c] = (ep0, ep1)
            c += 1
    return group

import gc
gc.disable()

from memory import memory
m0 = memory()
group = build_group()
m1 = memory(m0)
print m1 / 1024, 'KiB', m1 / (1024 * 1024), 'MiB'
universe.add(group)
print group.count(shapes.Sphere), 'spheres', group.count(shapes.Cylinder), 'cylinders'

import numpy
colormap = (
        (0, (0., 0., 0.)),
        (0.4, (1., 0., 0.)),
        (0.46, (0., 1., 0.)),
        (float('inf'),  (0., 0., 1.)),
)

def attribute_style():
    spheres = universe.select_items(shapes.Sphere)
    for s in spheres:
        s.color2 = (1., 1., 1.)

def array_style():
    Color_info = {
            'color3': geom.AttributeItem('3d'),
    }
    from chimera2 import shapes
    group.register(shapes.Sphere, Color_info)
    spheres = universe.select_items(shapes.Sphere)
    spheres.attr('color3').set((1., 1., 1.))

import timeit

m0 = memory()
t0 = timeit.timeit(attribute_style, number=20)
print t0
m1 = memory(m0)
print m1 / 1024, 'KiB', m1 / (1024 * 1024), 'MiB'

m0 = memory()
t1 = timeit.timeit(array_style, number=20)
print t1
m1 = memory(m0)
print m1 / 1024, 'KiB', m1 / (1024 * 1024), 'MiB'

print '%fX faster' % (t0 / t1)
