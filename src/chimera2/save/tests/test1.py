"""
Benchmark adding colors to sphere and cylinders
"""
import sys
sys.path.insert(0, '..')
from chimera2 import universe
#import simple as geom
import fancy as geom

#import pdbmtx_atoms as data
import pdb3fx2_atoms as data
#import pdb3k9f_atoms as data
#import pdb3cc4_atoms as data
# populate universe with Spheres, Cylinders, and Meshes
def build_group():
    group = geom.Transform()

    sphere_count = 0
    cylinder_count = 0
    for item in data.data:
        if item[0] == 's':
            sphere_count += 1
        elif item[0] == 'c':
            cylinder_count += 1
    group.reserve(geom.Sphere, sphere_count)
    group.reserve(geom.Cylinder, cylinder_count)

    from chimera2.math3d import Point, Xform
    for item in data.data:
        if item[0] == 's':
            # sphere: 's', radius, [x, y, z], [r, g, b, a]
            sphere = geom.Sphere(item[1], item[2])
            group.append(sphere)
        elif item[0] == 'c':
            # cylinder: 'c', radius, height, mat4x3, [r, g, b, a]
            xf = Xform(item[3])
            ep0 = xf * Point([0, -item[2], 0])
            ep1 = xf * Point([0, item[2], 0])
            cylinder = geom.Cylinder(item[1], (ep0, ep1))
            group.append(cylinder)
    return group

from memory import memory
m = memory()
group = build_group()
print group.count(geom.Sphere), 'spheres', group.count(geom.Cylinder), 'cylinders'
universe.add(group)

#import copy
#universe.add(copy.copy(group))
#universe.add(copy.copy(group))
#sphere_colors = sphere_colors * 3
#cylinder_colors = cylinder_colors * 3

if 0:
    import numpy
    Color_info = {
            'color': geom.AttributeItem('3d'),
    }
    from chimera2 import shapes
    group.register(shapes.Sphere, Color_info)
    group.register(shapes.Cylinder, Color_info)

def add_colors():
    from chimera2 import shapes
    spheres = universe.select_items(shapes.Sphere)
    spheres.attr('color').set((1., 1., 1.))

    cylinders = universe.select_items(shapes.Cylinder)
    cylinders.attr('color').set((1., 1., 1.))

add_colors()
mem = memory(m)
print mem / 1024, 'KiB', mem / (1024 * 1024), 'MiB'

if 1:
    import timeit
    print timeit.timeit(add_colors, number=20)

if 0:
    import cProfile
    cProfile.run('add_colors()', 'profile.out')
