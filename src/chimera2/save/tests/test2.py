"""
Benchmark bulk creation of attributes
"""
import sys
sys.path.insert(0, '..')
from chimera2 import universe
import fancy as geom

#import pdbmtx_atoms as data
import pdb3fx2_atoms as data
#import pdb3k9f_atoms as data
#import pdb3cc4_atoms as data
# populate universe with Spheres, Cylinders, and Meshes
def build_group1():
    # build an object first, then store in into the Aggregator
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

def build_group2():
    # Store directly into Aggregator arrays
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

    spheres = group.items(geom.Sphere)
    sphere_radii = spheres.arrays['radius']
    sphere_center = spheres.arrays['center']
    s = spheres.populate_proxies(sphere_count)

    cylinders = group.items(geom.Cylinder)
    cylinder_radii = cylinders.arrays['radius']
    cylinder_ends = cylinders.arrays['end_points']
    c = cylinders.populate_proxies(cylinder_count)

    from chimera2.math3d import Point, Xform
    for item in data.data:
        if item[0] == 's':
            # sphere: 's', radius, [x, y, z], [r, g, b, a]
            sphere_radii[s] = item[1]
            sphere_center[s] = item[2]
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

import timeit
from memory import memory
m0 = memory()
t0 = timeit.timeit(build_group1, number=1)
print t0
m1 = memory(m0)
print m1 / 1024, 'KiB', m1 / (1024 * 1024), 'MiB'

m0 = memory()
t1 = timeit.timeit(build_group2, number=1)
print t1
m1 = memory(m0)
print m1 / 1024, 'KiB', m1 / (1024 * 1024), 'MiB'

print '%fX faster' % (t0 / t1)
