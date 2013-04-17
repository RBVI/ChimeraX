"""
Benchmark zone
"""
import sys
sys.path.insert(0, '..')
from chimera2 import universe
import fancy as geom
from chimera2 import shapes

#import pdbmtx_atoms as data
import pdb3fx2_atoms as data
#import pdb3k9f_atoms as data
#import pdb3cc4_atoms as data
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

def zone0():
    spheres = group.items(shapes.Sphere)
    zone_center = spheres[len(spheres) // 2].center
    zone_distsq = 5 * 5
    result = []
    for s in spheres:
        c = s.center
        xyz = c - zone_center
        distsq = xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]
        if distsq < zone_distsq:
            result.append(s)
    #print 'zone0', len(result)
    from chimera2.selection import SelectionSet
    return SelectionSet(result)

def zone1():
    spheres = group.items(shapes.Sphere)
    zone_center = spheres[len(spheres) // 2].center
    zone_distsq = 5 * 5
    centers = spheres.arrays['center']
    mask = ((centers - zone_center) ** 2).sum(axis=1) >= zone_distsq
    #print 'zone1', (~mask).sum()
    from chimera2.arrayattr import AggregateSelection
    return AggregateSelection(spheres, numpy.ma.filled(mask, True))

def zone2():
    spheres = group.items(shapes.Sphere)
    zone_center = spheres[len(spheres) // 2].center
    zone_distsq = 5 * 5
    centers = spheres.arrays['center']
    import numexpr
    #mask = ((centers - zone_center) ** 2).sum(axis=1) >= zone_distsq
    mask = numexpr.evaluate("((centers - zone_center) ** 2).sum(axis=1) >= zone_distsq")
    print 'zone2', (~mask).sum()
    from chimera2.arrayattr import AggregateSelection
    return AggregateSelection(spheres, numpy.ma.filled(mask, True))

import timeit
t0 = timeit.timeit(zone0, number=20)
print t0
t1 = timeit.timeit(zone1, number=20)
print t1
print '%fX faster' % (t0 / t1)
try:
    import numexpr
    t2 = timeit.timeit(zone2, number=20)
    print t2
    print '%fX faster' % (t0 / t2)
except ImportError:
    pass
