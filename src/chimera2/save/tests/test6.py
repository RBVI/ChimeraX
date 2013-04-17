"""
Benchmark cost of Proxy attribute access
"""
import sys
sys.path.insert(0, '..')
from chimera2 import universe
import fancy as geom
from chimera2 import shapes

def build_group():
    group = geom.Transform()

    s = geom.Sphere(1.0, (0.0, 0.0, 0.0))
    s = group.append(s)
    return group, s

group, proxy = build_group()
proxy.member = None

class NormalObject(object):
    pass

normal = NormalObject()
normal.member = None

import timeit
t0 = timeit.timeit("normal.member", setup="from __main__ import normal")
print t0
t1 = timeit.timeit("proxy.member", setup="from __main__ import proxy")
print t1
if t1 < t0:
    print 'get %fX faster' % (t0 / t1)
else:
    print 'get %fX slower' % (t1 / t0)

import timeit
t0 = timeit.timeit("normal.member = None", setup="from __main__ import normal")
print t0
t1 = timeit.timeit("proxy.member = None", setup="from __main__ import proxy")
print t1
if t1 < t0:
    print 'set %fX faster' % (t0 / t1)
else:
    print 'set %fX slower' % (t1 / t0)
