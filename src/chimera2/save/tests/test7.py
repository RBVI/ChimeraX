"""
Benchmark cost of Proxy attribute access
"""
import sys
sys.path.insert(0, '..')
from chimera2 import universe
import fancy as geom
from chimera2 import shapes

COUNT = 100

def build_group():
    group = geom.Transform()
    group.reserve(geom.Sphere, COUNT)

    s = geom.Sphere(1.0, (0.0, 0.0, 0.0))
    for i in xrange(COUNT):
        group.append(s).member = None
    return group

group = build_group()
proxy_array = group.items(geom.Sphere).proxies
proxy_list = list(proxy_array)

class NormalObject(object):
    def __init__(self):
        self.member = None

normal_array = [NormalObject() for i in xrange(COUNT)]

def get_proxy_array():
    pa = proxy_array
    for obj in pa:
        obj.member

def set_proxy_array():
    na = proxy_array
    for obj in pa:
        obj.member = None

def get_proxy_list():
    pl = proxy_list
    for obj in pl:
        obj.member

def set_normal():
    pl = proxy_list
    for obj in pl:
        obj.member = None

def get_normal():
    na = normal_array
    for obj in na:
        obj.member

def set_normal():
    na = normal_array
    for obj in na:
        obj.member = None


import timeit
t0 = timeit.timeit(get_normal)
print t0
t1 = timeit.timeit(get_proxy_array)
print t1
t2 = timeit.timeit(get_proxy_list)
print t2
if t1 < t0:
    print 'get proxy array %fX faster' % (t0 / t1)
else:
    print 'get proxy array %fX slower' % (t1 / t0)
if t2 < t0:
    print 'get proxy list %fX faster' % (t0 / t2)
else:
    print 'get proxy list %fX slower' % (t2 / t0)

raise SystemExit, 0

import timeit
t0 = timeit.timeit("normal.member = None", setup="from __main__ import normal")
print t0
t1 = timeit.timeit("proxy.member = None", setup="from __main__ import proxy")
print t1
if t1 < t0:
    print 'set %fX faster' % (t0 / t1)
else:
    print 'set %fX slower' % (t1 / t0)
