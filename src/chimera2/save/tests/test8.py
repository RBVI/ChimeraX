import sys
sys.path.insert(0, '..')
from chimera2.arrayattr import AttributeItem, Group
from chimera2 import universe
from abc import ABCMeta

class Atom(object):
        __metaclass__ = ABCMeta
universe.Item.register(Atom)

Atom_info = {
        'attr': AttributeItem('i', False)
}

class Molecule(Group):

        attribute_info = {
                Atom: Atom_info
        }

m = Molecule()
m.reserve(Atom, 10000)
for i in range(10000):
        a = Atom()
        a.attr = i
        a = m.append(a)

def simple_timed_test():
	from time import clock
	t0 = clock()
	for a in m.select(Atom):
		x = a.attr + 5
	print clock() - t0

class Test(object):
	pass
t = Test()
t.attr = 1

"""
import timeit
for a in m.select(Atom):
	break
slow = timeit.timeit('x = a.attr + 5', 'from %s import a' % __name__, number=10000)

fast = timeit.timeit('x = t.attr + 5', 'from %s import t' % __name__, number=10000)

print 'slow time', slow
print 'fast time', fast
print 'array attributes are', slow / fast, 'slower'

# results before profiling:
#    attr not required:
#        w/o C proxy object:
#            145X slower
#        w/o C proxy object and w/o line 347 assert:
#            140X slower
#        w/C proxy object:
#            115X slower
#        w/o C proxy object and elide masked array access:
#            15X slower
#    attr required:
#        w/o C proxy object:
#            92X slower
#        w/o C proxy object and w/o line 347 assert:
#            85X slower
#        w/C proxy object:
#            55X slower
"""

"""
import trace
tracer = trace.Trace(
	count=1, trace=1, countfuncs=0, countcallers=0,
	ignoremods=['trace'], ignoredirs=[],
	infile=None, outfile=None, timing=False
)
tracer.runctx('x = a.attr + 5', globals())
r = tracer.results()
r.write_results(summary=True, coverdir='/tmp')

#Executing test8.py...
# --- modulename: test8, funcname: <module>
# 0.00 <string>(1):   --- modulename: arrayattr, funcname: __getattr__
# 0.00 arrayattr.py(338):             if self._Proxy_index == sys.maxsize:
# 0.00 arrayattr.py(341):             if name == "container":
# 0.00 arrayattr.py(343):             try:
# 0.00 arrayattr.py(344):                 array = self._Proxy_aggregate.arrays[name]
# 0.00 arrayattr.py(347):             assert self._Proxy_index < self._Proxy_aggregate.size
# 0.00 arrayattr.py(348):             return array[self._Proxy_index]
#  --- modulename: core, funcname: __getitem__
#  0.00 core.py(2940):         _data = ndarray.view(self, ndarray)
#  0.00 core.py(2941):         dout = ndarray.__getitem__(_data, indx)
#  0.00 core.py(2946):         _mask = self._mask
#  0.00 core.py(2947):         if not getattr(dout, 'ndim', False):
#  0.00 core.py(2949):             if isinstance(dout, np.void):
#  0.00 core.py(2958):             elif _mask is not nomask and _mask[indx]:
#  0.00 core.py(2975):         return dout
#  lines   cov%   module   (path)
#    323     1%   chimera2.arrayattr   (../chimera2/arrayattr.py)
#   2427     0%   numpy.ma.core   (/var/tmp/chimera-build/Linux64-X11/build/lib/python2.7/site-packages/numpy/ma/core.py)
#Executed test8.py
"""

"""
import hotshot, hotshot.stats

prof = hotshot.Profile('test8.prof')
benchtime = prof.runctx('for i in xrange(100000): x = a.attr + 5', globals(), globals())
prof.close()
stats = hotshot.stats.load('test8.prof')
stats.strip_dirs()
stats.print_stats()

#Executing test8.py...
#         200001 function calls in 1.301 seconds
#
#    Random listing order was used
#
#   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#   100000    0.360    0.000    0.360    0.000 core.py:2930(__getitem__)
#   100000    0.179    0.000    0.539    0.000 arrayattr.py:337(__getattr__)
#        0    0.000             0.000          profile:0(profiler)
#        1    0.762    0.762    1.301    1.301 <string>:1(<module>)
#
#
#Executed test8.py
"""

"""
import cProfile, pstats
cProfile.runctx('for i in xrange(100000): x = a.attr + 5', globals(), globals(), 'test8.prof')
p = pstats.Stats('test8.prof')
p.sort_stats('cumulative')
p.print_stats()

#Executing test8.py...
#Tue Nov 13 13:37:51 2012    test8.prof
#
#         400002 function calls in 1.246 seconds
#
#   Ordered by: cumulative time
#
#   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#        1    0.710    0.710    1.246    1.246 <string>:1(<module>)
#   100000    0.164    0.000    0.536    0.000 ../chimera2/arrayattr.py:337(__getattr__)
#   100000    0.291    0.000    0.372    0.000 /var/tmp/chimera-build/Linux64-X11/build/lib/python2.7/site-packages/numpy/ma/core.py:2930(__getitem__)
#   100000    0.047    0.000    0.047    0.000 {isinstance}
#   100000    0.034    0.000    0.034    0.000 {getattr}
#        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
#
#
#Executed test8.py

# With C Proxy object:
#Executing test8.py...
#Tue Nov 13 15:29:38 2012    test8.prof
#
#         100002 function calls in 0.754 seconds
#
#   Ordered by: cumulative time
#
#   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#        1    0.634    0.634    0.754    0.754 <string>:1(<module>)
#   100000    0.120    0.000    0.120    0.000 ../chimera2/arrayattr.py:337(__getattr__)
#        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
#
#
#Executed test8.py
"""
