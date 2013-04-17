# figure out approximate overhead for Duck objects

class Duck(object):
    __slots__ = ['_aggregate', '_index', '__weakref__']

    def __init__(self, i):
        self._aggregate = self
        self._index = i

from memory import memory
m = memory()

a = [Duck(i) for i in xrange(1024*1024)]

print memory(m) / (1024 * 1024)
