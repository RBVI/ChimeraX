# see if Proxy C object creation is slower -- No! (as expected, 2.9X faster)
import sys
sys.path.insert(0, '..')
from chimera2.arrayattr import Proxy

class Aggregator:

    def __init__(self):
        self.arrays = {}

def create():
    a = Aggregator()
    return [Proxy(a, i) for i in xrange(1000000)]

import timeit
t0 = timeit.timeit(create, number=20)
print t0
