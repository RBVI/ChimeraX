# Inspiration and code expanded from:
# Raymond Hettinger:
# http://stackoverflow.com/questions/7828444/indexable-weak-ordered-set-in-python
# and Stephan Schroevers:
# http://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set

import collections, weakref

class OrderedSet(collections.MutableSet):

    # TODO: replace generic MutableSet/Set algorithms with ones optimized
    # for OrderedDicts

    def __init__(self, values=()):
        self._od = collections.OrderedDict().fromkeys(values)

    def __len__(self):
        return len(self._od)

    def __iter__(self):
        return iter(self._od)

    def __contains__(self, value):
        return value in self._od

    def add(self, value):
        self._od[value] = None

    def discard(self, value):
        self._od.pop(value, None)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, list(self._od))

    __str__ = __repr__

    def __cmp__(self, other):
        raise TypeError("cannot compare sets using cmp()")

    def clear(self):
        self._od.clear()

    def copy(self):
        return OrderedSet(self)

    def __eq__(self, other):
        if isinstance(other, collections.Set):
            return collections.MutableSet.__eq__(self, other)
        return False

    def __ne__(self, other):
        if isinstance(other, collections.Set):
            return collections.MutableSet.__ne__(self, other)
        return True

    __hash__ = None

    difference = collections.MutableSet.__sub__
    difference_update = collections.MutableSet.__isub__
    intersection = collections.MutableSet.__and__
    intersection_update = collections.MutableSet.__iand__
    issubset = collections.MutableSet.__le__
    issuperset = collections.MutableSet.__ge__
    symmetric_difference = collections.MutableSet.__xor__
    symmetric_difference_update = collections.MutableSet.__ixor__
    union = collections.MutableSet.__or__
    update = collections.MutableSet.__ior__

    __rand__ = collections.MutableSet.__and__
    __ror__ = collections.MutableSet.__or__

    def __rsub__(self, other):
        # needed for generic xor
        return type(other)(iter(self)) - other

    def __rxor__(self, other):
        collections.MutableSet.__xor__(self, other)

#   isdisjoint, issubset, issuperset, pop, remove,

class OrderedWeakrefSet(weakref.WeakSet):
    def __init__(self, values=()):
        super(OrderedWeakrefSet, self).__init__()
        self.data = OrderedSet()
        for elem in values:
            self.add(elem)

