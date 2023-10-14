# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
orderedset: An ordered set
==========================
Inspiration and code expanded from
Raymond Hettinger's `indexable weak ordered set
<http://stackoverflow.com/questions/7828444/indexable-weak-ordered-set-in-python>`_
and Stephan Schroevers' `does python have an ordered set
<http://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set>`_
`stackoverflow.com <http://stackoverflow.com/>`_ postings.
"""

import collections, collections.abc, weakref

class OrderedSet(collections.abc.MutableSet):
    """Ordered set.
    
    Supports all of the operations that can happen on a :py:class:`set`."""

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
        if isinstance(other, collections.abc.Set):
            return collections.abc.MutableSet.__eq__(self, other)
        return False

    def __ne__(self, other):
        if isinstance(other, collections.abc.Set):
            return collections.abc.MutableSet.__ne__(self, other)
        return True

    __hash__ = None

    difference = collections.abc.MutableSet.__sub__
    difference_update = collections.abc.MutableSet.__isub__
    intersection = collections.abc.MutableSet.__and__
    intersection_update = collections.abc.MutableSet.__iand__
    issubset = collections.abc.MutableSet.__le__
    issuperset = collections.abc.MutableSet.__ge__
    symmetric_difference = collections.abc.MutableSet.__xor__
    symmetric_difference_update = collections.abc.MutableSet.__ixor__
    union = collections.abc.MutableSet.__or__
    update = collections.abc.MutableSet.__ior__

    __rand__ = collections.abc.MutableSet.__and__
    __ror__ = collections.abc.MutableSet.__or__

    def __rsub__(self, other):
        # needed for generic xor
        return type(other)(iter(self)) - other

    def __rxor__(self, other):
        collections.abc.MutableSet.__xor__(self, other)

#   isdisjoint, issubset, issuperset, pop, remove,

class OrderedWeakrefSet(weakref.WeakSet):
    """Ordered set of weak references"""
    def __init__(self, values=()):
        super(OrderedWeakrefSet, self).__init__()
        self.data = OrderedSet()
        for elem in values:
            self.add(elem)

