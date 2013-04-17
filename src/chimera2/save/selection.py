"""
selection: selection base classes
=================================

  A selection acts like a lazy evaluation set, i.e., a set with additional
  methods for lazy evaluation.

  Selection methods from set:

  __and__, __contains__, __cmp__, __delattr_, __eq__, __format__,
  __ge__, __gt__, __hash__, __iand__, __init__, __ior__, __isub__, __iter__,
  __ixor__, __le__, __len__, __lt__, __ne__, __new__, __or__, __rand__,
  __reduce__, __reduce_ex__, __repr__, __ror__, __rsub__, __rxor__,
  __setattr__, __sizeof__, __str__, __sub__, __subclasshook__, __xor__,
  add, clear, copy, difference, difference_update, discard, intersection,
  intersection_update, isdisjoint, issubset, issuperset, pop, remove,
  symmetric_difference, symmetric_difference_update, union, update

  Additional functions:

      attr(name)  -- return an Selection Attribute object that represents
                      a collection of the values of the given attribute
                      for the objects in the selection
      evaluate()  -- turn the Selection into a set

  Evaluation is always done if the set functions __iter__ or pop are used.
  And might be needed in other cases (e.g., __contains__).

  Selections are specialized for different kinds of Groups.
"""

__all__ = ['Selection', 'SelectionAttribute',
    'SelectionSet', 'SelectionOrderedSet', 'MetaSelection']

from abc import ABCMeta, abstractmethod
import collections, itertools

class Selection(collections.MutableSet):
    """Provide abstract base class for selections"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self):
        raise NotImplemented

    @abstractmethod
    def attr(self, name):
        raise NotImplemented

class SelectionAttribute(object):
    """Provide abstract base class for attributes of selected objects"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def get(self):
        raise NotImplemented

    @abstractmethod
    def set(self, value):
        raise NotImplemented

    @abstractmethod
    def __eq__(self, value):
        raise NotImplemented

    @abstractmethod
    def __ge__(self, value):
        raise NotImplemented

    @abstractmethod
    def __gt__(self, value):
        raise NotImplemented

    @abstractmethod
    def __le__(self, value):
        raise NotImplemented

    @abstractmethod
    def __lt__(self, value):
        raise NotImplemented

    @abstractmethod
    def __ne__(self, value):
        raise NotImplemented

class SelectionSet(set, Selection):
    """A selection specialization that is a set"""

    def __init__(self, objs):
        set.__init__(self, objs)

    def evaluate(self):
        return self

    def attr(self, name):
        return SelectionSetAttribute(self, name)

class SelectionSetAttribute(SelectionAttribute):
    """Specialization of SelectionAttribute for SelectionSet"""

    def __init__(self, objs, name):
        self._list = objs
        self._type = type(self._list)
        self.name = name

    def get(self):
        #return [getattr(o, self.name) for o in self._list]
        import operator
        f = operator.attrgetter(self.name)
        return self._type(f(o) for o in self._list)

    def set(self, value):
        name = self.name
        for o in self._list:
            setattr(o, name, value)

    # TODO: numerical operators?
    #def __abs__(self):
    #    return self._type(abs(getattr(o, self.name)) for o in self._list)

    def __eq__(self, value):
        return self._type(o for o in self._list
                            if getattr(o, self.name) == value)

    def __ge__(self, value):
        return self._type(o for o in self._list
                            if getattr(o, self.name) >= value)

    def __gt__(self, value):
        return self._type(o for o in self._list
                            if getattr(o, self.name) > value)

    def __le__(self, value):
        return self._type(o for o in self._list
                            if getattr(o, self.name) <= value)

    def __lt__(self, value):
        return self._type(o for o in self._list
                            if getattr(o, self.name) < value)

    def __ne__(self, value):
        return self._type(o for o in self._list
                            if getattr(o, self.name) != value)

from .orderedset import OrderedSet
class SelectionOrderedSet(OrderedSet, Selection):
    """A selection specialization that is an ordered set"""

    def __init__(self, objs):
        set.__init__(self, objs)

    def evaluate(self):
        return self

    def attr(self, name):
        return SelectionSetAttribute(self, name)

class MetaSelection(Selection):
    """A selection composed of selections"""

    def __init__(self, selections):
        assert all(issubclass(o, Selection) for o in selections)
        self._selections = selections

    def evaluate(self):
        return itertools.chain(s.evaluate() for s in self._selections)

    def attr(self, name):
        return MetaSelectionAttribute(self, name)

    def __len__(self):
        return sum(len(s) for s in self._selections)

    def __nonzero__(self):
        return any(self._selections)

    def __contains__(self, selection):
        return any(selection in s for s in self._selections)

    def __iter__(self):
        return itertools.chain(self._selections)

    def add(self, value):
        raise NotImplemented("can not add to a MetaSelection")

    def discard(self, value):
        raise NotImplemented("can not discard from a MetaSelection")

class MetaSelectionAttribute(SelectionAttribute):
    """Specialization of SelectionAttribute for SelectionSet"""

    def __init__(self, selections, name):
        self._selattrs = [s.attr(name) for s in selections]

    def get(self):
        return itertools.chain(sa.get() for sa in self._selattrs)

    def set(self, value):
        for sa in self._selattrs:
            sa.set(value)

    # TODO: numerical operators?
    #def __abs__(self):
    #    return itertools.chain(sa.__abs__() for sa in self._selattrs)

    def __eq__(self, value):
        return MetaSelection(sa == value for sa in self._selattrs)

    def __ge__(self, value):
        return MetaSelection(sa >= value for sa in self._selattrs)

    def __gt__(self, value):
        return MetaSelection(sa > value for sa in self._selattrs)

    def __le__(self, value):
        return MetaSelection(sa <= value for sa in self._selattrs)

    def __lt__(self, value):
        return MetaSelection(sa < value for sa in self._selattrs)

    def __ne__(self, value):
        return MetaSelection(sa != value for sa in self._selattrs)
