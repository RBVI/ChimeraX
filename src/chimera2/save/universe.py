"""
universe: Everything that is important
======================================

The universe singleton object (currently implemented as a Python module)
holds (nested) Group's of generic items.

A Group can be used outside of the universe, e.g., for templates.

Groups keep track of Items by their type using a dictionary,
keying on the type and the value is a collection of items.
The collection type should provide the python list API
with the additional requirement
that the collection's :meth:append method returns either the appended object
or a proxy for that object.

Units: Angstroms, radians, seconds.
"""

__all__ = [
    'add', 'remove', 'select_groups', 'select_items', 'Group', 'Item', 'List'
]

from .selection import SelectionOrderedSet, MetaSelection, SelectionSet

_groups = set()
_all_groups = set()

# need something that acts like a weak reference but always returns None
_WEAKREF_NULL = lambda: None

# need something that acts like a weak reference but always returns the universe
import sys
_WEAKREF_UNIVERSE = lambda m=sys.modules[__name__]: m

class List(list):
    """a simple homogeneous collection of :class:`Item`s"""

    def append(self, obj):
        """Place object item in the collection
        
        Parameters
        ----------
        obj : Item-like
            The object to save in container.

        Returns
        -------
        Returns `obj` or a proxy for it.
        """
        list.append(self, obj)
        return obj

def add(group):
    """Place a group into the universe"""
    group.container = _WEAKREF_UNIVERSE
    _groups.add(group)
    _all_groups.add(group)

def remove(group):
    """Take a group out of the universe"""
    group.container = _WEAKREF_NULL
    _all_groups.remove(group)
    _groups.remove(group)

def select_groups(group_types=None):
    """Select a subset of groups that match the given criteria
    
    Returns
    -------
    a :class:`selection.Selection`
    """
    groups = _all_groups
    if group_types:
        groups = [g for g in groups if isinstance(g, group_types)]
    return SelectionOrderedSet(groups)

def select_items(item_types=None, group_types=None):
    """Select a subset of items in groups
    
    Parameters
    ----------
    item_types : class-or-type-or-tuple, optional
       Restrict selected items to the given type(s).
    group_types : class-or-type-or-tuple, optional
       Restrict selected groups to the given type(s).

    Returns
    -------
    A selection.
    """
    groups = _all_groups
    if group_types:
        groups = [g for g in groups if isinstance(g, group_types)]
    selections = [g.select(item_types) for g in groups]
    selections = [s for s in selections if s]
    if len(selections) == 1:
        return selections[0]
    return MetaSelection(selections)

from abc import ABCMeta, abstractproperty

class Item(object):
    """Provide abstract base class for common items"""
    __metaclass__ = ABCMeta
    @abstractproperty
    def container(self):
        """Return collection that item is in"""
        raise NotImplemented

# have the universe act like an Item
container = _WEAKREF_NULL

class Group(Item):
    """A container that holds generic items. 

    Attributes
    ----------
    container : weak reference
    _items : map of item types to item collections
    _item_type_map : combines related item subtypes
    """
    container = _WEAKREF_NULL

    def __init__(self):
        self._items = {}
        self._item_type_map = {}

    def items(self, item_type):
        """return an internal homogeneous collection of items
        
        Parameters
        ----------
        item_type : the type of the requested items
        """
        if item_type not in self._item_type_map:
            self._add_type_synonyms(item_type)
            if item_type not in self._item_type_map:
                raise ValueError("group has no items of type %s" % item_type.__name__)
        return self._items[self._item_type_map[item_type]]

    def _add_type_synonyms(self, item_type):
        for t in item_type.mro():
            if t in self._item_type_map:
                self._item_type_map[item_type] = self._item_type_map[t]
                break

    def append(self, item, item_type=None):
        """Place an item in the group
        
        Parameters
        ----------
        item : an item to place in group
        item_type : subtype of item, optional optimization
        """
        if hasattr(item, 'container') and item.container is not None:
            raise ValueError('already in another group')
        if item_type is None:
            item_type = type(item)
        try:
            items = self.items(item_type)
        except ValueError:
            items = self.reserve(item_type, 4)
        p = items.append(item)
        from weakref import ref
        p.container = ref(self)
        return p

    def remove(self, item):
        """Take an existing item out of the group"""
        try:
            items = self.items(type(item))
            items.remove(item)
        except ValueError:
            raise ValueError("item is not in group")

    def _reserve(self, item_type, capacity):
        # override in subclass
        if item_type in self._items:
            return self._items[item_type]
        items = self._items[item_type] = List()
        return items

    def reserve(self, item_type, capacity):
        """Provide hint about how many of a given item are expected"""
        types = []
        for t in item_type.mro():
            if t is not Item and issubclass(t, Item):
                if t in self._item_type_map:
                    return self._reserve(self._item_type_map[t], capacity)
                types.append(t)
        for t in types:
            self._item_type_map[t] = item_type
        return self._reserve(item_type, capacity)

    def count(self, item_type):
        """Return how many of a particular type of item are in the group"""
        try:
            items = self.items(item_type)
        except ValueError:
            raise
        return len(items)

    def select(self, item_types=None):
        """Return selected items"""
        items = []
        if item_types is None:
            for t in self._items:
                items.extend(self._items[t])
        else:
            for t in self._items:
                if issubclass(t, item_types):
                    items.extend(self._items[t])
        return SelectionSet(items)
