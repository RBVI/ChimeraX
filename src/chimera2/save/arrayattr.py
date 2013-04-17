"""
arrayattr: Support for using NumPy arrays for Item Collections
==============================================================
"""
from . import universe, selection
import numpy
import sys

GROWTH_FACTOR = 1.5	# should be >= 1.5 and <= 2
DEFAULT_INITIAL_CAPACITY = 2

class AttributeItem(object):
    """Specify numpy dtype for a given attribute
    
    The dtype should include the full description of the type,
    including its shape, i.e., "3d" or (numpy.float64, (3, 1)).
    """
    __slots__ = ['dtype', 'required']
    def __init__(self, dtype, required=False):
        self.dtype = dtype
        self.required = required

class Aggregator(object):
    """Collect homogeneous objects' attributes into NumPy arrays
    
    Each object that an Aggregator takes in must subsequently be referred
    to by its Proxy object.
    NumPy arrays are created on-demand for any added attribute. 
    The Python list API is implemented as appropriate."""

    def __init__(self, container, generic_item_type, attr_info, capacity):
        assert capacity >= 0
        import weakref
        if not isinstance(container, weakref.ReferenceType):
            container = weakref.ref(container)
        self.container = container
        self.attr_info = attr_info.copy()
        self.size = 0
        self.capacity = capacity
        self.proxy_type = type('Proxy' + generic_item_type.__name__, (Proxy,),
                                                            {'__slots__': []})
        generic_item_type.register(self.proxy_type)
        self.proxies = numpy.ma.empty((capacity,), numpy.object_)
        self.proxies.mask = True
        self.arrays = {}
        for name, ai in self.attr_info.iteritems():
            if ai.required:
                array = numpy.empty((capacity,), ai.dtype)
            else:
                array = numpy.ma.empty((capacity,), ai.dtype)
                array.mask = True
            self.arrays[name] = array

    def __len__(self):
        return self.size

    def __nonzero__(self):
        return self.size != 0

    def __getitem__(self, i):
        # TODO: document that it may return numpy.ma.masked
        if isinstance(i, slice):
            return self.proxies[xrange(*slice.indices(self.size))]
        if not isinstance(i, int):
            raise TypeError('indices must be integers')
        if i < 0:
            i += self.size
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')
        return self.proxies[i]

    def __setitem__(self, i, value):
        if isinstance(i, slice):
            raise RuntimeError("slices are not supported for assignment")
        if not isinstance(i, int):
            raise TypeError('indices must be integers')
        if i < 0:
            i += self.size
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')
        self._set(i, value)

    def _set(self, i, obj):
        # fill in row with object's attributes
        attrs = set(obj.keys() if isinstance(obj, dict) else dir(obj))
        for name, ai in self.attr_info.iteritems():
            attrs.discard(name)
            try:
                value = getattr(obj, name)
            except AttributeError:
                if ai.required:
                    raise ValueError("missing attribute: %s" % name)
                continue
            array = self.arrays[name]
            array[i] = value
        if not attrs:
            return
        p = self.proxies[i]
        for attr in attrs:
            if attr.startswith('__'):
                continue    # skip private attributes
            setattr(p, attr, getattr(obj, attr))

    def __delitem__(self, i):
        # mask out entries
        if isinstance(i, slice):
            self.proxies.mask[xrange(*slice.indices(self.size))] = True
            return
        if not isinstance(i, int):
            raise TypeError('indices must be integers')
        if i < 0:
            i += self.size
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')
        self.proxies.mask[i] = True
        self.proxies[i]._Proxy_index = sys.maxsize

    def __contains__(self, obj):
        return (isinstance(obj, Proxy)
                and obj._Proxy_aggregate == self
                and obj._Proxy_index != sys.maxsize)

    def __iter__(self):
        # skip masked (aka, unused) proxies
        masked = numpy.ma.masked
        for obj in self.proxies:
            if obj is masked:
                continue
            yield obj
            if obj._Proxy_index == self.size - 1:
                return

    def append(self, obj):
        """Add object to end"""
        if self.size == self.capacity:
            if self.capacity == 0:
                new_capacity = DEFAULT_INITIAL_CAPACITY
            else:
                new_capacity = round(self.capacity * GROWTH_FACTOR)
            self.reserve(new_capacity)
        d = self.proxies[self.size] = self.proxy_type(self, self.size)
        self.size += 1
        try:
            self._set(self.size - 1, obj)
        except ValueError:
            self.size -= 1
            raise
        return d

    def extend(self, objs):
        """Add objects to end"""
        for o in objs:
            self.append(o)

    def remove(self, obj):
        """remove object"""
        # obj should be a Proxy object
        if (not isinstance(obj, Proxy)
        or obj._Proxy_aggregate != self
        or obj._Proxy_index == sys.maxsize):
            raise ValueError('not in container')
        self.proxies.mask[obj._Proxy_index] = True
        obj._Proxy_index = sys.maxsize

    def reserve(self, capacity):
        """Change amount of preallocated memory.
        
        The capacity should be greater or equal to the current size.  The
        capacity may be lowered to the size to try to save memory."""
        if capacity < self.capacity:
            if capacity < self.size:
                return
            # else: truncate
        new_indices = list(xrange(self.capacity, capacity))
        self.proxies.resize((capacity,))
        self.proxies.mask[new_indices] = True
        for name, ai in self.attr_info.iteritems():
            array = self.arrays[name]
            array.resize((capacity,))
            if not ai.required:
                array.mask[new_indices] = True
        self.capacity = capacity

    def populate_proxies(self, count):
        """Assist bulk interface by creating proxy objects.
        
        Return the starting index of the first proxy."""
        assert count > 0
        cur_size = self.size
        if cur_size + count > self.capacity:
            self.reserve(self.size + count)
        for i in xrange(cur_size, cur_size + count):
            self.proxies[i] = self.proxy_type(self, i)
        self.size += count
        return cur_size

    def register(self, attr_info):
        """Register additional attributes to be aggregated.
        
        Parameters:
        -----------
        attr_info : map of {name : AttributeItem} for new named attributes

        All required attributes share the same mask.
        It is possible to change the type of an existing attribute."""
        for name, ai in attr_info.iteritems():
            if not isinstance(ai, AttributeItem):
                # for C implementation of Proxy
                ai = AttributeItem(ai)
                attr_info[name] = ai
            if name in self.arrays:
                # known attribute, changing dtype or required
                cur_ai = self.attr_info.pop(name)
                # TODO: disallow losing requiredness?
                # TODO: disallow changing dtype of required attribute?
                if cur_ai.dtype == ai.dtype and cur_ai.required == ai.required:
                    continue
                array = self.arrays[name]
                if ai.required:
                    self.arrays[name] = numpy.asarray(array, ai.dtype)
                else:
                    self.arrays[name] = numpy.ma.asarray(array, ai.dtype)
                    # TODO: confirm that mask is preserved
                continue
            # new attribute
            if ai.required:
                array = numpy.empty((self.capacity,), ai.dtype)
            else:
                array = numpy.ma.empty((self.capacity,), ai.dtype)
                array.mask = True
            self.arrays[name] = array
        self.attr_info.update(attr_info)

    def select(self):
        return AggregateSelection(self, self.proxies.mask)

class AggregateSelection(selection.Selection):
    """Manipulate selections of items in Aggregates"""

    def __init__(self, aggregate, mask):
        self._aggregate = aggregate
        self._mask = mask

    def __nonzero__(self):
        return self._nonzero

    def evaluate(self):
        return self.__iter__()

    def attr(self, name):
        # don't know if we're going to get or set the attribute,
        # so always raise AttributeError if it isn't there
        try:
            array = self._aggregate.arrays[name]
        except KeyError:
            raise AttributeError(name)
        return AggregateAttribute(self, array)

    def __len__(self):
        return len(self._mask) - sum(self._mask)

    def __nonzero__(self):
        return not all(self._mask)

    def __contains__(self, obj):
        return obj in self._aggregate

    def __iter__(self):
        masked = numpy.ma.masked
        mask = self._mask
        for i, obj in enumerate(self._aggregate.proxies):
            if mask[i] or obj is masked:
                continue
            yield obj

    def add(self, value):
        raise NotImplemented("can not add to an aggregate selection")

    def discard(self, value):
        raise NotImplemented("can not discard from an aggregate selection")

class AggregateAttribute(selection.SelectionAttribute):

    def __init__(self, selection, array):
        self._selection = selection
        self._array = array

    def get(self):
        try:
            return self._array[self._selection._mask]
        except KeyError:
            raise AttributeError

    def set(self, value):
        self._array[self._selection._mask] = value

    def array(self):
        return self._array, self._selection._mask

    def __eq__(self, value):
        return AggregateSelection(self._selection._aggregate,
                numpy.logical_or(self._selection._mask, self._array != value))

    def __ge__(self, value):
        return AggregateSelection(self._selection._aggregate,
            numpy.logical_or(self._selection._mask, self._array < value))

    def __gt__(self, value):
        return AggregateSelection(self._selection._aggregate,
            numpy.logical_or(self._selection._mask, self._array <= value))

    def __le__(self, value):
        return AggregateSelection(self._selection._aggregate,
            numpy.logical_or(self._selection._mask, self._array > value))

    def __lt__(self, value):
        return AggregateSelection(self._selection._aggregate,
            numpy.logical_or(self._selection._mask, self._array >= value))

    def __ne__(self, value):
        return AggregateSelection(self._selection._aggregate,
            numpy.logical_or(self._selection._mask, self._array == value))

try:
    # Replace Python Proxy object with C implementation
    from . import _proxy
    Proxy = _proxy.Proxy
except ImportError:
    class Proxy(object):
        """Proxy for Item stored in numpy array"""
        __slots__ = ['_Proxy_aggregate', '_Proxy_index', '__weakref__']

        def __init__(self, aggregate, index):
            object.__setattr__(self, '_Proxy_aggregate', aggregate)
            object.__setattr__(self, '_Proxy_index', index)

        def __getattr__(self, name):
            if self._Proxy_index == sys.maxsize:
                raise ValueError('proxied object has been deleted')
            # only called if no instance attribute matches
            if name == "container":
                return self._Proxy_aggregate.container
            try:
                array = self._Proxy_aggregate.arrays[name]
            except KeyError:
                raise AttributeError(name)
            #assert self._Proxy_index < self._Proxy_aggregate.size
            return array[self._Proxy_index]

        def __setattr__(self, name, value):
            if self._Proxy_index == sys.maxsize:
                raise ValueError('proxied object has been deleted')
            if name.startswith('__'):
                raise AttributeError("can not set private attributes: %s: %s" % (name, value))
            if name == "container":
                assert value == self._Proxy_aggregate.container
                return
            if name not in self._Proxy_aggregate.arrays:
                # unknown attribute, add it as an object
                info = { name: AttributeItem(numpy.object_) }
                self._Proxy_aggregate.register(info)
            array = self._Proxy_aggregate.arrays[name]
            assert self._Proxy_index < self._Proxy_aggregate.size
            array[self._Proxy_index] = value

        def __delattr__(self, name):
            if self._Proxy_index == sys.maxsize:
                raise ValueError('proxied object has been deleted')
            if name.startswith('__'):
                raise AttributeError("can not del private attributes")
            try:
                array = self._Proxy_aggregate.arrays[name]
            except KeyError:
                raise AttributeError(name)
            if isinstance(array, numpy.ma.masked_array):
                array[self._Proxy_index] = numpy.ma.masked
            else:
                raise ValueError("can not del required attributes")
universe.Item.register(Proxy)

class Group(universe.Group):

    attribute_info = {}  # { generic_item_type: AttributeInfo }

    def __init__(self, *args, **kw):
        universe.Group.__init__(self, *args, **kw)
        self.attribute_info = self.attribute_info.copy()

    def append(self, item, item_type=None):
        """Place an item in the group"""
        # no need to set 'container' attribute as in universe.Group.append
        if item_type is None:
            item_type = type(item)
        try:
            items = self.items(item_type)
        except ValueError:
            items = self.reserve(item_type, 4)
        p = items.append(item)
        return p

    # way to register additional array attributes
    def register(self, generic_item_type, attr_info, capacity=0):
        """Register attributes for a particular item type"""
        if generic_item_type in self._items:
            self._items[generic_item_type].register(attr_info)
        else:
            self._items[generic_item_type] = Aggregator(self,
                                    generic_item_type, attr_info, capacity)

    def _reserve(self, item_type, capacity):
        """Reserve space for a particular item type"""
        if item_type in self._items:
            items = self._items[item_type]
            if issubclass(items, Aggregator):
                items.reserve(capacity)
            return items
        for generic_item_type in self.attribute_info:
            if not issubclass(item_type, generic_item_type):
                continue
            items = Aggregator(self, generic_item_type,
                    self.attribute_info[generic_item_type].copy(), capacity)
            self._items[item_type] = items
            break
        else:
            items = self._items[item_type] = universe.List()
        return items

    def select(self, item_types=None):
        """Select objects of a particular item type"""
        if item_types is None:
            containers = self._items.itervalues()
        else:
            containers = [self._items[t]
                    for t in self._items if issubclass(t, item_types)]
        selections = []
        for items in containers:
            if isinstance(items, Aggregator):
                selections.append(items.select())
            else:
                selections.append(selection.SelectionSet(iter(items)))
        if len(selections) == 1:
            return selections[0]
        return universe.MetaSelection(selections)
