# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
history: application history support
====================================

This module provides support for caching information across
application invocations.  In particular, it is useful for
command and file history.

History files are kept in an application user's cache directory,
so there is the assumption that they can be removed and the
application will still work.
"""
from .orderedset import OrderedSet


def filename(session, tag, unversioned=True):
    """Return appropriate filename for cache file.

    Parameters
    ----------
    session : :py:class:`~chimera.core.session.Session` instance
    tag : str, a unique tag to identify the history
    unversioned : bool, optional

        If *unversioned* is True, then the history is kept
        for all versions of the application.
    """
    if unversioned:
        cache_dir = session.app_dirs_unversioned.user_cache_dir
    else:
        cache_dir = session.app_dirs.user_cache_dir
    import os
    return os.path.join(cache_dir, tag)


class ObjectCache:
    """Maintain an object in application's cache on disk.

    Parameters
    ----------
    session : :py:class:`~chimera.core.session.Session` instance
    tag : str, a unique tag to identify the cache
    unversioned : bool, optional, defaults to False

    Notes
    -----
    Uses JSON to serialize and deserialize history object.
    """

    def __init__(self, session, tag, unversioned=True):
        self.filename = filename(session, tag, unversioned)

    def load(self):
        """Return deserialized object from history file."""
        import json
        import os
        if not os.path.exists(self.filename):
            return None
        with open(self.filename) as f:
            return json.load(f)

    def save(self, obj):
        """Serialize object into history file.

        Parameters
        ----------
        obj : object
            The object to save.
        """
        import json
        from .safesave import SaveTextFile
        with SaveTextFile(self.filename) as f:
            json.dump(obj, f, ensure_ascii=False)


class FIFOHistory:
    """A fixed capacity FIFO queue with backing store.

    Parameters
    ----------
    capacity : int, a limit on the number of items in the history
    session : :py:class:`~chimera.core.session.Session` instance
    tag : str, a unique tag to identify the history
    unversioned : bool, optional, defaults to True
    auto_save : bool, optional, defaults to True

        If *unversioned* is true, then the history is
        for all versions of application.
        If *auto_save* is true, then the history is flushed to disk everyime
        it is updated.

    Notes
    -----
    Iterating returns oldest first.
    """
    # FIFO from http://code.activestate.com/recipes/68436/

    def __init__(self, capacity, session, tag, unversioned=True,
                 auto_save=True):
        self._capacity = capacity
        self._auto_save = auto_save
        self._history = ObjectCache(session, tag, unversioned)
        obj = self._history.load()
        if obj is None:
            obj = []
        if not isinstance(obj, list):
            session.logger.warning("Corrupt %s history: cleared" % tag)
            obj = []
        self._front, self._back = obj, []
        while len(self._front) + len(self._back) > self._capacity:
            self.dequeue()

    def enqueue(self, value):
        self._back.append(value)
        while len(self._front) + len(self._back) > self._capacity:
            self.dequeue(_skip_save=True)
        if self._auto_save:
            self.save()

    def extend(self, iterable):
        self._back.extend(iterable)
        while len(self._front) + len(self._back) > self._capacity:
            self.dequeue(_skip_save=True)
        if self._auto_save:
            self.save()

    def dequeue(self, _skip_save=False):
        front = self._front
        if not front:
            self._front, self._back = self.back, front
            front = self._front
            front.reverse()
        value = front.pop()
        if not _skip_save and self._auto_save:
            self.save()
        return value

    def clear(self):
        self._front.clear()
        self._back.clear()

    def save(self):
        """Save fifo to history file."""
        self._front = list(reversed(self._back)) + self._front
        self._back = []
        self._history.save(self._front)

    def __len__(self):
        return len(self._front) + len(self._back)

    def __iter__(self):
        # return oldest first
        import itertools
        return itertools.chain(reversed(self._front), self._back)

    def __getitem__(self, index):
        # return oldest first
        front_len = len(self._front)
        back_len = len(self._back)
        total_len = front_len + back_len
        if isinstance(index, int):
            if index < -total_len or index >= total_len:
                raise IndexError("index out of range")
            if index < 0:
                index += total_len
            if index < front_len:
                return self._front[front_len - 1 - index]
            index -= front_len
            return self._back[index]
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(total_len))]
        raise TypeError("Expected integer or slice")


class LRUSetHistory(OrderedSet):
    """A fixed capacity LRU set with backing store.

    Saves and restores a set of data from a history file.
    Use the :py:meth:`add` method to put items into the set
    and to update it.
    The last member of the set is the most recent.
    ALl of the normal :py:class:`set` methods are supported as well.

    Parameters
    ----------
    capacity : int, a limit on the number of items in the history
    session : :py:class:`~chimera.core.session.Session` instance
    tag : str, a unique tag to identify the history
    unversioned : bool, optional, defaults to True
    auto_save : bool, optional, defaults to True

        If *unversioned* is true, then the history is
        for all versions of application.
        If *auto_save* is true, then the history is flushed to disk everyime
        it is updated.
    """

    def __init__(self, capacity, session, tag, unversioned=True,
                 auto_save=True):
        self._capacity = capacity
        self._auto_save = auto_save
        self._history = ObjectCache(session, tag, unversioned)
        obj = self._history.load()
        if obj is None:
            obj = []
        if not isinstance(obj, list):
            session.logger.warning("Corrupt %s history: cleared" % tag)
            obj = []
        if len(obj) > capacity:
            del obj[capacity:]
        OrderedSet.__init__(self, obj)

    def save(self):
        """Save set to history file."""
        obj = list(self)
        self._history.save(obj)

    def add(self, item):
        """Add item to set and make it the most recent.

        Parameters
        ----------
        item : simple type suitable for Python's :py:mod:`JSON` module.
        """
        if item in self:
            self.discard(item)
        OrderedSet.add(self, item)
        if len(self) > self._capacity:
            self.pop()
        if self._auto_save:
            self.save()

if __name__ == '__main__':
    # simple test
    import json
    import os

    def check_contents(filename, contents):
        with open(filename) as f:
            data = json.load(f)
        if data != contents:
            print('Filename:', filename)
            print('    data:', data)
            print('contents:', contents)
        assert(data == contents)

    session = Chimera2_session  # noqa

    print('test LRU history file', flush=True)
    history = LRUSetHistory(128, session, 'test_history')
    testfile = filename(session, 'test_history')

    if os.path.exists(testfile):
        print('testfile:', testfile, 'already exists')
        raise SystemExit(1)
    try:
        # create testfile with initial contents
        history.update([1, 2, 3])
        check_contents(testfile, list(history))

        print('test updated history', flush=True)
        history.add(1)
        assert(list(history) == [2, 3, 1])
        check_contents(testfile, list(history))

    finally:
        if os.path.exists(testfile):
            os.unlink(testfile)

    print('test FIFO history file', flush=True)
    history = FIFOHistory(3, session, 'test_history')
    testfile = filename(session, 'test_history')

    if os.path.exists(testfile):
        print('testfile:', testfile, 'already exists')
        raise SystemExit(1)
    try:
        # create testfile with initial contents
        history.enqueue(1)
        history.enqueue(2)
        history.enqueue(3)
        check_contents(testfile, [3, 2, 1])

        print('iter/getitem test')
        for i, v in enumerate(history):
            assert(history[i] == v)

        print('test updated history', flush=True)
        history.enqueue(4)
        check_contents(testfile, [4, 3, 2])
        assert(history[-2:] == [3, 4])

    finally:
        if os.path.exists(testfile):
            os.unlink(testfile)
