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
    """Return appropriate filename for history file.
    
    Parameters
    ----------
    session : :py:class:`~chimera.core.session.Session` instance
    tag : str, a "unique" tag to identify the history
    unversioned : bool, optional

        If *unversioned* is True, then the history is kept
        for all versions of the application.
    """
    if unversioned:
        cache_dir = session.app_dirs_unversioned.user_cache_dir
    else:
        cache_dir = session.app_dirs.user_cache_dir
    return os.path.join(cache_dir, tag)


class JSONHistory:
    """Use JSON to serialize and deserialize history object.

    Parameters
    ----------
    session : :py:class:`~chimera.core.session.Session` instance
    tag : str, a "unique" tag to identify the history
    unversioned : bool, optional, defaults to False
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


class LRUSetHistory(OrderedSet):
    """LRU set with fixed capacity with backing store.

    Saves and restores a set of data from a history file.
    Use the :py:meth:`add` method to put items into the set
    and to update it.
    The last member of the set is the most recent.
    ALl of the normal :py:class:`set` methods are supported as well.

    Parameters
    ----------
    capacity : int, a limit on the number of items in the history
    session : :py:class:`~chimera.core.session.Session` instance
    tag : str, a "unique" tag to identify the history
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
        self._history = JSONHistory(session, tag, unversioned)
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
        assert(data == contents)

    session = Chimera2_session  # noqa

    history = LRUSetHistory(128, session, 'test_history')
    testfile = filename(session, 'test_history')

    if os.path.exists(testfile):
        print('testfile:', testfile, 'already exists')
        raise SystemExit(1)
    try:
        # create testfile with initial contents
        print('test history file', flush=True)
        history.update([1, 2, 3])
        history.save()
        check_contents(testfile, list(history))

        print('test updated history', flush=True)
        history.add(1)
        assert(list(history) == [2, 3, 1])
        history.save()
        check_contents(testfile, list(history))

    finally:
        if os.path.exists(testfile):
            os.unlink(testfile)
