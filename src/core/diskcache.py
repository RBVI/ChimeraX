# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
diskcache: application cache support
====================================

This module provides support for caching information across
application invocations.  In particular, it is useful for
command and file history.
"""
from .orderedset import OrderedSet


def filename(session, tag, unversioned=False):
    """Return appropriate filename for cache
    
    Parameters
    ----------
    session : Chimera2 session
    tag : str, a "unique" tag to identify the cache
    unversioned : bool, optional

    If unversioned is True, then cache is for all versions of Chimera2
    """
    if unversioned:
        cache_dir = session.app_dirs_unversioned.user_cache_dir
    else:
        cache_dir = session.app_dirs.user_cache_dir
    return os.path.join(cache_dir, tag)


class JSONDiskCache:
    """Use json to save and restore disk cache

    Parameters
    ----------
    session : Chimera2 session
    tag : str, a "unique" tag to identify the cache
    unversioned : bool, optional, defaults to False
    """

    def __init__(self, session, tag, unversioned=False):
        self.filename = filename(session, tag, unversioned)

    def load(self):
        """Return deserialized object from cache file""" 
        import json
        import os
        if not os.path.exists(self.filename):
            return None
        with open(self.filename) as f:
            return json.load(f)

    def save(self, obj):
        """Serialize object into cache file
        
        Parameters
        ----------
        obj: object
        """
        import json
        from .safesave import SaveTextFile
        with SaveTextFile(self.filename) as f:
            json.dump(obj, f, ensure_ascii=False)


class LRUHistory(OrderedSet):
    """LRU set with fixed capacity

    Parameters
    ----------
    capacity : int, a limit on the number of items in the history
    session : Chimera2 session
    tag : str, a "unique" tag to identify the cache
    unversioned : bool, optional, defaults to False
    auto_save : bool, optional, defaults to False

    If unversioned is true, then cache is for all versions of Chimera2.
    If auto_save is true, then the history is flushed to disk everyime
    it is updated.

    Use the add method to put items into the history and to update it.
    """

    def __init__(self, capacity, session, tag, unversioned=False,
            auto_save=False):
        self._capacity = capacity
        self._auto_save = auto_save
        self._disk_cache = JSONDiskCache(session, tag, unversioned)
        obj = self._disk_cache.load()
        if obj is None:
            obj = []
        if not isinstance(obj, list):
            session.logger.warning("Corrupt %s history: cleared" % tag)
            obj = []
        if len(obj) > capacity:
            del obj[capacity:]
        OrderedSet.__init__(self, obj)

    def save(self):
        obj = list(self)
        self._disk_cache.save(obj)

    def add(self, item):
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

    history = LRUHistory(128, session, 'test_history')
    testfile = history._disk_cache.filename

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
