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
history: Application history support
====================================

This module provides support for caching information across
application invocations.  In particular, it is useful for
command and file history.

History files are kept in an application user's cache directory,
so there is the assumption that they can be removed and the
application will still work.
"""
from .orderedset import OrderedSet


def _history_filename(tag, unversioned=True):
    """Return appropriate filename for history file.

    Parameters
    ----------
    tag : str, a unique tag to identify the history
    unversioned : bool, optional

        If *unversioned* is True, then the history is kept
        for all versions of the application.
    """
    from chimerax import app_dirs, app_dirs_unversioned
    if unversioned:
        cache_dir = app_dirs_unversioned.user_data_dir
    else:
        cache_dir = app_dirs.user_data_dir
    import os.path
    return os.path.join(cache_dir, tag)


def _old_history_filename(tag, unversioned=True):
    """Return old filename for history file.
    A different directory was used in older ChimeraX versions
    and this routine provides the old file location for backwards
    compatibility.
    """
    from chimerax import app_dirs, app_dirs_unversioned
    if unversioned:
        cache_dir = app_dirs_unversioned.user_cache_dir
    else:
        cache_dir = app_dirs.user_cache_dir
    import os.path
    return os.path.join(cache_dir, tag)


class ObjectHistory:
    """Maintain an object in application's history on disk.

    Parameters
    ----------
    session : :py:class:`~chimerax.core.session.Session` instance
    tag : str, a unique tag to identify the cache
    unversioned : bool, optional, defaults to False

    Notes
    -----
    Uses JSON to serialize and deserialize history object.
    """

    def __init__(self, tag, unversioned=True):
        self._filename = _history_filename(tag, unversioned)
        # Older ChimeraX saved history in a location that was deleted on Mac on OS upgrades.
        # Look in old location if file does not exist in new location for backwards compatibility.
        self._old_filename = _old_history_filename(tag, unversioned)

    def load(self):
        """Return deserialized object from history file."""
        import json
        from os.path import exists
        if exists(self._filename):
            path = self._filename
        elif exists(self._old_filename):
            # Backwards compatibility, history files used to be saved in a different location.
            path = self._old_filename
        else:
            return None
        with open(path, encoding='utf-8') as f:
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
        with SaveTextFile(self._filename) as f:
            json.dump(obj, f, ensure_ascii=False)

    def backup(self):
        import os.path
        if os.path.exists(self._filename):
            from time import strftime
            suffix = '.backup.%s' % strftime("%Y%m%d-%H%M%S")
            backup_path = self._filename + suffix
            from shutil import copyfile
            copyfile(self._filename, backup_path)
        else:
            backup_path = ''
        return backup_path


class FIFOHistory:
    """A fixed capacity FIFO queue with backing store.

    Parameters
    ----------
    capacity : int, a limit on the number of items in the history
    session : :py:class:`~chimerax.core.session.Session` instance
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
        self._history = ObjectHistory(tag, unversioned)
        from json.decoder import JSONDecodeError
        try:
            obj = self._history.load()
        except JSONDecodeError:
            obj = False
        if obj is None:
            obj = []
        if not isinstance(obj, list):
            session.logger.warning("Corrupt %s history: cleared" % tag)
            obj = []
        self._queue = obj
        while len(self._queue) > self._capacity:
            self.dequeue()

    def enqueue(self, value):
        """Add newest item"""
        self._queue.append(value)
        while len(self._queue) > self._capacity:
            self.dequeue(_skip_save=True)
        if self._auto_save:
            self.save()

    def extend(self, iterable):
        """Add newest items"""
        self._queue.extend(iterable)
        while len(self._queue) > self._capacity:
            self.dequeue(_skip_save=True)
        if self._auto_save:
            self.save()

    def dequeue(self, _skip_save=False):
        """Remove and return oldest item"""
        value = self._queue.pop(0)
        if not _skip_save and self._auto_save:
            self.save()
        return value

    def clear(self):
        """Remove all items"""
        self._queue.clear()
        if self._auto_save:
            self.save()

    def replace(self, iterable):
        """Replace current items"""
        self._queue.clear()
        self.extend(iterable)

    def save(self):
        """Save to history file."""
        self._history.save(self._queue)

    def __len__(self):
        return len(self._queue)

    def __iter__(self):
        # return oldest first
        return iter(self._queue)

    def __getitem__(self, index):
        # return oldest first
        if isinstance(index, int):
            return self._queue[index]
        if isinstance(index, slice):
            return [self._queue[i] for i in range(*index.indices(len(self._queue)))]
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
    session : :py:class:`~chimerax.core.session.Session` instance
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
        self._history = ObjectHistory(tag, unversioned)
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

    print('test LRU history file', flush=True)
    history = LRUSetHistory(128, session, 'test_history')
    testfile = _history_filename('test_history')

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
    testfile = _history_filename('test_history')

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
