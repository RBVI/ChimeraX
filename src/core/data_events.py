# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
data_events: Support for bulk tracking of data changes
=======================================================

This module provides a singleton of class :py:class:`Tracker`,
:py:data:`tracker`,
that is used to batch notifications of changes of various data
structures.
Notifications are ordered, so the nofications for particular
data types will always be in the same order.

Data types that should be tracked are registered with
:py:func:`Tracker.register_data_type` and the order given.

::

    def computation():
        from chimera.core.data_events import tracker
        with tracker:
            # manipulate data
        # changes propagated

"""
from . import triggerset


class Changes:

    __slots__ = ['created', 'modified', 'deleted', 'reasons', '_dirty']

    def __init__(self):
        self.created = set()
        self.modified = set()
        self.deleted = set()
        self.reasons = set()
        self._dirty = False

    def clear(self):
        self.created.clear()
        self.modified.clear()
        self.deleted.clear()
        self.reasons.clear()
        self._dirty = False

    def finalize(self):
        """instances should only appear in one of the
        created, modified, or deleted sets"""
        if not self._dirty:
            return
        common = self.created.intersection(self.deleted)
        if common:
            # if created and deleted, ignore
            self.created.difference_update(common)
            self.modified.difference_update(common)
            self.deleted.difference_update(common)
        self.modified.difference_update(self.created)
        self.modified.difference_update(self.deleted)
        self._dirty = False

    def empty(self):
        return not any((self.created, self.modified, self.deleted))

    def update(self, changes):
        self.created.update(changes.created)
        self.modified.update(changes.modified)
        self.deleted.update(changes.deleted)
        self.reasons.update(changes.reasons)
        self._dirty = True


class Tracker:

    def __init__(self):
        self._ts = triggerset.TriggerSet()
        self._blocked = 0
        self._changes = {}
        self._pending_changes = {}
        self._processing = False

    def register_data_type(self, data_type=None, usage_cb=None, after=(),
                           before=()):
        """Add data_type to those that are monitored for changes

        :param data_type: a consistent identifer for the data type
            to to monitored, often the class object.
        :param usage_cb: callback for when number of handlers changes
            from zero to non-zero or vice-versa.
        :param after: list of data types given to
            :py:method:`add_dependency'.

        Can be used as a decorator.
        """
        if data_type is None:
            # act as a decorator
            def wrapper(data_type, usage_cb=usage_cb, after=after,
                        before=before, self=self):
                return self.register_data_type(data_type, usage_cb, after,
                                               before)
            return wrapper
        self._changes[data_type] = Changes()
        self._pending_changes[data_type] = Changes()
        self._ts.add_trigger(data_type, usage_cb, after)
        for dt in before:
            self._ts.add_dependency(dt, [data_type])
        return data_type  # needed when used as a decorator

    def add_dependency(self, data_type, after):
        """Specify firing order dependency for 'data_type'.

        Specifies that 'data_type' should be monitored after all
        data types in the 'after' list.
        """
        self._ts.add_dependency(data_type, after)

    def add_handler(self, data_type, func):
        assert(self._changes[data_type] is not None)

        def wrapper(trigger, trigger_data, track=self, data_type=data_type,
                    func=func):
            changes = track._changes[data_type]
            changes.finalize()
            if track._processing:
                track._processed.add(data_type)
            if not changes.empty():
                return func(changes)
        return self._ts.add_handler(data_type, wrapper)

    def delete_handler(self, handler):
        self._ts.delete_handler(handler)

    def block(self):
        self._blocked += 1

    def release(self):
        assert(self._blocked > 0)
        self._processing = True
        self._processed = set()
        try:
            self._ts.block()
            for data_type, changes in self._changes.items():
                self._ts.activate_trigger(data_type, None)
            self._ts.release()
            for data_type, changes in self._changes.items():
                changes.clear()
                pending = self._pending_changes[data_type]
                if not pending.empty():
                    changes.update(pending)
                    pending.clear()
        finally:
            self._processing = False
            self._blocked -= 1

    def __enter__(self):
        self.block()

    def __exit__(self):
        self.release()

    def _activate(self, data_type):
        # TODO: simplier report of unblocked activations
        # import traceback #DEBUG
        # traceback.print_stack() #DEBUG
        changes = self._changes[data_type]
        changes.finalize()
        if not changes.empty():
            self._ts.activate_trigger(data_type, None)
            changes.clear()

    def created(self, data_type, instances):
        if data_type not in self._changes:
            raise ValueError("unknown data type")
        if self._processing and data_type in self._processed:
            changes = self._pending_changes[data_type]
            changes.created.update(instances)
            changes._dirty = True
            return
        changes = self._changes[data_type]
        changes.created.update(instances)
        changes._dirty = True
        if self._blocked:
            return
        self._activate(data_type)

    def modified(self, data_type, instances, reason):
        if data_type not in self._changes:
            raise ValueError("unknown data type")
        if self._processing and data_type in self._processed:
            changes = self._pending_changes[data_type]
            changes.modified.update(instances)
            changes.reasons.add(reason)
            changes._dirty = True
            return
        changes = self._changes[data_type]
        changes.modified.update(instances)
        changes.reasons.add(reason)
        changes._dirty = True
        if self._blocked:
            return
        self._activate(data_type)

    def deleted(self, data_type, instances):
        if data_type not in self._changes:
            raise ValueError("unknown data type")
        if self._processing and data_type in self._processed:
            changes = self._pending_changes[data_type]
            changes.deleted.update(instances)
            changes._dirty = True
            return
        changes = self._changes[data_type]
        changes.deleted.update(instances)
        changes._dirty = True
        if self._blocked:
            return
        self._activate(data_type)

tracker = Tracker()
