# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
triggerset: Support for managing triggers and handlers
======================================================

This module defines one class, TriggerSet, which implements a simple callback
mechanism.  A TriggerSet instance contains a set of named triggers, each of
which may have a number of handlers registered with it. Activating a trigger
in the instance causes all its handlers to be called, in the order of
registration.

Example
-------

The following example creates a TriggerSet instance named ts and adds a
trigger named conrad. Two handlers are registered: the first reports its
arguments; the second reports its arguments and then deregisters itself.

::

    import triggerset

    ts = triggerset.TriggerSet()
    ts.add_trigger('conrad')

    def first(trigger, trigger_data):
        print('trigger =', trigger)
        print('  trigger_data =', trigger_data)

    class Second:

        def __init__(self, ts):
            self.triggerset = ts
            self.handler = None

        def trigger_handler(self, trigger, trigger_data):
            print('trigger =', trigger)
            print('  triggerset =', self.triggerset)
            print('  handler =', self.handler)
            print('  trigger_data =', trigger_data)
            if self.triggerset and self.handler:
                self.triggerset.delete_handler(self.handler)
                self.handler = None

    h1 = ts.add_handler('conrad', first)
    o = Second(ts)
    o.handler = ts.add_handler('conrad', o.trigger_handler)

    ts.activate_trigger('conrad', 1)
    print()
    ts.activate_trigger('conrad', 2)

The output from this example is:

::

    trigger = conrad
      trigger_data = 1
    trigger = conrad
      triggerset = <triggerset.TriggerSet instance at 1400f3010>
      handler = <triggerset._TriggerHandler instance at 140097ac0>
      trigger_data = 1

    trigger = conrad trigger_data = 2

If a handler returns the value triggerset.DEREGISTER, then the handler will
be deregistered after it returns.  Therfore, the 'Second.handler()' method
above could have been written more simply as:

::

    def trigger_handler(trigger, trigger_data):
        print('trigger =', trigger,
            'triggerset =', self.triggerset,
            'handler =', self.handler,
            'trigger_data =', trigger_data)
        self.handler = None
        return triggerset.DEREGISTER
"""

DEREGISTER = "delete handler"
TRIGGER_ERROR = "Error processing trigger"


def _basic_report(msg):
    import sys
    import traceback
    sys.stdout.write(msg + "\n")
    traceback.print_exc(file=sys.stdout)


_report = _basic_report


def set_exception_reporter(f):
    global _report
    old = _report
    if f is None:
        _report = _basic_report
    else:
        _report = f
    return old


class _TriggerHandler:
    """Describes callback routine registered with _Trigger"""

    def __init__(self, name, func):
        self._name = name
        self._func = func

    def invoke(self, data, remove_if_error):
        try:
            return self._func(self._name, data)
        except Exception:
            _report('%s "%s"' % (TRIGGER_ERROR, self._name))
            if remove_if_error:
                return DEREGISTER


class _Trigger:
    """Keep track of handlers to invoke when activated"""

    def __init__(self, name, usage_cb, default_one_time, remove_bad_handlers):
        self._name = name
        self._handlers = set()
        self._pending_add = set()
        self._pending_del = set()
        self._locked = 0
        self._blocked = 0
        self._need_activate = set()
        self._need_activate_data = []
        self._usage_cb = usage_cb
        self._default_one_time = default_one_time
        self._remove_bad_handlers = remove_bad_handlers

    def add(self, handler):
        if self._locked:
            self._pending_add.add(handler)
        else:
            self._handlers.add(handler)
        if (self._usage_cb
                and len(self._pending_add) + len(self._handlers) == 1):
            self._usage_cb(self._name, 1)

    def delete(self, handler):
        if self._locked:
            try:
                self._pending_add.remove(handler)
            except KeyError:
                self._pending_del.add(handler)
        else:
            self._handlers.discard(handler)
        if self._usage_cb and len(self._handlers) == len(self._pending_del):
                self._usage_cb(self._name, 0)

    def activate(self, data):
        if self._blocked:
            # don't raise trigger multiple times for identical
            # data
            if id(data) not in self._need_activate:
                self._need_activate.add(id(data))
                self._need_activate_data.append(data)
            return
        locked = self._locked
        self._locked = True
        for handler in self._handlers:
            if handler in self._pending_del:
                continue
            if self._default_one_time:
                self._pending_del.append(handler)
            try:
                ret = handler.invoke(data, self._remove_bad_handlers)
            except:
                self._locked = locked
                raise
            if ret == DEREGISTER and handler not in self._pending_del:
                    self._pending_del.add(handler)
        self._locked = locked
        if not self._locked:
            self._handlers -= self._pending_del
            self._pending_del.clear()
            self._handlers |= self._pending_add
            self._pending_add.clear()

    def block(self):
        self._blocked = self._blocked + 1

    def is_blocked(self):
        return bool(self._blocked)

    def release(self):
        if self._blocked <= 0:
            raise RuntimeError("more releases than blocks")
        self._blocked = self._blocked - 1
        if not self._need_activate or self._blocked:
            return
        for data in self._need_activate_data:
            self.activate(data)
        self._need_activate_data.clear()
        self._need_activate.clear()

    def num_handlers(self):
        return (len(self._handlers) + len(self._pending_add)
                - len(self._pending_del))

    def list_handlers(self):
        '''Return a list of all the enabled handlers
        - each list element is a callback function
        '''
        return [h._func for h in self._handlers if h not in self._pending_del]


class TriggerSet:
    """Keep track of related groups of triggers."""

    def __init__(self):
        self._triggers = {}
        self._roots = set()
        self._dependents = {}
        self._block_data = {}
        self._blocked = 0

    def add_trigger(self, name, *, usage_cb=None, after=None,
                    default_one_time=False, remove_bad_handlers=False):
        """Add a trigger with the given name.

        triggerset.add_trigger(name) => None

        The name should be a string.  If a trigger by the same name
        already exists, an exception is raised.

        The optional 'usage_cb' argument can be used to provide a
        callback function for when the trigger goes from no handlers
        registered to at least one registered, and vice versa.
        The callback function will be given the trigger name and 1
        or 0 (respectively) as arguments.

        The optional argument 'default_one_time' (default False)
        may be used to designate triggers whose registered handlers
        should only be called once.  For example, an "exit trigger"
        may only want its handler run once and then discarded
        without having the handler explicitly deregister itself
        or return DEREGISTER.

        The optional argument 'after' (default None) may be a list
        of trigger names, and is passed to a call to 'add_dependency'
        after the new trigger has been created.

        if 'remove_bad_handlers' is True, then handlers that throw
        errors will be removed from the list of handlers if they
        throw an error.
        """
        if name in self._triggers:
            raise KeyError("Trigger '%s' already exists" % name)
        self._triggers[name] = _Trigger(name, usage_cb, default_one_time,
            remove_bad_handlers)
        self._roots.add(name)
        if after:
            self.add_dependency(name, after)

    def delete_trigger(self, name):
        """Remove a trigger with the given name.

        triggerset.delete_trigger(name) => None

        The name should be a string.  If no trigger corresponds to
        the name, an exception is raised.
        """
        del self._triggers[name]

    def activate_trigger(self, name, data, absent_okay=False):
        """Invoke all handlers registered with the given name.

        triggerset.activate_trigger(name, data) => None

        If no trigger corresponds to name, an exception is raised.
        Handlers are invoked in the order in which they were
        registered, and are called in the following manner:

        func(name, data)

        where func is the function previously registered with
        add_handler, name is the name of the trigger, and data
        is the data argument to the activate_trigger() call.

        During trigger activation, handlers may add new handlers or
        delete existing handlers.  These operations, however, are
        deferred until after all handlers have been invoked; in
        particular, for the current trigger activation, newly added
        handlers will not be invoked and newly deleted handlers will
        be invoked.
        """
        if self._blocked:
            try:
                dl = self._block_data[name]
            except KeyError:
                dl = []
                self._block_data[name] = dl
            dl.append((data, absent_okay))
            return
        try:
            trigger = self._triggers[name]
        except KeyError:
            if not absent_okay:
                raise
        else:
            trigger.activate(data)

    def block_trigger(self, name):
        """Block all handlers registered with the given name.

        triggerset.block_trigger(name) => None

        If no trigger corresponds to name, an exception is raised.
        block_trigger()/release_trigger() may be nested inside other
        block_trigger()/release_trigger() pairs.
        """
        self._triggers[name].block()

    def is_trigger_blocked(self, name):
        """Is named trigger blocked?"""
        return self._triggers[name].is_blocked()

    def release_trigger(self, name):
        """Release all handlers registered with the given name.

        triggerset.release_trigger(name) => None

        If no trigger corresponds to name, an exception is raised.
        The last call to activate_trigger() made between the outermost
        block_trigger()/release_trigger() pair is executed.
        """
        self._triggers[name].release()

    def has_trigger(self, name):
        """Check if trigger exists."""
        return name in self._triggers

    def add_handler(self, name, func):
        """Register a function with the trigger with the given name.

        triggerset.add_handler(name, func) => handler

        If no trigger corresponds to name, an exception is raised.
        add_handler returns a handler for use with delete_handler.
        """
        if name not in self._triggers:
            raise KeyError("No trigger named '%s'" % name)
        handler = _TriggerHandler(name, func)
        self._triggers[name].add(handler)
        return handler

    def delete_handler(self, handler):
        """Deregister a handler.

        triggerset.delete_handler(handler) => None

        The handler should be the return value from a previous call
        to add_handler.  If the given handler is invalid, an
        exception is raised.
        """
        self._triggers[handler._name].delete(handler)

    def has_handlers(self, name):
        """Return if the trigger with the given name has any handlers.

        triggerset.has_handlers(name) => bool

        If no trigger corresponds to name, an exception is raised.
        """
        return self._triggers[name].num_handlers() != 0

    def trigger_handlers(self, name):
        """Return functions registered for trigger with the given name.

        triggerset.trigger_handlers(name) => list

        If no trigger corresponds to name, an exception is raised.
        """
        return self._triggers[name].list_handlers()

    def trigger_names(self):
        """Return an unordered list of the trigger names (dict keys).

        triggerset.trigger_names() => list
        """
        return self._triggers.keys()

    def block(self):
        """Block all triggers from firing until released.

        triggerset.block() => None
        """
        self._blocked += 1

    def release(self):
        """Release trigger blocks and fire trigger in dependency order.

        triggerset.release() => boolean

        Return 'True' if triggers were blocked, 'False' otherwise.

        Dependency order is defined by either the 'add_dependency'
        method or providing list specified in the 'after' keyword
        to 'add_trigger'.
        """
        if self._blocked <= 0:
            raise RuntimeError("more releases than blocks")
        self._blocked -= 1
        if self._blocked:
            return
        # Fire triggers in dependency order
        for name in self._roots:
            self._activate_trigger_tree(name)

    def is_blocked(self):
        return bool(self._blocked)

    def add_dependency(self, trigger, after):
        """Specify firing order dependency for 'trigger'.

        triggerset.add_dependency(trigger, after) => None

        Specifies that 'trigger' should be fired after all
        triggers in the 'after' list when a trigger set is
        blocked and released.  If the dependency relationship
        conflicts with a previously specified dependency,
        a 'ValueError' exception is thrown.  If any name given
        is not a trigger name, a 'KeyError' exception is thrown.

        Note that the dependency information is _only_ used
        when an entire trigger set is blocked and released,
        and not used for blocking and releasing of individual
        triggers."""
        # Add dependencies one-by-one, tracking which
        # ones have been added.  If any dependency creates a
        # loop, delete all the dependencies that have been
        # added and raise an exception.
        if not self.has_trigger(trigger):
            raise KeyError("No trigger named '%s'" % trigger)
        added = []
        for name in after:
            if not self.has_trigger(name):
                self._remove_dependencies(added)
                raise KeyError("No trigger named '%s'" % name)
            if self._find_dependency(name, trigger):
                self._remove_dependencies(added)
                raise ValueError("Circular dependency '%s'->'%s'"
                                 % (trigger, name))
            try:
                dl = self._dependents[trigger]
            except KeyError:
                dl = []
                self._dependents[trigger] = dl
            dl.append(name)
            added.append((trigger, name))
        for src, dst in added:
            self._roots.discard(dst)

    def _remove_dependencies(self, added):
        for trigger, name in added:
            try:
                dl = self._dependents[trigger]
            except KeyError:
                continue
            dl.remove(name)
            if not dl:
                del self._dependents[trigger]

    def _find_dependency(self, src, dst):
        # Check whether there is a path from src to dst.
        try:
            dl = self._dependents[src]
        except KeyError:
            return False
        for ssrc in dl:
            if ssrc == dst:
                return True
            if self._find_dependency(ssrc, dst):
                return True
        else:
            return False

    def _activate_trigger_tree(self, name):
        # Fire dependent triggers
        try:
            dl = self._dependents[name]
        except KeyError:
            pass
        else:
            for aname in dl:
                self._activate_trigger_tree(aname)
        # Fire this trigger if it has pending data
        try:
            bl = self._block_data[name]
        except KeyError:
            return
        else:
            for data, ao in bl:
                self.activate_trigger(name, data, absent_okay=ao)
            del self._block_data[name]

if __name__ == "__main__":
    first_trigger = None

    def first(trigger, trigger_data):
        global first_trigger
        first_trigger = trigger_data

    class Second:

        def __init__(self, ts):
            self.data = None
            self.triggerset = ts
            self.handler = None

        def trigger_handler(self, trigger, trigger_data):
            self.data = trigger_data
            if self.triggerset and self.handler:
                self.triggerset.delete_handler(self.handler)
                self.handler = None

    def bad_handler(trigger, trigger_data):
        raise RuntimeError("cannot deal with '%s' trigger" % trigger)

    import unittest

    class TestTriggerset(unittest.TestCase):

        def test_single_trigger(self):
            ts = TriggerSet()
            ts.add_trigger('conrad')
            h1 = ts.add_handler('conrad', first)    # noqa
            o = Second(ts)
            o.handler = ts.add_handler('conrad', o.trigger_handler)

            global first_trigger
            ts.activate_trigger('conrad', 1)
            self.assertEqual(first_trigger, 1)
            self.assertEqual(o.data, 1)

            first_trigger = None
            o.data = None
            ts.activate_trigger('conrad', 2)
            self.assertEqual(first_trigger, 2)
            self.assertEqual(o.data, None)

        def test_dependency(self):
            ts = TriggerSet()
            ts.add_trigger('a')
            ts.add_trigger('b', after=['a'])
            ts.add_trigger('c', after=['a'])
            ts.add_trigger('d', after=['b', 'c'])

            result = []

            def log_trigger(trigger, data, result=result):
                    result.append(trigger)
            ts.add_handler('a', log_trigger)
            ts.add_handler('b', log_trigger)
            ts.add_handler('c', log_trigger)
            ts.add_handler('d', log_trigger)

            ts.block()
            ts.activate_trigger('a', 1)
            ts.activate_trigger('d', 2)
            ts.activate_trigger('c', 3)
            ts.activate_trigger('b', 4)
            ts.release()
            self.assertTrue(result == ['a', 'b', 'c', 'd'] or
                            result == ['a', 'c', 'b', 'd'])

        def test_bad_dependency(self):
            ts = TriggerSet()
            ts.add_trigger('a')
            ts.add_trigger('b', after=['a'])
            self.assertRaises(KeyError, ts.add_dependency,
                              'c', ['a'])
            self.assertRaises(KeyError, ts.add_dependency,
                              'a', ['c'])
            self.assertRaises(ValueError, ts.add_dependency,
                              'a', ['b'])

        def test_handler_exceptions(self):
            import sys
            save = sys.stdout

            ts = TriggerSet()
            ts.add_trigger('a')
            ts.add_handler('a', bad_handler)

            try:
                from StringIO import StringIO
            except ImportError:
                from io import StringIO
            sys.stdout = StringIO()
            ts.activate_trigger('a', 1)
            self.assertTrue(TRIGGER_ERROR in sys.stdout.getvalue())
            error_message = []

            def report_error(msg):
                error_message.append(msg)
            set_exception_reporter(report_error)
            sys.stdout = StringIO()
            ts.activate_trigger('a', 2)
            self.assertFalse(TRIGGER_ERROR in sys.stdout.getvalue())
            self.assertTrue(TRIGGER_ERROR in error_message[0])

            set_exception_reporter(None)
            sys.stdout = save

    unittest.main()
