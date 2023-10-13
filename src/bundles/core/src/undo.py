# vim: set expandtab ts=4 sw=4:

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

"""This module defines classes for maintaining stacks of "undo"
and "redo" callbacks.  Actions can register "undo" and "redo"
functions which may be invoked via GUI, command or programmatically.
"""

import abc
from .state import StateManager


class Undo(StateManager):
    """A per-session undo manager for tracking undo/redo callbacks.

    'Undo' managers are per-session singletons that track
    undo/redo callbacks in two stacks: the undo and redo stacks.
    Actions can register objects that conform to the
    'UndoAction'.  When registered, an UndoAction instance
    is pushed on to the undo stack and the redo stack is cleared.

    When an "undo" is requested, an UndoAction is popped
    off the undo stack, and its undo callback is invoked
    If the undo callback throws an error or the UndoAction
    'can_redo' attribute is false, the redo stack is cleared,
    because we cannot establish the "original" state for the
    next redo; otherwise, the UndoAction is pushed onto the
    redo stack.
    
    When a "redo" is requested, an UndoAction is popped off the
    redo stack, its redo callback is invoked, and the UndoAction
    is pushed on to the undo stack.

    Maximum stack depths are supported.  If zero, there is no
    limit.  Otherwise, if a stack grows deeper than its
    allowed maximum, the bottom of stack is discarded.

    Attributes
    ----------
    max_depth : int
        Maximum depth for both the undo and redo stacks.
        Default is 10.  Setting to 0 removes limit.
    redo_stack : list
        List of UndoAction instances
    undo_stack : list
        List of UndoAction instances
    """
    # Most of this code is modeled after tools.Tools

    def __init__(self, session, first=False, max_depth=10):
        """Initialize per-session state manager for undo/redo actions.

        Parameters
        ----------
        session : instance of chimerax.core.session.Session
            Session for which this state manager was created.
        """
        import weakref
        self._session = weakref.ref(session)
        self.max_depth = max_depth
        self.undo_stack = []
        self.redo_stack = []
        self._register_stack = []

    @property
    def session(self):
        """Returns the session this undo state manager is in.
        """
        return self._session()

    def register_push(self, handler):
        """Push handler onto undo registration stack.

        Parameters
        ----------
        handler : instance of UndoHandler
            Handler that processes registration requests

        Returns
        -------
        The registered handler.
        """
        self._register_stack.insert(0, handler)
        return handler

    def register_pop(self):
        """Pop last pushed handler from undo registration stack.

        Returns
        -------
        The popped handler.
        """
        handler = self._register_stack.pop(0)
        return handler

    def aggregate(self, name):
        return UndoAggregateHandler(self, name)

    def block(self):
        return UndoBlockHandler(self, None)

    def register(self, action):
        """Register undo/redo actions with state manager.

        Parameters
        ----------
        action : instance of UndoAction
            Action that can change session between "before"
            and "after" states.

        Returns
        -------
        The registered action.
        """
        if len(self._register_stack):
            return self._register_stack[0].register(action)
        self._push(self.undo_stack, action)
        self.redo_stack.clear()
        self._update_ui()
        return action

    def deregister(self, action, delete_history=True):
        """Deregisters undo/redo actions from state manager.
        If the action is on the undo stack, all prior undo
        actions are deleted if 'delete_history' is True
        (default).  Similarly, if the action is on the redo
        stack all subsequent redo actions are deleted if
        'delete_history' is True.  The 'delete_history'
        default is True because the deregistering action is
        the one to establish the "current" state for the
        next undo/redo action, so removing the action would
        likely prevent the next undo/redo action from working
        properly.

        Parameters
        ----------
        action : instance of UndoAction
            A previously registered UndoAction instance.
        """
        self._remove(self.undo_stack, action, delete_history)
        self._remove(self.redo_stack, action, delete_history)
        self._update_ui()

    def clear(self):
        """Clear both undo and redo stacks.
        """
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_ui()

    def top_undo_name(self):
        """Return name for top undo action, or None if stack is empty.
        """
        return self._name(self.undo_stack)

    def top_redo_name(self):
        """Return name for top redo action, or None if stack is empty.
        """
        return self._name(self.redo_stack)

    def undo(self, silent=True):
        """Execute top undo action.  Normally, if no undo action is
        available, nothing happens.  If "silent" is False, an IndexError
        is raised for accessing invalid stack location.
        """
        try:
            inst = self._pop(self.undo_stack)
        except IndexError:
            if not silent:
                raise
            else:
                return
        from .errors import UserError
        try:
            inst.undo()
        except UserError:
            raise
        except Exception as e:
            self.session.logger.report_exception("undo failed: %s" % str(e))
            self.redo_stack.clear()
        else:
            if inst.can_redo:
                self._push(self.redo_stack, inst)
            else:
                self.redo_stack.clear()
        self._update_ui()

    def redo(self, silent=True):
        """Execute top redo action.  Normally, if no redo action is
        available, nothing happens.  If "silent" is False, an IndexError
        is raised for accessing invalid stack location.
        """
        try:
            inst = self._pop(self.redo_stack)
        except IndexError:
            if not silent:
                raise
            else:
                return
        from .errors import UserError
        try:
            inst.redo()
        except UserError:
            raise
        except Exception as e:
            self.session.logger.report_exception("redo failed: %s" % str(e))
        else:
            self._push(self.undo_stack, inst)
        self._update_ui()

    def set_depth(self, depth):
        """Set the maximum depth for the undo and redo stacks.

        Parameter
        ---------
        depth : int
            Maximum depth for stacks.  Values <= 0 means unlimited.
        """
        if depth < 0:
            depth = 0
        self.max_depth = depth
        self._trim(self.undo_stack)
        self._trim(self.redo_stack)

    # State methods

    def take_snapshot(self, session, flags):
        return {"version":1, "max_depth":self.max_depth}

    @classmethod
    def restore_snapshot(cls, session, data):
        return cls(session, max_depth=data["max_depth"])

    def reset_state(self, session):
        """Reset state to data-less state"""
        self.clear()

    # Internal methods

    def _trim(self, stack):
        if self.max_depth > 0:
            while len(stack) > self.max_depth:
                stack.pop(0)

    def _push(self, stack, inst):
        stack.append(inst)
        self._trim(stack)

    def _pop(self, stack):
        return stack.pop()

    def _remove(self, stack, action, delete_history):
        try:
            n = stack.index(action)
        except ValueError:
            pass
        else:
            if delete_history:
                del stack[:n+1]
            else:
                del stack[n]

    def _name(self, stack):
        try:
            return stack[-1].name
        except IndexError:
            return None

    def _update_ui(self):
        session = self._session()
        if session is None:
            return
        try:
            f = session.ui.update_undo
        except AttributeError:
            pass
        else:
            f(self)


class UndoAction:
    """An instance holding the name for a pair of undo/redo callbacks.

    Attributes
    ----------
    name : str
        Name for the pair of undo/redo callbacks that changes
        session between start and end states.
    can_redo : boolean
        Whether this instance supports redoing an action after
        undoing it.
    """

    def __init__(self, name, can_redo=True):
        self.name = name
        self.can_redo = can_redo

    def undo(self):
        """Undo an action.
        """
        raise NotImplementedError("undo")

    def redo(self):
        """Redo an action.
        """
        raise NotImplementedError("redo")


class UndoState(UndoAction):
    """An instance that stores tuples of (owner,
    attribute name, old values, new values) and uses the
    information to undo/redo actions.  'owner' may be
    a simple instance or an ordered container such as
    a list or an 'atomic.molarray.Collection' instance.

    Attributes
    ----------
    name : str
    can_redo : boolean
        Inherited from UndoAction.
    state : list
        List of (owner, attribute, old, new, options) tuples that
        have been added to the action.
    """

    _valid_options = ["A", "M", "MA", "MK", "S"]

    def __init__(self, name, can_redo=True):
        super().__init__(name, can_redo)
        self.state = []

    def add(self, owner, attribute, old_value, new_value, option="A", *, deleted_check=lambda obj:
            hasattr(obj.__class__, 'deleted') and type(obj.__class__.deleted) == property and obj.deleted):
        """Add another tuple of (owner, attribute, old_value, new_value,
        option) to the undo action state.

        Parameters
        ----------
        owner : instance
            An instance or a container of instances.  If owner
            is a container, then undo/redo callbacks will check
            to make sure that old_value has the same number of
            elements.
        attribute : string
            Name of attribute whose value changes with undo/redo.
        old_value : object
            Value for attribute after undo.
            If owner is a container, then old_value should be
            a container of values with the same number of elements.
            Otherwise, any value is acceptable.
        new_value : object
            Value for attribute after redo.
            Even if owner is a container, new_value may be a
            simple value, in which case all elements in the
            owner container will receive the same attribute value.
        option : string
            Option specifying how the attribute and values are
            used for updating state.  If option is "A" (default),
            the attribute is changed to the value using setattr.
            If option is "M", the attribute is assumed to be callable
            with a single argument of the value.  If option is "MA",
            the attribute is called with the values as its argument
            list, i.e., attribute(*value).  If option is "MK", the
            attribute iscalled with the values as keywords, i.e.,
            attribute(**value). If option is "S" then owner is a sequence
            and old and new values are sequences of the same length
            and setattr is used to set each element of the owner sequence
            to the corresponding element of the value sequence.
        deleted_check: function
            Used to determine if the instance is still "alive" -- typically
            for instances that have a C++ component that may have been
            destroyed.  The function takes the instance as its only argument.
            By default, 'deleted_check' uses the instance's "deleted" property
            (if it exists) to decide if the object is still alive.
            Dead instances will be skipped during the undo process.
        """
        if option not in self._valid_options:
            raise ValueError("invalid UndoState option: %s" % option)
        self.state.append((owner, attribute, old_value, new_value, option, deleted_check))

    def undo(self):
        """Undo action (set owner attributes to old values).
        """
        self._consistency_check()
        for owner, attribute, old_value, new_value, option, deleted_check in reversed(self.state):
            self._update_owner(owner, attribute, old_value, option, deleted_check)

    def redo(self):
        """Redo action (set owner attributes to new values).
        """
        self._consistency_check()
        for owner, attribute, old_value, new_value, option, deleted_check in self.state:
            self._update_owner(owner, attribute, new_value, option, deleted_check)

    def _consistency_check(self):
        for owner, attribute, old_value, new_value, option, deleted_check in self.state:
            try:
                owner_length = len(owner)
            except TypeError:
                # Not a container, so move on
                continue
            else:
                # Is a container, old_value must be the same length
                try:
                    value_length = len(old_value)
                except TypeError:
                    value_length = 1
                if value_length != owner_length:
                    from .errors import UserError
                    raise UserError("Undo failed, probably because "
                                     "structures have been modified.")

    def _update_owner(self, owner, attribute, value, option, deleted_check):
        if option != "S":
            if deleted_check(owner):
                return
        if option == "A":
            setattr(owner, attribute, value)
        elif option == "M":
            getattr(owner, attribute)(value)
        elif option == "MK":
            getattr(owner, attribute)(**value)
        elif option == "MA":
            getattr(owner, attribute)(*value)
        elif option == "S":
            for e,v in zip(owner, value):
                if deleted_check(e):
                    continue
                setattr(e, attribute, v)


class UndoHandler(metaclass=abc.ABCMeta):
    """An instance that intercepts undo registration
    requests.  For example, multiple undo actions may
    be aggregated into a single undo action; or
    undo actions may be blocked and replaced with
    a more efficient undo mechanism.
    """

    def __init__(self, mgr, name):
        """Initialize undo handler.

        Parameters
        ----------
        mgr : instance of Undo
            Undo manager for which this undo handler was created.
        name : str
            Name of undo action to register
        """
        self.name = name
        self.mgr = mgr

    def __enter__(self):
        self.mgr.register_push(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.mgr.register_pop()
        if not exc_type:
            self.finish()

    @abc.abstractmethod
    def register(self, action):
        """Register undo/redo actions.

        Parameters
        ----------
        action : instance of UndoAction
            Action that can change session between "before"
            and "after" states.

        Returns
        -------
        The registered action.
        """
        pass

    @abc.abstractmethod
    def finish(self):
        """Finish processing intercepted registration requests."""
        pass


class UndoAggregateHandler(UndoHandler):
    """An instance that intercepts undo registration
    requests and aggregates them into a single undo action.
    """

    def __init__(self, mgr, name):
        super().__init__(mgr, name)
        self.actions = []

    def register(self, action):
        self.actions.append(action)

    def finish(self):
        a = UndoAggregateAction(self.name, self.actions)
        self.mgr.register(a)


class UndoAggregateAction(UndoAction):
    """An instance that executes a list of UndoAction
    instances as a group."""

    def __init__(self, name, actions):
        can_redo = all([a.can_redo for a in actions])
        super().__init__(name, can_redo=can_redo)
        self.actions = actions

    def undo(self):
        for a in reversed(self.actions):
            a.undo()

    def redo(self):
        for a in self.actions:
            a.redo()


class UndoBlockHandler(UndoHandler):
    """An instance that intercepts undo registration
    requests and discards them.
    """

    def register(self, action):
        pass

    def finish(self):
        pass
