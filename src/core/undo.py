# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2017 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""This module defines classes for maintaining stacks of "undo"
and "redo" callbacks.  Actions can register "undo" and "redo"
functions which may be invoked via GUI, command or programmatically.
"""


class Undo:
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

    The undo manager is not derived from state.State because it
    is not possible to save and restore callback functions safely.

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

    @property
    def session(self):
        """Returns the session this undo state manager is in.
        """
        return self._session()

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

    def top_undo_name(self):
        """Return name for top undo action, or None if stack is empty.
        """
        return self._name(self.undo_stack)

    def top_redo_name(self):
        """Return name for top redo action, or None if stack is empty.
        """
        return self._name(self.redo_stack)

    def undo(self):
        """Execute top undo action.
        """
        inst = self._pop(self.undo_stack)
        try:
            inst.undo()
        except Exception as e:
            self.session.logger.report_exception("undo failed: %s" % str(e))
            self.redo_stack.clear()
        else:
            if inst.can_redo:
                self._push(self.redo_stack, inst)
            else:
                self.redo_stack.clear()
        self._update_ui()

    def redo(self):
        """Execute top redo item.
        """
        inst = self._pop(self.redo_stack)
        try:
            inst.redo()
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

    _valid_options = ["A", "M", "MA", "MK"]

    def __init__(self, name, can_redo=True):
        super().__init__(name, can_redo)
        self.state = []

    def add(self, owner, attribute, old_value, new_value, option="A"):
        """Add another tuple of (owner, attribute, old_value, new_value,
        option) to the undo action state.

        Arguments
        ---------
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
            attribute(**value).
        """
        if option not in self._valid_options:
            raise ValueError("invalid UndoState option: %s" % option)
        self.state.append((owner, attribute, old_value, new_value, option))

    def undo(self):
        """Undo action (set owner attributes to old values).
        """
        self._consistency_check()
        for owner, attribute, old_value, new_value, option in reversed(self.state):
            self._update_owner(owner, attribute, old_value, option)

    def redo(self):
        """Redo action (set owner attributes to new values).
        """
        self._consistency_check()
        for owner, attribute, old_value, new_value, option in self.state:
            self._update_owner(owner, attribute, new_value, option)

    def _consistency_check(self):
        for owner, attribute, old_value, new_value, option in self.state:
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
                    raise ValueError("undo action with different number "
                                     "of owners and old values: %d != %d" %
                                     (owner_length, value_length))

    def _update_owner(self, owner, attribute, value, option):
        if option == "A":
            setattr(owner, attribute, value)
        elif option == "M":
            getattr(owner, attribute)(value)
        elif option == "MK":
            getattr(owner, attribute)(**value)
        elif option == "MA":
            getattr(owner, attribute)(*value)
