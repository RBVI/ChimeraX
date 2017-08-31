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

    'Undo' instances are per-session singletons that track
    undo/redo callbacks in two stacks: the undo and redo stacks.
    Actions can register a pair of undo and redo callbacks along
    with a name; the triplet is stored as an UndoInstance.
    When actions register new undo/redo callbacks, the UndoInstance
    is pushed on to the undo stack and the redo stack is cleared.
    When an "undo" is requested, an UndoInstance is popped
    off the undo stack, the undo callback is invoked, and
    the UndoInstance is pushed onto the redo stack.  When
    a "redo" is requested, an UndoInstance is popped off the
    redo stack, the redo callback is invoked, and the UndoInstance
    is pushed on to the undo stack.
    
    If an action registers only an undo but no redo callback,
    then the UndoInstance is not pushed onto the redo stack when
    "undo" is requested; instead, the redo stack is cleared (since
    the "original" state for the next redo cannot be restored).

    A maximum stack depth is supported.  If zero, there is no
    limit.  Otherwise, if either stack grows deeper than the
    allowed maximum, the bottom of stack is discarded.

    The undo manager is not derived from state.State because it
    is not possible to save and restore callback functions safely.

    Attributes
    ----------
    max_depth : int
        Maximum depth for both undo and redo stacks.
        Default is 10.  Setting to 0 removes limit.
    undo_stack : list
        List of UndoInstance instances
    redo_stack : list
        List of UndoInstance instances
    """
    # Most of this code is modeled after tools.Tools

    def __init__(self, session, first=False, max_depth=10):
        """Initialize per-session state manager for undo/redo callbacks.

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

    def register(self, name, undo, redo):
        """Register undo/redo callbacks with state manager.

        Parameters
        ----------
        name : str
            Name for the pair of undo/redo callbacks that changes
            session between start and end states.
        undo : function
            Function to execute to go from end state to start state.
        redo : function
            Function to execute to go from start state to end state.
        """
        if not name:
            raise ValueError("undo name must not be empty")
        self._push(self.undo_stack, UndoInstance(name, undo, redo))
        self.redo_stack.clear()
        self._update_ui()

    def top_undo_name(self):
        """Return name for top undo item, or None if stack is empty.
        """
        return self._name(self.undo_stack)

    def top_redo_name(self):
        """Return name for top redo item, or None if stack is empty.
        """
        return self._name(self.redo_stack)

    def undo(self):
        """Execute top undo item.
        """
        inst = self._pop(self.undo_stack)
        inst.undo()
        if inst.redo:
            self._push(self.redo_stack, inst)
        else:
            self.redo_stack.clear()
        self._update_ui()

    def redo(self):
        """Execute top redo item.
        """
        inst = self._pop(self.redo_stack)
        inst.redo()
        self._push(self.undo_stack, inst)
        self._update_ui()

    def _push(self, stack, inst):
        stack.append(inst)
        if self.max_depth > 0:
            while len(stack) > self.max_depth:
                stack.pop(0)

    def _pop(self, stack):
        return stack.pop()

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


class UndoInstance:
    """An instance holding the name for a pair of undo/redo callbacks.

    Attributes
    ----------
    name : str
        Name for the pair of undo/redo callbacks that changes
        session between start and end states.
    undo : function
        Function to execute to go from end state to start state.
    redo : function
        Function to execute to go from start state to end state.
    """

    def __init__(self, name, undo, redo):
        self.name = name
        self.undo = undo
        self.redo = redo
