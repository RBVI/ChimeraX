# vi: set expandtab ts=4 sw=4:

"""This module defines classes for running tools and their state manager.

Attributes
----------
ADD_TOOL_INSTANCE : str
    Name of trigger that is fired when a new tool
    registers with the state manager.

REMOVE_TOOL_INSTANCE : str
    Name of trigger that is fired when a running tool
    deregisters with the state manager.

Notes
-----
'ToolInstance' and 'Tools' instances are session-specific.
The 'Tools' instance is a singleton per session and may be
referenced as `session.tools`.
All running tools may be found via `session.tools`.

The triggers are also session-specific.  To add and remove
'ToolInstance' handlers, use `session.trigger.add_handler`
and `session.trigger.delete_handler`.
"""


# Tools and ToolInstance are session-specific
from .session import State
ADD_TOOL_INSTANCE = 'add tool instance'
REMOVE_TOOL_INSTANCE = 'remove tool instance'


class ToolInstance(State):
    """Base class for instances of running tools.

    Classes for running tools should inherit from 'ToolInstance'
    and override methods to implement tool-specific functionality.
    In particularly, methods from `session.State` should be defined
    so that saving and restoring of scenes and sessions work properly.

    Attributes
    ----------
    id : readonly int
        `id` is a unique identifier among ToolInstance instances
        registered with the session state manager.

    """

    def __init__(self, session, id=None, **kw):
        """Initialize a ToolInstance.

        Parameters
        ----------
        session : instance of chimera.core.session.Session
            Session in which this tool instance was created.

        """
        self.id = id
        import weakref
        self._session = weakref.ref(session)
        # TODO: track.created(ToolInstance, [self])

    @property
    def session(self):
        """Read-only property for session that contains this tool instance."""
        return self._session()

    def display_name(self):
        """Name to display to user for this ToolInstance.

        This method should be overridden, particularly
        for multi-instance tools.

        """
        return self.__class__.__name__

    def delete(self):
        """Delete this tool instance.

        This method should be overridden to clean up
        tool data structures.  This base method should be
        called as the last step of tool deletion.

        """
        if self.id is not None:
            raise ValueError("tool instance is still in use")
        # TODO: track.deleted(ToolInstance, [self])

    def display(self, b):
        """Show or hide this tool instance in the user interface.

        Parameters
        ----------
        b : boolean
            Boolean value for whether the tool should be shown or hidden.

        """
        pass


class Tools(State):
    """A per-session state manager for running tools.

    'Tools' instances are per-session singletons that track
    running tool instances in the session, as well as managing
    saving and restoring tool states for scenes and sessions.
    """
    # Most of this code is modeled after models.Models

    VERSION = 1     # snapshot version

    def __init__(self, session):
        """Initialize per-session state manager for running tools.

        Parameters
        ----------
        session : instance of chimera.core.session.Session
            Session for which this state manager was created.

        """
        import weakref
        self._session = weakref.ref(session)
        session.triggers.add_trigger(ADD_TOOL_INSTANCE)
        session.triggers.add_trigger(REMOVE_TOOL_INSTANCE)
        self._tool_instances = {}
        import itertools
        self._id_counter = itertools.count(1)

    def take_snapshot(self, session, flags):
        """Save state of running tools.

        Overrides chimera.core.session.State default method to save
        state of all registered running tool instances.

        Parameters
        ----------
        session : instance of chimera.core.session.Session
            Session for which state is being saved.
            Should match the `session` argument given to `__init__`.
        flags : int
            Flags indicating whether snapshot is being taken to
            save scene or session.  See `chimera.core.session` for
            more details.

        """
        data = {}
        for tid, ti in self._tool_instances.items():
            assert(isinstance(ti, ToolInstance))
            data[tid] = [session.unique_id(ti), ti.take_snapshot(session, flags)]
        return [self.VERSION, data]

    def restore_snapshot(self, phase, session, version, data):
        """Restore state of running tools.

        Overrides chimera.core.session.State default method to restore
        state of all registered running tools.

        Parameters
        ----------
        phase : str
            Restoration phase.  See `chimera.core.session` for more details.
        session : instance of chimera.core.session.Session
            Session for which state is being saved.
            Should match the `session` argument given to `__init__`.
        version : any
            Version of state manager that saved the data.
            Used for determining how to parse the `data` argument.
        data : any
            Data saved by state manager during `take_snapshot`.

        """
        if version != self.VERSION or not data:
            raise RuntimeError("Unexpected version or data")

        session = self._session()   # resolve back reference
        for tid, [uid, [ti_version, ti_data]] in data.items():
            if phase == State.PHASE1:
                try:
                    cls = session.class_of_unique_id(uid, ToolInstance)
                except KeyError:
                    class_name = session.class_name_of_unique_id(uid)
                    session.log.warning("Unable to restore tool instance %s (%s)"
                                        % (tid, class_name))
                    continue
                ti = cls(session, id=tid)
                session.restore_unique_id(ti, uid)
            else:
                ti = session.unique_obj(uid)
            ti.restore_snapshot(phase, session, ti_version, ti_data)

    def reset_state(self):
        """Reset state manager to default state.

        Overrides chimera.core.session.State default method to reset
        to default state.  Since the default state has no running
        tools, all registered tool instances are deleted.

        """
        ti_list = list(self._tool_instances.values())
        for ti in ti_list:
            ti.delete()
        self._tool_instances.clear()

    def list(self):
        """Return list of running tools.

        Returns
        -------
        list
            List of ToolInstance instances.

        """
        return list(self._tool_instances.values())

    def add(self, ti_list):
        """Register running tools with state manager.

        Parameters
        ----------
        ti_list : list of ToolInstance instances
            List of newly created running tool instances.

        """
        session = self._session()   # resolve back reference
        for ti in ti_list:
            if ti.id is None:
                ti.id = next(self._id_counter)
            self._tool_instances[ti.id] = ti
        session.triggers.activate_trigger(ADD_TOOL_INSTANCE, ti_list)

    def remove(self, ti_list):
        """Deregister running tools with state manager.

        Parameters
        ----------
        ti_list : list of ToolInstance instances
            List of registered running tool instances.

        """
        session = self._session()   # resolve back reference
        for ti in ti_list:
            tid = ti.id
            if tid is None:
                # Not registered in a session
                continue
            ti.id = None
            del self._tool_instances[tid]
        session.triggers.activate_trigger(REMOVE_TOOL_INSTANCE, ti_list)

    def find_by_id(self, tid):
        """Return a tool instance with the matching identifier.

        Parameters
        ----------
        tid : int
            Unique per-session identifier for a registered tool.

        """
        return self._tool_instances.get(tid, None)

    def find_by_class(self, cls):
        """Return a list of tools of the given class.

        All tool instances that match `cls` as defined by `isinstance`
        are returned.

        Parameters
        ----------
        cls : class object
            Class object used to match tool instances.

        """
        return [ti for ti in self._tool_instances.values() if isinstance(ti, cls)]
