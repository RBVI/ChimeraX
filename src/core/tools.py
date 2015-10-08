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
from .session import State, RestoreError
ADD_TOOL_INSTANCE = 'add tool instance'
REMOVE_TOOL_INSTANCE = 'remove tool instance'


class ToolInstance(State):
    """Base class for instances of running tools.

    Classes for running tools should inherit from 'ToolInstance'
    and override methods to implement tool-specific functionality.
    In particularly, methods from `session.State` should be defined
    so that saving and restoring of scenes and sessions work properly.

    Parameters
    ----------
    session : instance of chimera.core.session.Session
        Session in which this tool instance was created.
    tool_info : a :py:class:`~chimera.core.toolshed.ToolInfo` instance
        The tool information used to create this tool.
    id : int, optional
        See attribute.

    Attributes
    ----------
    id : readonly int
        `id` is a unique identifier among ToolInstance instances
        registered with the session state manager.
    display_name : str
        If a different name is desired (e.g. multi-instance tool) make sure
        to set the attribute before creating the first tool window.
        Defaults to ``tool_info.display_name``.
    SESSION_ENDURING : bool, class-level optional
        If True, then tool survives across sessions.
    SESSION_SKIP : bool, class-level optional
        If True, then tool is not saved in sessions.
    help : str
        URL for tool's help
    """

    SESSION_ENDURING = False
    SESSION_SKIP = False
    help = None

    def __init__(self, session, tool_info, id=None, **kw):
        self.id = id
        import weakref
        self._session = weakref.ref(session)
        self.tool_info = tool_info
        self.display_name = tool_info.display_name
        # TODO: track.created(ToolInstance, [self])

    @property
    def session(self):
        """Read-only property for session that contains this tool instance."""
        return self._session()

    def delete(self):
        """Delete this tool instance.

        This method should be overridden to clean up
        tool data structures.  This base method should be
        called as the last step of tool deletion.

        """
        if self.session.ui.is_gui:
            self.session.ui.remove_tool(self)
        self.session.tools.remove([self])
        # TODO: track.deleted(ToolInstance, [self])

    def display(self, b):
        """Show or hide this tool instance in the user interface.

        Parameters
        ----------
        b : boolean
            Boolean value for whether the tool should be shown or hidden.

        """
        if self.session.ui.is_gui:
            self.session.ui.set_tool_shown(self, b)

    def display_help(self):
        """Show the help for this tool in the help viewer."""
        from chimera.core.commands import run
        run(self.session,
            'help %s' % self.help if self.help is not None else "")


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

    def take_snapshot(self, phase, session, flags):
        """Save state of running tools.

        Overrides chimera.core.session.State default method to save
        state of all registered running tool instances.

        Parameters
        ----------
        session : instance of chimera.core.session.Session
            Session for which state is being saved.
            Should match the `session` argument given to `__init__`.
        phase : str
            Take phase.  See `chimera.core.session` for more details.
        flags : int
            Flags indicating whether snapshot is being taken to
            save scene or session.  See `chimera.core.session` for
            more details.

        """
        if phase == self.CLEANUP_PHASE:
            for ti in self._tool_instances.values():
                if ti.SESSION_SKIP:
                    continue
                ti.take_snapshot(session, phase, flags)
            return
        if phase == self.SAVE_PHASE:
            data = {}
            for tid, ti in self._tool_instances.items():
                assert(isinstance(ti, ToolInstance))
                if ti.SESSION_SKIP:
                    continue
                data[tid] = [ti.tool_info.name, ti.tool_info.version,
                             session.unique_id(ti),
                             ti.take_snapshot(session, phase, flags)]
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
            raise RestoreError("Unexpected version")

        session = self._session()   # resolve back reference
        for tid, [tool_name, tool_version, uid, [ti_version, ti_data]] in data.items():
            if phase == State.CREATE_PHASE:
                t = session.toolshed.find_tool(tool_name, version=tool_version)
                if t is None:
                    # TODO: load tool from toolshed
                    session.logger.error("Missing '%s' (internal name) tool" % tool_name)
                    return
                try:
                    ti = t.start(session)
                    if ti is None:
                        # GUI tool restored in nogui application
                        continue
                    session.restore_unique_id(ti, uid)
                except Exception as e:
                    class_name = session.class_name_of_unique_id(uid)
                    session.logger.error(
                        "Code error restoring tool instance: %s (%s): %s" %
                        (tid, class_name, str(e)))
                    raise
            else:
                ti = session.unique_obj(uid)
                if ti is None:
                    # GUI tool restored in nogui application
                    continue
            ti.restore_snapshot(phase, session, ti_version, ti_data)

    def reset_state(self):
        """Reset state manager to default state.

        Overrides chimera.core.session.State default method to reset
        to default state.  Since the default state has no running
        tools, all registered tool instances are deleted.

        """
        items = list(self._tool_instances.items())
        for id, ti in items:
            if ti.SESSION_ENDURING:
                continue
            ti.delete()
            assert(id not in self._tool_instances)

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

    def autostart(self):
        """Start tools that should start when applications starts up."""
        session = self._session()   # resolve back reference
        from .toolshed import ToolshedError
        from .core_settings import settings
        for ti in session.toolshed.tool_info():
            if ti.name not in settings.autostart:
                continue
            try:
                ti.start(session)
            except ToolshedError as e:
                self.session.logger.info("Tool \"%s\" failed to start"
                                         % ti.name)
                print("{}".format(e))
