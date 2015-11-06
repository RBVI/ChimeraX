# vim: set expandtab ts=4 sw=4:

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
from .state import State, RestoreError, CORE_STATE_VERSION
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

    def __init__(self, session, tool_info, id=None):
        self.id = id
        import weakref
        self._session = weakref.ref(session)
        self.tool_info = tool_info
        self.display_name = tool_info.display_name
        # TODO: track.created(ToolInstance, [self])
        session.tools.add([self])

    def take_snapshot(self, session, flags):
        return CORE_STATE_VERSION, [self.id, self.tool_info.name]

    def restore_snapshot_init(self, session, tool_info, version, data):
        id, tool_name = data
        tool_info = session.toolshed.find_tool(tool_name)
        if tool_info is None:
            session.logger.info('unable to find tool "%s"' % tool_name)
            return
        ToolInstance.__init__(self, session, tool_info, id=id)

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
            self.session.ui.thread_safe(lambda s=self, show=b:
                s.session.ui.set_tool_shown(s, show))

    def display_help(self):
        """Show the help for this tool in the help viewer."""
        from chimera.core.commands import run
        run(self.session,
            'help %s' % self.help if self.help is not None else "")


def get_singleton(session, tool_class, tool_name, create=True, display=False, **kw):
    if not session.ui.is_gui:
        return None
    running = [t for t in session.tools.find_by_class(tool_class)
               if t.tool_info.name == tool_name]
    if len(running) > 1:
        raise RuntimeError("too many %s instances running" % tool_name)
    if not running:
        if create:
            tool_info = session.toolshed.find_tool(tool_name)
            tinst = tool_class(session, tool_info, **kw)
        else:
            tinst = None
    else:
        tinst = running[0]
    if display and tinst:
        tinst.display(True)
    return tinst

class Tools(State):
    """A per-session state manager for running tools.

    'Tools' instances are per-session singletons that track
    running tool instances in the session, as well as managing
    saving and restoring tool states for scenes and sessions.
    """
    # Most of this code is modeled after models.Models

    def __init__(self, session, first=False):
        """Initialize per-session state manager for running tools.

        Parameters
        ----------
        session : instance of chimera.core.session.Session
            Session for which this state manager was created.

        """
        import weakref
        self._session = weakref.ref(session)
        if first:
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
        phase : str
            Take phase.  See `chimera.core.session` for more details.
        flags : int
            Flags indicating whether snapshot is being taken to
            save scene or session.  See `chimera.core.session` for
            more details.

        """
        data = {}
        for tid, tool_inst in self._tool_instances.items():
            assert(isinstance(tool_inst, ToolInstance))
            if tool_inst.SESSION_SKIP:
                continue
            data[tid] = tool_inst
        return CORE_STATE_VERSION, [data, next(self._id_counter)]

    @classmethod
    def restore_snapshot_new(cls, session, tool_info, version, data):
        try:
            return session.tools
        except AttributeError:
            return cls.__new__(cls)

    def restore_snapshot_init(self, session, tool_info, version, data):
        """Restore state of running tools.

        Overrides chimera.core.session.State default method to restore
        state of all registered running tools.

        Parameters
        ----------
        session : instance of chimera.core.session.Session
            Session for which state is being saved.
            Should match the `session` argument given to `__init__`.
        tool_info : instance of :py:class:`~chimera.core.toolshed.ToolInfo`
        version : any
            Version of state manager that saved the data.
            Used for determining how to parse the `data` argument.
        data : any
            Data saved by state manager during `take_snapshot`.

        """
        self.__init__(session)
        self._tool_instances.update(data[0])
        import itertools
        self._id_counter = itertools.count(data[1])


    def reset_state(self, session):
        """Reset state manager to default state.

        Overrides chimera.core.session.State default method to reset
        to default state.  Since the default state has no running
        tools, all registered tool instances are deleted.

        """
        items = list(self._tool_instances.items())
        for id, tool_inst in items:
            if tool_inst.SESSION_ENDURING:
                continue
            tool_inst.delete()
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
        for tool_inst in ti_list:
            if tool_inst.id is None:
                tool_inst.id = next(self._id_counter)
            self._tool_instances[tool_inst.id] = tool_inst
        session.triggers.activate_trigger(ADD_TOOL_INSTANCE, ti_list)

    def remove(self, ti_list):
        """Deregister running tools with state manager.

        Parameters
        ----------
        ti_list : list of ToolInstance instances
            List of registered running tool instances.

        """
        session = self._session()   # resolve back reference
        for tool_inst in ti_list:
            tid = tool_inst.id
            if tid is None:
                # Not registered in a session
                continue
            tool_inst.id = None
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
        auto_ti = [None] * len(settings.autostart)
        for tool_inst in session.toolshed.tool_info():
            try:
                auto_ti[settings.autostart.index(tool_inst.name)] = tool_inst
            except ValueError:
                continue
        # start them in the same order as given in the setting
        for tool_inst in auto_ti:
            if tool_inst is None:
                continue
            try:
                tool_inst.start(session)
            except ToolshedError as e:
                session.logger.info("Tool \"%s\" failed to start"
                                    % tool_inst.name)
                print("{}".format(e))
