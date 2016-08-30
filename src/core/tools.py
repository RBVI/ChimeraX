# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

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
and `session.trigger.remove_handler`.
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
    session : instance of chimerax.core.session.Session
        Session in which this tool instance was created.
    bundle_info : a :py:class:`~chimerax.core.toolshed.BundleInfo` instance
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
        Defaults to ``bundle_info.display_name``.
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

    def __init__(self, session, bundle_info, id=None):
        self.id = id
        import weakref
        self._session = weakref.ref(session)
        self.bundle_info = bundle_info
        self.display_name = bundle_info.display_name
        # TODO: track.created(ToolInstance, [self])
        session.tools.add([self])

    def take_snapshot(self, session, flags):
        data = {'id':self.id,
                'name':self.bundle_info.name,
                'version': CORE_STATE_VERSION}
        if hasattr(self, 'tool_window'):
            data['shown'] = self.tool_window.shown
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        bundle_info = session.toolshed.find_bundle(data['name'])
        if 'version' in data and data['version'] not in bundle_info.session_versions:
            from chimerax.core.state import RestoreError
            raise RestoreError('unexpected version restoring tool "%s", got %d, expected %s'
                               % (cls.__name__, data['version'],
                                  ', '.join(str(v) for v in bundle_info.session_versions)))
        bundle_info = session.toolshed.find_bundle(data['name'])
        if bundle_info is None:
            session.logger.info('unable to find tool "%s"' % data['name'])
            return None
        if hasattr(cls, 'get_singleton'):
            ti = cls.get_singleton(session)
        else:
            ti = cls(session, bundle_info)
        if ti:
            ti.set_state_from_snapshot(session, data)
        return ti
                            
    def set_state_from_snapshot(self, session, data):
        self.id = data['id']
        if 'shown' in data:
            self.display(data['shown'])

    def reset_state(self, session):
        pass

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
        from chimerax.core.commands import run
        run(self.session,
            'help %s' % self.help if self.help is not None else "")


def get_singleton(session, tool_class, tool_name, create=True, display=False, **kw):
    if not session.ui.is_gui:
        return None
    running = [t for t in session.tools.find_by_class(tool_class)
               if t.bundle_info.name == tool_name]
    if len(running) > 1:
        raise RuntimeError("too many %s instances running" % tool_name)
    if not running:
        if create:
            bundle_info = session.toolshed.find_bundle(tool_name)
            tinst = tool_class(session, bundle_info, **kw)
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
        session : instance of chimerax.core.session.Session
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

        Overrides chimerax.core.session.State default method to save
        state of all registered running tool instances.

        Parameters
        ----------
        session : instance of chimerax.core.session.Session
            Session for which state is being saved.
            Should match the `session` argument given to `__init__`.
        phase : str
            Take phase.  See `chimerax.core.session` for more details.
        flags : int
            Flags indicating whether snapshot is being taken to
            save scene or session.  See `chimerax.core.session` for
            more details.

        """
        tmap = {}
        for tid, tool_inst in self._tool_instances.items():
            assert(isinstance(tool_inst, ToolInstance))
            if tool_inst.SESSION_SKIP:
                continue
            tmap[tid] = tool_inst
        data = {'tools': tmap,
                'next id': next(self._id_counter),
                'version': CORE_STATE_VERSION}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        """Restore state of running tools.

        Overrides chimerax.core.session.State default method to restore
        state of all registered running tools.

        Parameters
        ----------
        session : instance of chimerax.core.session.Session
            Session for which state is being saved.
            Should match the `session` argument given to `__init__`.
        data : any
            Data saved by state manager during `take_snapshot`.

        """
        t = session.tools
        # Session save can put a None tool instance into file if tool instance
        # has no take_snapshot method and does not use SESSION_SKIP.
        # Filter these None tool instances out.
        tools = {id:ti for id, ti in data['tools'].items()
                 if ti is not None and not ti.SESSION_ENDURING}
        t._tool_instances.update(tools)
        import itertools
        t._id_counter = itertools.count(data['next id'])
        return t


    def reset_state(self, session):
        """Reset state manager to default state.

        Overrides chimerax.core.session.State default method to reset
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
        from .core_settings import settings
        self.start_tools(settings.autostart)

    def start_tools(self, tool_names):
        """Start tools that are specified by name."""
        session = self._session()   # resolve back reference
        from .toolshed import ToolshedError
        from .core_settings import settings
        auto_ti = [None] * len(tool_names)
        for tool_inst in session.toolshed.bundle_info():
            try:
                auto_ti[tool_names.index(tool_inst.name)] = tool_inst
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
