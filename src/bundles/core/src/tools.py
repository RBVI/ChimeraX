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
from .state import State, StateManager
ADD_TOOL_INSTANCE = 'add tool instance'
REMOVE_TOOL_INSTANCE = 'remove tool instance'
# If any of the *STATE_VERSIONs change, then increase the (maximum) core session
# number in setup.py.in
TOOL_INSTANCE_STATE_VERSION = 2
TOOLS_STATE_VERSION = 1


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

    Attributes
    ----------
    display_name : str
        If a different name is desired (e.g. multi-instance tool) make sure
        to set the attribute before creating the first tool window.
    SESSION_ENDURING : bool, class-level optional
        If True, then tool survives across sessions.
    SESSION_SAVE : bool, class-level optional
        If False, then tool is not saved in sessions.
    help : str
        URL for tool's help
    """

    SESSION_ENDURING = False
    SESSION_SAVE = False
    help = None

    def __init__(self, session, tool_name):
        self._id = None
        import weakref
        self._session = weakref.ref(session)
        self.tool_name = tool_name
        self.display_name = tool_name
        # TODO: track.created(ToolInstance, [self])
        session.tools.add([self])

    def take_snapshot(self, session, flags):
        data = {
            'name': self.display_name,
            'version': TOOL_INSTANCE_STATE_VERSION
        }
        if hasattr(self, 'tool_window'):
            data['shown'] = self.tool_window.shown
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        if data is None:
            return None
        bundle_info = session.toolshed.find_bundle_for_class(cls)
        tool_name = data['name']
        # dropped data['id'] from version 1
        if bundle_info is None:
            print("can't find bundle info", data)
            return
        if 'version' in data and data['version'] not in range(1, TOOL_INSTANCE_STATE_VERSION + 1):
            from chimerax.core.state import RestoreError
            raise RestoreError('unexpected version restoring tool "%s", got %d, expected %s'
                               % (tool_name, data['version'],
                                  ', '.join(str(v) for v in bundle_info.session_versions)))
        if hasattr(cls, 'get_singleton'):
            ti = cls.get_singleton(session)
        else:
            ti = cls(session, tool_name)
        if ti:
            ti.set_state_from_snapshot(session, data)
        return ti

    def set_state_from_snapshot(self, session, data):
        if 'shown' in data:
            self.display(data['shown'])

    def reset_state(self, session):
        pass

    @property
    def session(self):
        """Read-only property for session that contains this tool instance."""
        return self._session()

    @property
    def bundle_info(self):
        return self.session.toolshed.find_bundle_for_class(self.__class__)

    @property
    def tool_info(self):
        for ti in self.bundle_info.tools:
            if ti.name == self.tool_name:
                return ti
        return None

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

    def displayed(self):
        if hasattr(self, 'tool_window'):
            return self.tool_window.shown
        raise NotImplementedError(
            "%s tool has not implemented 'displayed' method" % self.display_name)

    def display(self, b):
        """Show or hide this tool instance in the user interface.

        Parameters
        ----------
        b : boolean
            Boolean value for whether the tool should be shown or hidden.

        """
        if self.session.ui.is_gui:
            self.session.ui.thread_safe(
                lambda s=self, show=b: s.session.ui.set_tool_shown(s, show))

    def display_help(self):
        """Show the help for this tool in the help viewer."""
        from chimerax.core.commands import run
        run(self.session,
            'help %s' % self.help if self.help is not None else "")


def get_singleton(session, tool_class, tool_name, create=True, display=False, **kw):
    if not session.ui.is_gui:
        return None
    running = [t for t in session.tools.find_by_class(tool_class)
               if t.display_name == tool_name]
    if len(running) > 1:
        raise RuntimeError("too many %s instances running" % tool_name)
    if not running:
        if create:
            tinst = tool_class(session, tool_name, **kw)
        else:
            tinst = None
    else:
        tinst = running[0]
    if display and tinst:
        tinst.display(True)
    return tinst


class Tools(StateManager):
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
        self._tool_instances = set()
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
        tlist = []
        for tool_inst in self._tool_instances:
            assert(isinstance(tool_inst, ToolInstance))
            if not tool_inst.SESSION_SAVE:
                continue
            tlist.append(tool_inst)
        data = {
            'tools': tlist,
            'version': TOOLS_STATE_VERSION
        }
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
        # Tools already add themselves to Tools in ToolInstance.__init__()
        return session.tools

    def reset_state(self, session):
        """Reset state manager to default state.

        Overrides chimerax.core.session.State default method to reset
        to default state.  Since the default state has no running
        tools, all registered tool instances are deleted.

        """
        for tool_inst in list(self._tool_instances):
            if tool_inst.SESSION_ENDURING:
                continue
            name = tool_inst.display_name
            tool_inst.delete()
            if tool_inst in self._tool_instances:
                session.logger.warning("Unable to delete tool %r during reset" % name)

    def list(self):
        """Return list of running tools.

        Returns
        -------
        list
            List of ToolInstance instances.

        """
        return list(self._tool_instances)

    def __getitem__(self, i):
        '''index into tools using square brackets (e.g. session.models[i])'''
        return list(self._tool_instances)[i]

    def __iter__(self):
        '''iterator over tools'''
        return iter(self._tool_instances)

    def __len__(self):
        '''number of tools'''
        return len(self._tool_instances)

    def __bool__(self):
        return len(self._tool_instances) != 0

    def add(self, ti_list):
        """Register running tools with state manager.

        Parameters
        ----------
        ti_list : list of ToolInstance instances
            List of newly created running tool instances.

        """
        session = self._session()   # resolve back reference
        self._tool_instances.update(ti_list)
        session.triggers.activate_trigger(ADD_TOOL_INSTANCE, ti_list)

    def remove(self, ti_list):
        """Deregister running tools with state manager.

        Parameters
        ----------
        ti_list : list of ToolInstance instances
            List of registered running tool instances.

        """
        session = self._session()   # resolve back reference
        self._tool_instances -= set(ti_list)
        session.triggers.activate_trigger(REMOVE_TOOL_INSTANCE, ti_list)

    def find_by_class(self, cls):
        """Return a list of tools of the given class.

        All tool instances that match `cls` as defined by `isinstance`
        are returned.

        Parameters
        ----------
        cls : class object
            Class object used to match tool instances.

        """
        return [ti for ti in self._tool_instances if isinstance(ti, cls)]

    def __iter__(self):
        '''iterator over tools'''
        return iter(self._tool_instances)

    def __len__(self):
        '''number of tools'''
        return len(self._tool_instances)

    def autostart(self):
        """Start tools that should start when applications starts up."""
        from .core_settings import settings
        self.start_tools(settings.autostart)

    def start_tools(self, tool_names):
        """Start tools that are specified by name."""
        # Errors are printed instead of logged, since logging goes to
        # the splash screen, and that disappears before the user can
        # see it.
        session = self._session()   # resolve back reference
        start_bi = [None] * len(tool_names)
        for bi in session.toolshed.bundle_info(session.logger):
            for ti in bi.tools:
                try:
                    start_bi[tool_names.index(ti.name)] = bi
                except ValueError:
                    continue
        # start them in the same order as given in the setting
        for tool_name, bi in zip(tool_names, start_bi):
            if bi is None:
                print("Could not find tool \"%s\"" % tool_name)
                continue
            try:
                bi.start_tool(session, tool_name)
            except Exception:
                msg = "Tool \"%s\" failed to start" % tool_name
                session.logger.report_exception(preface=msg)
