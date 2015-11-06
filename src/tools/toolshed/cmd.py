# vim: set expandtab ts=4 sw=4:

from chimera.core.commands import EnumOf, CmdDesc, StringArg, BoolArg

_tool_types = EnumOf(["all", "installed", "available"])


def _display_tools(ti_list, logger):
    for ti in ti_list:
        logger.info(" %s (%s %s): %s" % (ti.display_name, ti.name,
                                         ti.version, ti.synopsis))


def ts_list(session, tool_type="installed"):
    '''List installed tools in the log.

    Parameters
    ----------
    tool_type : string
      Types are "installed", "available", or "all"
    '''
    ts = session.toolshed
    logger = session.logger
    if tool_type == "installed" or tool_type == "all":
        ti_list = ts.tool_info(installed=True, available=False)
        if ti_list:
            logger.info("List of installed tools:")
            _display_tools(ti_list, logger)
        else:
            logger.info("No installed tools found.")
    if tool_type == "available" or tool_type == "all":
        ti_list = ts.tool_info(installed=False, available=True)
        if ti_list:
            logger.info("List of available tools:")
            _display_tools(ti_list, logger)
        else:
            logger.info("No available tools found.")
ts_list_desc = CmdDesc(optional=[("tool_type", _tool_types)],
                       non_keyword=['tool_type'])


def ts_refresh(session, tool_type="installed"):
    '''
    Check for new tools or new tool vesions on server and display
    them in the tool shed window.

    Parameters
    ----------
    tool_type : string
      Types are "installed", "available", or "all"
    '''
    ts = session.toolshed
    logger = session.logger
    if tool_type == "installed":
        ts.reload(logger, rebuild_cache=True, check_remote=False)
    elif tool_type == "available":
        ts.reload(logger, rebuild_cache=False, check_remote=True)
    elif tool_type == "all":
        ts.reload(logger, rebuild_cache=True, check_remote=True)
ts_refresh_desc = CmdDesc(optional=[("tool_type", _tool_types)])


def _tool_string(tool_name, version):
    if version is None:
        return tool_name
    else:
        return "%s (%s)" % (tool_name, version)


def ts_install(session, tool_name, user_only=True, version=None):
    '''
    Install a tool.

    Parameters
    ----------
    tool_name : string
    user_only : bool
      Install for this user only, or install for all users.
    version : string
    '''
    ts = session.toolshed
    logger = session.logger
    ti = ts.find_tool(tool_name, installed=True, version=version)
    if ti:
        logger.error("\"%s\" is already installed" % tool_name)
        return
    ti = ts.find_tool(tool_name, installed=False, version=version)
    if ti is None:
        logger.error("\"%s\" does not match any tools"
                     % _tool_string(tool_name, version))
        return
    ts.install_tool(ti, logger, not user_only)
ts_install_desc = CmdDesc(required=[("tool_name", StringArg)],
                          optional=[("user_only", BoolArg),
                                    ("version", StringArg)])


def ts_remove(session, tool_name):
    '''
    Remove an installed tool.

    Parameters
    ----------
    tool_name : string
    '''
    ts = session.toolshed
    logger = session.logger
    ti = ts.find_tool(tool_name, installed=True)
    if ti is None:
        logger.error("\"%s\" does not match any tools" % tool_name)
        return
    ts.uninstall_tool(ti, logger)
ts_remove_desc = CmdDesc(required=[("tool_name", StringArg)])


def ts_start(session, tool_name, *args, **kw):
    '''
    Start a tool.

    Parameters
    ----------
    tool_name : string
    '''
    ts = session.toolshed
    logger = session.logger
    ti = ts.find_tool(tool_name, installed=True)
    if ti is None:
        logger.error("\"%s\" does not match any tools" % tool_name)
        return
    ti.start(session, *args, **kw)
ts_start_desc = CmdDesc(required=[("tool_name", StringArg)])


def ts_update(session, tool_name, version=None):
    '''
    Update a tool to the latest version.

    Parameters
    ----------
    tool_name : string
    version : string
    '''
    ts = session.toolshed
    logger = session.logger
    new_ti = ts.find_tool(tool_name, installed=False, version=version)
    if new_ti is None:
        logger.error("\"%s\" does not match any tools"
                     % _tool_string(tool_name, version))
        return
    ti = ts.find_tool(tool_name, installed=True)
    if ti is None:
        logger.error("\"%s\" does not match any installed tools" % tool_name)
        return
    if (version is None and not new_ti.newer_than(ti)
            or new_ti.version == ti.version):
        logger.info("\"%s\" is up to date" % tool_name)
        return
    ts.install_tool(new_ti, logger)
ts_update_desc = CmdDesc(required=[("tool_name", StringArg)],
                         optional=[("version", StringArg)])


#
# Commands that deal with GUI (singleton)
#


def get_singleton(session, create=False):
    if not session.ui.is_gui:
        return None
    from .gui import ToolshedUI
    running = session.tools.find_by_class(ToolshedUI)
    if len(running) > 1:
        raise RuntimeError("too many toolshed instances running")
    if not running:
        if create:
            tool_info = session.toolshed.find_tool('toolshed')
            return ToolshedUI(session, tool_info)
        else:
            return None
    else:
        return running[0]


def ts_hide(session):
    '''Hide the Tool Shed user interface.'''
    ts = get_singleton(session)
    if ts is not None:
        ts.display(False)
ts_hide_desc = CmdDesc()


def ts_show(session):
    '''Show the Tool Shed user interface.'''
    ts = get_singleton(session, create=True)
    if ts is not None:
        ts.display(True)
ts_show_desc = CmdDesc()

def ts_start(session, tool_name):
    '''
    Start an instance of a tool.

    Parameters
    ----------
    tool_name : string
    '''
    ts = session.toolshed
    tinfo = ts.find_tool(tool_name)
    if tinfo is None:
        from chimera.core.errors import UserError
        raise UserError('No installed tool named "%s"' % tool_name)
    tinfo.start(session)
from chimera.core.commands import StringArg
ts_start_desc = CmdDesc(required = [('tool_name', StringArg)])

def ts_display(session, tool_name, _display = True):
    '''
    Display instances of a tool, or start one if none is running.

    Parameters
    ----------
    tool_name : string
    '''
    ts = session.toolshed
    tinfo = ts.find_tool(tool_name)
    if tinfo is None:
        from chimera.core.errors import UserError
        raise UserError('No installed tool named "%s"' % tool_name)
    tinst = [t for t in session.tools.list() if t.tool_info is tinfo]
    for t in tinst:
        t.display(_display)
    if len(tinst) == 0:
        tinfo.start(session)
from chimera.core.commands import StringArg
ts_display_desc = CmdDesc(required = [('tool_name', StringArg)])

def ts_undisplay(session, tool_name):
    '''
    Undisplay instances of a tool.

    Parameters
    ----------
    tool_name : string
    '''
    ts_display(session, tool_name, _display = False)
ts_undisplay_desc = CmdDesc(required = [('tool_name', StringArg)])
