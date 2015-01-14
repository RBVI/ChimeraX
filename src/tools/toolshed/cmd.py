# vim: set expandtab ts=4 sw=4:

from chimera.core import cli

_tool_types = cli.EnumOf([ "installed", "available", "all" ])

def ts_list(session, tool_type="installed"):
    if tool_type == "installed":
        installed = True
        available = False
    elif tool_type == "available":
        installed = False
        available = True
    elif tool_type == "all":
        installed = True
        available = True
    ts = session.toolshed
    ti_list = ts.tool_info(installed=installed, available=available)
    logger = session.logger
    logger.info("List of tools:")
    for ti in ti_list:
        logger.info(" %s" % str(ti))
ts_list_desc = cli.CmdDesc(optional=[("tool_type", _tool_types)])

def ts_refresh(session, tool_type="installed"):
    ts = session.toolshed
    logger = session.logger
    if tool_type == "installed":
        ts.reload(logger, rebuild_cache=True, check_remote=False)
    elif tool_type == "available":
        ts.reload(logger, rebuild_cache=False, check_remote=True)
    elif tool_type == "all":
        ts.reload(logger, rebuild_cache=True, check_remote=True)
ts_refresh_desc = cli.CmdDesc(optional=[("tool_type", _tool_types)])

def ts_install(session, tool_name, user_only=True):
    ts = session.toolshed
    logger = session.logger
    ti = ts.find_tool(tool_name)
    if ti is None:
        logger.error("\"%s\" does not match any tools")
        return
    if ti.installed:
        logger.error("\"%s\" is already installed")
        return
    ts.install_tool(self, ti, logger, not user_only)
ts_install_desc = cli.CmdDesc(required=[("tool_name", cli.StringArg)],
                                optional=[("user_only", cli.BoolArg)])

def ts_remove(session, tool_name):
    ts = session.toolshed
    logger = session.logger
    ti = ts.find_tool(tool_name)
    if ti is None:
        logger.error("\"%s\" does not match any tools")
        return
    if ti.installed:
        logger.error("\"%s\" is not installed")
        return
    ts.uninstall_tool(self, ti, logger, not user_only)
ts_remove_desc = cli.CmdDesc(required=[("tool_name", cli.StringArg)])

# TODO: Add more subcommands here
