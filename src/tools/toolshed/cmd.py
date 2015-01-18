# vim: set expandtab ts=4 sw=4:

from chimera.core import cli

_tool_types = cli.EnumOf([ "all", "installed", "available" ])

def _display_tools(ti_list, logger):
    for ti in ti_list:
        logger.info(" %s (%s %s): %s" % (ti.display_name, ti.name,
                                            ti.version, ti.synopsis))

def ts_list(session, tool_type="installed"):
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

def ts_install(session, tool_name, user_only=True, version=None):
    ts = session.toolshed
    logger = session.logger
    ti = ts.find_tool(tool_name, installed=True, version=version)
    if ti:
        logger.error("\"%s\" is already installed" % tool_name)
        return
    ti = ts.find_tool(tool_name, installed=False, version=version)
    if ti is None:
        if version is None:
            name = tool_name
        else:
            name = "%s (%s)" % (tool_name, version)
        logger.error("\"%s\" does not match any tools" % name)
        return
    ts.install_tool(ti, logger, not user_only)
ts_install_desc = cli.CmdDesc(required=[("tool_name", cli.StringArg)],
                                optional=[("user_only", cli.BoolArg),
                                            ("version", cli.StringArg)])

def ts_remove(session, tool_name):
    ts = session.toolshed
    logger = session.logger
    ti = ts.find_tool(tool_name, installed=True)
    if ti is None:
        logger.error("\"%s\" does not match any tools" % tool_name)
        return
    ts.uninstall_tool(ti, logger)
ts_remove_desc = cli.CmdDesc(required=[("tool_name", cli.StringArg)])

# TODO: Add more subcommands here
