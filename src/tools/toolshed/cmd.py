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
    ts = session.tools
    ti_list = ts.tool_info(installed=installed, available=available)
    logger = session.logger
    logger.info("List of tools:")
    for ti in ti_list:
        s = "  " + ti.display_name + ": " + ti.name
        if ti.installed:
            s += " (installed)"
        else:
            s += " (available)"
        logger.info(s)
ts_list_desc = cli.CmdDesc(optional=[("tool_type", _tool_types)])

def ts_refresh(session, tool_type="installed"):
    ts = session.tools
    logger = session.logger
    if tool_type == "installed":
        logger.info("refresh installed")
    elif tool_type == "available":
        logger.info("refresh available")
    elif tool_type == "all":
        logger.info("refresh all")
ts_refresh_desc = cli.CmdDesc(optional=[("tool_type", _tool_types)])

# TODO: Add more subcommands here
