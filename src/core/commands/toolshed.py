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

from . import CmdDesc, EnumOf, StringArg, BoolArg

_bundle_types = EnumOf(["all", "installed", "available"])


def _display_bundles(bi_list, logger):
    def bundle_key(bi):
        return bi.name
    for bi in sorted(bi_list, key=bundle_key):
        logger.info(" %s (%s) [%s]: %s" % (bi.name, bi.version, ', '.join(bi.categories), bi.synopsis))
        for t in bi.tools:
            logger.info("    Tool: %s: %s" % (t.name, t.synopsis))
        for c in bi.commands:
            logger.info("    Command: %s: %s" % (c.name, c.synopsis))
        for f in bi.formats:
            logger.info("    Formats: %s [%s]" % (f.name, f.category))


def ts_list(session, bundle_type="installed"):
    '''List installed bundles in the log.

    Parameters
    ----------
    bundle_type : string
      Types are "installed", "available", or "all"
    '''
    ts = session.toolshed
    logger = session.logger
    if bundle_type == "installed" or bundle_type == "all":
        bi_list = ts.bundle_info(installed=True, available=False)
        if bi_list:
            logger.info("List of installed bundles:")
            _display_bundles(bi_list, logger)
        else:
            logger.info("No installed bundles found.")
    if bundle_type in ("available", "all"):
        bi_list = ts.bundle_info(installed=False, available=True)
        if bi_list:
            logger.info("List of available bundles:")
            _display_bundles(bi_list, logger)
        else:
            logger.info("No available bundles found.")
ts_list_desc = CmdDesc(optional=[("bundle_type", _bundle_types)],
                       non_keyword=['bundle_type'])


def ts_refresh(session, bundle_type="installed"):
    '''
    Check for new bundles or new bundle vesions on server and display
    them in the toolshed window.

    Parameters
    ----------
    bundle_type : string
      Types are "installed", "available", or "all"
    '''
    ts = session.toolshed
    logger = session.logger
    if bundle_type == "installed":
        ts.reload(logger, session=session, rebuild_cache=True, check_remote=False)
    elif bundle_type == "available":
        ts.reload(logger, session=session, rebuild_cache=False, check_remote=True)
    elif bundle_type == "all":
        ts.reload(logger, session=session, rebuild_cache=True, check_remote=True)
ts_refresh_desc = CmdDesc(optional=[("bundle_type", _bundle_types)])


def _bundle_string(bundle_name, version):
    if version is None:
        return bundle_name
    else:
        return "%s (%s)" % (bundle_name, version)


def ts_install(session, bundle_name, user_only=True, version=None):
    '''
    Install a bundle.

    Parameters
    ----------
    bundle_name : string
    user_only : bool
      Install for this user only, or install for all users.
    version : string
    '''
    ts = session.toolshed
    logger = session.logger
    bi = ts.find_bundle(bundle_name, installed=True, version=version)
    if bi:
        logger.error("\"%s\" is already installed" % bundle_name)
        return
    bi = ts.find_bundle(bundle_name, installed=False, version=version)
    if bi is None:
        logger.error("\"%s\" does not match any bundles"
                     % _bundle_string(bundle_name, version))
        return
    ts.install_bundle(bi, logger, not user_only, session=session)
ts_install_desc = CmdDesc(required=[("bundle_name", StringArg)],
                          optional=[("user_only", BoolArg),
                                    ("version", StringArg)])


def ts_remove(session, bundle_name):
    '''
    Remove an installed bundle.

    Parameters
    ----------
    bundle_name : string
    '''
    ts = session.toolshed
    logger = session.logger
    bi = ts.find_bundle(bundle_name, installed=True)
    if bi is None:
        logger.error("\"%s\" does not match any bundles" % bundle_name)
        return
    ts.uninstall_bundle(bi, logger, session=session)
ts_remove_desc = CmdDesc(required=[("bundle_name", StringArg)])


def ts_update(session, bundle_name, version=None):
    '''
    Update a bundle to the latest version.

    Parameters
    ----------
    bundle_name : string
    version : string
    '''
    ts = session.toolshed
    logger = session.logger
    new_bi = ts.find_bundle(bundle_name, installed=False, version=version)
    if new_bi is None:
        logger.error("\"%s\" does not match any bundles"
                     % _bundle_string(bundle_name, version))
        return
    bi = ts.find_bundle(bundle_name, installed=True)
    if bi is None:
        logger.error("\"%s\" does not match any installed bundles" % bundle_name)
        return
    if (version is None and not new_bi.newer_than(bi) or
            new_bi.version == bi.version):
        logger.info("\"%s\" is up to date" % bundle_name)
        return
    ts.install_bundle(new_bi, logger)
ts_update_desc = CmdDesc(required=[("bundle_name", StringArg)],
                         optional=[("version", StringArg)])


#
# Commands that deal with tools
#

def ts_show(session, tool_name, _show=True):
    '''
    Show a tool, or start one if none is running.

    Parameters
    ----------
    tool_name : string
    '''
    if not session.ui.is_gui:
        from chimerax.core.errors import UserError
        raise UserError("Need a GUI to show or hide tools")
    ts = session.toolshed
    bi, tool_name = ts.find_bundle_for_tool(tool_name)
    if bi is None:
        from chimerax.core.errors import UserError
        raise UserError('No installed tool named "%s"' % tool_name)
    tinst = [t for t in session.tools.list() if t.display_name == tool_name]
    for ti in tinst:
        ti.display(_show)
    if _show and len(tinst) == 0:
        bi.start_tool(session, tool_name)
ts_show_desc = CmdDesc(required=[('tool_name', StringArg)],
                       synopsis="Show tool.  Start if necessary")


def ts_hide(session, tool_name):
    '''
    Hide tool.

    Parameters
    ----------
    tool_name : string
    '''
    ts_show(session, tool_name, _show=False)
ts_hide_desc = CmdDesc(required=[('tool_name', StringArg)],
                       synopsis="Hide tool from view")


def register_command(session):
    from . import register, create_alias

    register("toolshed list", ts_list_desc, ts_list)
    register("toolshed refresh", ts_refresh_desc, ts_refresh)
    register("toolshed install", ts_install_desc, ts_install)
    register("toolshed remove", ts_remove_desc, ts_remove)
    # register("toolshed update", ts_update_desc, ts_update)
    register("toolshed show", ts_show_desc, ts_show)
    register("toolshed hide", ts_hide_desc, ts_hide)

    create_alias("ts", "toolshed $*")
