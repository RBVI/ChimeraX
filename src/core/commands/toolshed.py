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

from . import CmdDesc, EnumOf, StringArg, BoolArg, plural_form

_bundle_types = EnumOf(["all", "installed", "available"])


def _display_bundles(bi_list, logger, use_html=False):
    def bundle_key(bi):
        return bi.name
    info = ""
    if use_html:
        from html import escape
        info = """
<style>
table.bundle {
    border-collapse: collapse;
    border-spacing: 2px;
}
th.bundle {
    font-style: italic;
    text-align: left;
}
</style>
        """
        info += "<dl>\n"
        for bi in sorted(bi_list, key=bundle_key):
            info += "<dt><b>%s</b> (%s) [%s]: <i>%s</i>\n" % (
                bi.name, bi.version, ', '.join(bi.categories), escape(bi.synopsis))
            info += "<dd>\n"
            # TODO: convert description's rst text to HTML
            info += escape(bi.description).replace('\n\n', '<p>\n')
            if bi.tools or bi.commands or bi.formats:
                info += "<table class='bundle' border='1'>\n"
            if bi.tools:
                info += "<tr><th class='bundle' colspan='3'>%s:</th></tr>\n" % plural_form(bi.tools, "Tool")
            for t in bi.tools:
                info += "<tr><td><b>%s</b></td> <td colspan='2'><i>%s</i></td></tr>\n" % (t.name, escape(t.synopsis))
            if bi.commands:
                info += "<tr><th class='bundle' colspan='3'>%s:</th></tr>\n" % plural_form(bi.commands, "Command")
            for c in bi.commands:
                info += "<tr><td><b>%s</b></td> <td colspan='2'><i>%s</i></td></tr>\n" % (c.name, escape(c.synopsis))
            if bi.selectors:
                info += "<tr><th class='bundle' colspan='3'>%s:</th></tr>\n" % plural_form(bi.selectors, "Selector")
            for s in bi.selectors:
                info += "<tr><td><b>%s</b></td> <td colspan='2'><i>%s</i></td></tr>\n" % (s.name, escape(s.synopsis))
            if bi.formats:
                info += "<tr><th class='bundle' colspan='3'>%s:</th></tr>\n" % plural_form(bi.formats, "Format")
            for f in bi.formats:
                can_open = ' open' if f.has_open else ''
                can_save = ' save' if f.has_save else ''
                info += "<tr><td><b>%s</b></td> <td><i>%s</i></td><td>%s%s</td></tr>\n" % (
                    f.name, f.category, can_open, can_save)
            if bi.tools or bi.commands or bi.formats:
                info += "</table>\n"
        info += "</dl>\n"
    else:
        for bi in sorted(bi_list, key=bundle_key):
            info += "%s (%s) [%s]: %s\n" % (
                bi.name, bi.version, ', '.join(bi.categories), bi.synopsis)
            if bi.tools:
                info += "   %s:\n" % plural_form(bi.tools, "Tool")
            for t in bi.tools:
                info += "    %s: %s\n" % (t.name, t.synopsis)
            if bi.commands:
                info += "   %s:\n" % plural_form(bi.commands, "Command")
            for c in bi.commands:
                info += "    %s: %s\n" % (c.name, c.synopsis)
            if bi.selectors:
                info += "   %s:\n" % plural_form(bi.selectors, "Selector")
            for s in bi.selectors:
                info += "    %s: %s\n" % (s.name, s.synopsis)
            if bi.formats:
                info += "   %s:\n" % plural_form(bi.formats, "Format")
            for f in bi.formats:
                can_open = ' open' if f.has_open else ''
                can_save = ' save' if f.has_save else ''
                info += "    %s [%s]%s%s\n" % (f.name, f.category, can_open,
                                               can_save)
    logger.info(info, is_html=use_html)


def ts_list(session, bundle_type="installed"):
    '''List installed bundles in the log.

    Parameters
    ----------
    bundle_type : string
      Types are "installed", "available", or "all"
    '''
    ts = session.toolshed
    logger = session.logger
    use_html = session.ui.is_gui
    if bundle_type == "installed" or bundle_type == "all":
        bi_list = ts.bundle_info(installed=True, available=False)
        if bi_list:
            logger.info("List of installed bundles:")
            _display_bundles(bi_list, logger, use_html)
        else:
            logger.info("No installed bundles found.")
    if bundle_type in ("available", "all"):
        bi_list = ts.bundle_info(installed=False, available=True)
        if bi_list:
            logger.info("List of available bundles:")
            _display_bundles(bi_list, logger, use_html)
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
