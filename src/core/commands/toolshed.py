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

from . import CmdDesc, EnumOf, StringArg, BoolArg, plural_form, commas

_bundle_types = EnumOf(["all", "installed", "user", "available"])
_reload_types = EnumOf(["all", "cache", "installed", "available"])


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
            name = bi.name
            if name.startswith('ChimeraX-'):
                name = name[len('ChimeraX-'):]
            info += "<dt><b>%s</b> (%s): <i>%s</i>\n" % (
                name, bi.version, escape(bi.synopsis))
            info += "<dd>\n"
            info += "%s: %s<p>" % (
                plural_form(bi.categories, "Category"),
                commas(bi.categories, ' and '))
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
            name = bi.name
            if name.startswith('ChimeraX-'):
                name = name[len('ChimeraX-'):]
            info += "%s (%s) [%s]: %s\n" % (
                name, bi.version, ', '.join(bi.categories), bi.synopsis)
            if bi.tools:
                info += "  %s:\n" % plural_form(bi.tools, "Tool")
            for t in bi.tools:
                info += "    %s: %s\n" % (t.name, t.synopsis)
            if bi.commands:
                info += "  %s:\n" % plural_form(bi.commands, "Command")
            for c in bi.commands:
                info += "    %s: %s\n" % (c.name, c.synopsis)
            if bi.selectors:
                info += "  %s:\n" % plural_form(bi.selectors, "Selector")
            for s in bi.selectors:
                info += "    %s: %s\n" % (s.name, s.synopsis)
            if bi.formats:
                info += "  %s:\n" % plural_form(bi.formats, "Format")
            for f in bi.formats:
                can_open = ' open' if f.has_open else ''
                can_save = ' save' if f.has_save else ''
                info += "    %s [%s]%s%s\n" % (f.name, f.category, can_open,
                                               can_save)
    logger.info(info, is_html=use_html)


def toolshed_list(session, bundle_type="installed", outdated=False):
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
        bi_list = ts.bundle_info(logger, installed=True, available=False)
        if bi_list:
            logger.info("List of installed bundles:")
            _display_bundles(bi_list, logger, use_html)
        else:
            logger.info("No installed bundles found.")
    if bundle_type in ("available", "all"):
        bi_list = ts.bundle_info(logger, installed=False, available=True)
        if bi_list:
            logger.info("List of available bundles:")
            _display_bundles(bi_list, logger, use_html)
        else:
            logger.info("No available bundles found.")
toolshed_list_desc = CmdDesc(optional=[("bundle_type", _bundle_types),
                                       ("outdated", BoolArg),],
                             non_keyword=['bundle_type'],
                             synopsis='List installed bundles')


def toolshed_reload(session, reload_type="installed"):
    '''
    Rebuild the bundle metadata cache using information from
    currently installed bundle.
    '''
    ts = session.toolshed
    logger = session.logger
    if reload_type == "installed":
        kw = {"reread_cache":True,
              "rebuild_cache":True,
              "check_remote":False}
    elif reload_type == "cache":
        kw = {"reread_cache":True,
              "rebuild_cache":False,
              "check_remote":True}
    elif reload_type == "available":
        kw = {"reread_cache":False,
              "rebuild_cache":False,
              "check_remote":True}
    elif reload_type == "all":
        kw = {"reread_cache":True,
              "rebuild_cache":True,
              "check_remote":True}
    ts.reload(session.logger, **kw)
toolshed_reload_desc = CmdDesc(optional=[("reload_type", _reload_types),],
                               non_keyword=['reload_type'],
                               synopsis='Refresh cached bundle metadata')


def _bundle_string(bundle_name, version):
    if version is None:
        return bundle_name
    else:
        return "%s (%s)" % (bundle_name, version)


def toolshed_install(session, bundle_name, user_only=True,
                     reinstall=None, version=None):
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
    if bundle_name.endswith(".whl"):
        bi = bundle_name
    elif version == "latest":
        bi = ts.find_bundle(bundle_name, logger, installed=False)
        cur_bi = ts.find_bundle(bundle_name, logger, installed=True)
        if bi.version == cur_bi.version:
            logger.info("latest version of \"%s\" is already installed" % bundle_name)
            return
    else:
        bi = ts.find_bundle(bundle_name, logger, installed=True, version=version)
        if bi:
            logger.error("%s (%s) is already installed" % (bi.name, bi.version))
            return
        bi = ts.find_bundle(bundle_name, logger, installed=False, version=version)
        if bi is None:
            logger.error("%s does not match any bundles"
                         % _bundle_string(bundle_name, version))
            return
    kw = {"session":session,
          "per_user":user_only}
    if reinstall is not None:
        kw["reinstall"] = reinstall
    ts.install_bundle(bi, logger, **kw)
toolshed_install_desc = CmdDesc(required=[("bundle_name", StringArg)],
                          optional=[("user_only", BoolArg),
                                    ("reinstall", BoolArg),
                                    ("version", StringArg)],
                          synopsis='Install a bundle')


def toolshed_uninstall(session, bundle_name):
    '''
    Uninstall an installed bundle.

    Parameters
    ----------
    bundle_name : string
    '''
    ts = session.toolshed
    logger = session.logger
    bi = ts.find_bundle(bundle_name, session.logger, installed=True)
    if bi is None:
        logger.error("\"%s\" does not match any bundles" % bundle_name)
        return
    ts.uninstall_bundle(bi, logger, session=session)
toolshed_uninstall_desc = CmdDesc(required=[("bundle_name", StringArg)],
                                  synopsis='Uninstall a bundle')


def toolshed_url(session, url=None, wait=False):
    '''
    Show or set toolshed URL

    Parameters
    ----------
    url : string
    '''
    ts = session.toolshed
    logger = session.logger
    if url is None:
        logger.info("Toolshed URL: %s" % ts.remote_url)
    else:
        ts.remote_url = url
        logger.info("Toolshed URL set to %s" % ts.remote_url)
        if wait:
            ts.reload_available(logger)
        else:
            ts.async_reload_available(logger)
toolshed_url_desc = CmdDesc(optional=[("url", StringArg),
                                      ("wait", BoolArg)],
                            synopsis='show or set toolshed url')


def toolshed_cache(session):
    '''
    Show toolshed cache location
    '''
    ts = session.toolshed
    logger = session.logger
    logger.info("Toolshed cache: %s" % ts._cache_dir)
toolshed_cache_desc = CmdDesc(synopsis='show toolshed cache location')


#
# Commands that deal with tools
#

def toolshed_show(session, tool_name, _show=True):
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
toolshed_show_desc = CmdDesc(required=[('tool_name', StringArg)],
                       synopsis="Show tool.  Start if necessary")


def toolshed_hide(session, tool_name):
    '''
    Hide tool.

    Parameters
    ----------
    tool_name : string
    '''
    toolshed_show(session, tool_name, _show=False)
toolshed_hide_desc = CmdDesc(required=[('tool_name', StringArg)],
                       synopsis="Hide tool from view")


def register_command(session):
    from . import register

    register("toolshed list", toolshed_list_desc, toolshed_list,
             logger=session.logger)
    register("toolshed reload", toolshed_reload_desc, toolshed_reload,
             logger=session.logger)
    register("toolshed install", toolshed_install_desc, toolshed_install,
             logger=session.logger)
    register("toolshed uninstall", toolshed_uninstall_desc, toolshed_uninstall,
             logger=session.logger)
    register("toolshed url", toolshed_url_desc, toolshed_url,
             logger=session.logger)
    register("toolshed cache", toolshed_cache_desc, toolshed_cache,
             logger=session.logger)
    register("toolshed show", toolshed_show_desc, toolshed_show,
             logger=session.logger)
    register("toolshed hide", toolshed_hide_desc, toolshed_hide,
             logger=session.logger)
