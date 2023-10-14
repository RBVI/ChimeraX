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

from chimerax.core.commands import (
    CmdDesc, EnumOf, StringArg, NoArg, BoolArg, plural_form, commas,
    AnnotationError, OpenFileNameArg, ListOf, Or, next_token
)

_bundle_types = EnumOf(["all", "installed", "user", "available"])
_reload_types = EnumOf(["all", "cache", "installed", "available"])


class WheelArg(OpenFileNameArg):
    name = "a wheel file name"

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % WheelArg.name)
        token, text, rest = OpenFileNameArg.parse(text, session)
        import os
        if os.path.splitext(token)[1] != ".whl":
            raise AnnotationError("Expected %s" % WheelArg.name)
        return token, text, rest


class BundleNameArg(StringArg):
    name = "a bundle name"
    # PEP 427, distribution names are alphanumeric with underscores.
    # pypi projct names can have dashes.

    @staticmethod
    def parse(text, session):
        import re
        token, text, rest = next_token(text, convert=True)
        canonical = re.sub(r"[^\w\d.]+", "_", token, re.UNICODE)
        simple = token.replace('-', '_')
        if simple != canonical:
            raise AnnotationError("Invalid bundle name")
        return token, text, rest


def _reSt_to_html(source):
    # from https://wiki.python.org/moin/reStructuredText
    from docutils import core
    parts = core.publish_parts(source=source, writer_name='html')
    return parts['body_pre_docinfo'] + parts['fragment']


def _display_bundles(bi_list, toolshed, logger, use_html=False, full=True):
    def bundle_key(bi):
        prefix = "ChimeraX-"
        if bi.name.startswith(prefix):
            return bi.name[len(prefix):].casefold()
        return bi.name.casefold()
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
        info += "<ul>\n"
        for bi in sorted(bi_list, key=bundle_key):
            name = bi.name
            if full:
                info += "<p>\n"
            info += "<li>\n"
            if full:
                info += "<dt>\n"
            info += "<b>%s</b> (%s): <i>%s</i>\n" % (
                toolshed.bundle_link(name), bi.version, escape(bi.synopsis))
            if full:
                info += "<dd>\n"
                info += "%s: %s<p>" % (
                    plural_form(bi.categories, "Category"),
                    commas(bi.categories, 'and'))
                info += _reSt_to_html(bi.description)
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
            info += "</li>\n"
        info += "</ul>\n"
    else:
        for bi in sorted(bi_list, key=bundle_key):
            name = bi.name
            if name.startswith('ChimeraX-'):
                name = name[len('ChimeraX-'):]
            info += "%s (%s) [%s]: %s\n" % (
                name, bi.version, ', '.join(bi.categories), bi.synopsis)
            if full:
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


def _newest_by_name(bi_list):
    from pkg_resources import parse_version
    bundle_map = {}
    for bi in bi_list:
        try:
            seen = bundle_map[bi.name]
        except KeyError:
            bundle_map[bi.name] = bi
        else:
            if parse_version(bi.version) > parse_version(seen.version):
                bundle_map[bi.name] = bi
    return bundle_map.values()


def toolshed_list(session, bundle_type="installed",
                  full=False, outdated=False, newest=True):
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
            _display_bundles(bi_list, ts, logger, use_html, full)
        else:
            logger.info("No installed bundles found.")
    if bundle_type in ("available", "all"):
        bi_list = ts.bundle_info(logger, installed=False, available=True)
        if bi_list:
            logger.info("List of available bundles:")
            if newest:
                bi_list = _newest_by_name(bi_list)
            _display_bundles(bi_list, ts, logger, use_html, full)
        else:
            logger.info("No available bundles found.")


toolshed_list_desc = CmdDesc(optional=[("bundle_type", _bundle_types),
                                       ("full", NoArg)],
                             keyword=[("outdated", BoolArg),
                                      ("newest", BoolArg)],
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
        kw = {
            "reread_cache": True,
            "rebuild_cache": True,
            "check_remote": False
        }
    elif reload_type == "cache":
        kw = {
            "reread_cache": True,
            "rebuild_cache": False,
            "check_remote": True
        }
    elif reload_type == "available":
        kw = {
            "reread_cache": False,
            "rebuild_cache": False,
            "check_remote": True
        }
    elif reload_type == "all":
        kw = {
            "reread_cache": True,
            "rebuild_cache": True,
            "check_remote": True
        }
    ts.reload(logger, **kw)


toolshed_reload_desc = CmdDesc(optional=[("reload_type", _reload_types)],
                               non_keyword=['reload_type'],
                               synopsis='Refresh cached bundle metadata')


def _bundle_string(bundle_name, version):
    if version is None:
        return bundle_name
    else:
        return "%s (%s)" % (bundle_name, version)


def toolshed_install(session, bundle_names, user_only=True,
                     reinstall=None, version=None, no_deps=None):
    '''
    Install a bundle.

    Parameters
    ----------
    bundle_names : sequence of bundle name or wheel filename
    user_only : bool
      Install for this user only, or install for all users.
    no_deps : bool
      Don't install any dependencies.
    version : string
    '''
    ts = session.toolshed
    logger = session.logger
    bundles = []
    for bundle_name in bundle_names:
        if bundle_name.endswith(".whl"):
            bundles.append(bundle_name)
        elif version == "latest":
            bi = ts.find_bundle(bundle_name, logger, installed=False)
            if bi is None:
                logger.info("no newer version of \"%s\" is available" % bundle_name)
                return
            cur_bi = ts.find_bundle(bundle_name, logger, installed=True)
            if bi.version == cur_bi.version and not reinstall:
                logger.info("latest version of \"%s\" is already installed" % bundle_name)
                return
            bundles.append(bi)
        else:
            bi = ts.find_bundle(bundle_name, logger, installed=True, version=version)
            if bi:
                if not reinstall:
                    logger.error("%s (%s) is already installed" % (bi.name, bi.version))
                    return
            else:
                bi = ts.find_bundle(bundle_name, logger, installed=False, version=version)
                if bi is None:
                    logger.error("%s does not match any bundles"
                                 % _bundle_string(bundle_name, version))
                    return
            bundles.append(bi)
    kw = {
        "session": session,
        "per_user": user_only,
        "no_deps": no_deps,
    }
    if reinstall is not None:
        kw["reinstall"] = reinstall
    ts.install_bundle(bundles, logger, **kw)
    if getattr(session, 'is_gui', False):
        from chimerax import help_viewer
        help_viewer.reload_toolshed_tabs(session)


toolshed_install_desc = CmdDesc(required=[("bundle_names", ListOf(Or(BundleNameArg, WheelArg)))],
                                optional=[("version", StringArg)],
                                keyword=[("user_only", BoolArg),
                                         ("reinstall", BoolArg),
                                         ("no_deps", BoolArg)],
                                hidden=["user_only"],
                                synopsis='Install a bundle')


def toolshed_uninstall(session, bundle_names, force_remove=False):
    '''
    Uninstall an installed bundle.

    Parameters
    ----------
    bundle_names : sequence of bundle names
    force_remove : boolean
    '''
    ts = session.toolshed
    logger = session.logger
    bundles = set()
    for bundle_name in bundle_names:
        bi = ts.find_bundle(bundle_name, logger, installed=True)
        if bi is None:
            logger.error("\"%s\" does not match any bundles" % bundle_name)
            return
        bundles.add(bi)
    ts.uninstall_bundle(bundles, logger, session=session, force_remove=force_remove)
    if getattr(session, 'is_gui', False):
        from chimerax import help_viewer
        help_viewer.reload_toolshed_tabs(session)


toolshed_uninstall_desc = CmdDesc(required=[("bundle_names", ListOf(BundleNameArg))],
                                  keyword=[("force_remove", BoolArg)],
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
        from chimerax.core import toolshed
        if url == 'default':
            ts.remote_url = toolshed.default_toolshed_url()
        elif url == 'preview':
            ts.remote_url = toolshed.preview_toolshed_url()
        else:
            ts.remote_url = url
        logger.info("Toolshed URL set to %s" % ts.remote_url)
        if wait:
            ts.reload_available(logger)
        else:
            ts.async_reload_available(logger)


toolshed_url_desc = CmdDesc(optional=[("url", StringArg)],
                            keyword=[("wait", BoolArg)],
                            synopsis='show or set toolshed url')


def toolshed_cache(session):
    '''
    Show toolshed cache location
    '''
    ts = session.toolshed
    logger = session.logger
    logger.info("Toolshed cache: %s" % ts._cache_dir)


toolshed_cache_desc = CmdDesc(synopsis='show toolshed cache location')


def toolshed_show(session, bundle_name=None):
    from chimerax import help_viewer
    ts = session.toolshed
    if bundle_name is None:
        url = ts.remote_url
    else:
        bi = ts.find_bundle(bundle_name, session.logger, installed=False)
        if bi is None:
            from ..errors import UserError
            raise UserError("Cannot find bundle '%s' in Toolshed" % bundle_name)
        url = session.toolshed.bundle_url(bi.name)
    help_viewer.show_url(session, url)


toolshed_show_desc = CmdDesc(optional=[("bundle_name", StringArg)],
                             synopsis='show the toolshed or bundle in toolshed')


def register_command(logger):
    from chimerax.core.commands import register

    register("toolshed list", toolshed_list_desc, toolshed_list, logger=logger)
    register("toolshed reload", toolshed_reload_desc, toolshed_reload, logger=logger)
    register("toolshed install", toolshed_install_desc, toolshed_install, logger=logger)
    register("toolshed uninstall", toolshed_uninstall_desc, toolshed_uninstall, logger=logger)
    register("toolshed url", toolshed_url_desc, toolshed_url, logger=logger)
    register("toolshed cache", toolshed_cache_desc, toolshed_cache, logger=logger)
    register("toolshed show", toolshed_show_desc, toolshed_show, logger=logger)
