# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.core.commands import CmdDesc, Or, EnumOf, EmptyArg, RestOfLine, run, cli
from collections import OrderedDict
import os


def help(session, topic=None, *, option=None):
    '''Display help

    Parameters
    ----------
    topic : string
        Show documentation for the specified topic.  If no topic is
        specified then the overview is shown.  Topics that are command names
        can be abbreviated.
    '''
    from chimerax.core import toolshed
    url = None
    html = None
    if topic is None:
        topic = 'help:index.html'
    if topic.startswith('help:'):
        import sys
        from urllib.parse import urlparse, urlunparse, quote
        from urllib.request import url2pathname, pathname2url
        (_, _, url_path, _, _, fragment) = urlparse(topic)
        url_path = quote(url_path)
        help_path = url2pathname(url_path)
        # make sure path is a relative path
        if os.path.isabs(help_path):
            if sys.platform.startswith('win'):
                help_path = os.path.relpath(help_path, os.path.splitdrive(help_path)[0])
            else:
                help_path = os.path.relpath(help_path, '/')
        if not os.path.splitext(help_path)[1]:
            help_path = os.path.join(help_path, 'index.html')
        for hd in toolshed.get_help_directories():
            path = os.path.join(hd, help_path)
            if os.path.exists(path):
                break
        else:
            # TODO? handle missing doc -- redirect to web server
            session.logger.error("No help found for '%s'" % topic)
            return
        if url_path in ('user', 'user/index.html'):
            with open(path) as f:
                new_path = _generate_index(f, session.logger)
                if new_path is not None:
                    path = new_path
        url = urlunparse(('file', '', pathname2url(path), '', '', fragment))
    else:
        cmd_name = topic
        found = False
        while True:
            try:
                url = cli.command_url(cmd_name)
            except ValueError:
                session.logger.error("No help found for '%s'" % topic)
                return
            if url:
                found = True
                break
            alias = cli.expand_alias(cmd_name)
            if not alias:
                break
            alias_words = alias.split()
            for i in range(len(alias_words)):
                try:
                    cmd_name = ' '.join(alias_words[0:i + 1])
                    cli.command_url(cmd_name)
                except ValueError:
                    cmd_name = ' '.join(alias_words[0:i])
                    break
        if not found:
            run(session, "usage %s" % topic, log=False)
            return
    from . import show_url
    show_url(session, url, new_tab=(option == 'newTab'), html=html)


help_desc = CmdDesc(
    optional=[
        ('option',
         Or(EnumOf(['newTab'], abbreviations=False), EmptyArg)),
        ('topic', RestOfLine)
    ],
    non_keyword=('option', 'topic'),
    synopsis='display help'
)


def _generate_index(source, logger):
    # Take contents of source, look for lists of tools and commands,
    # and insert tools and commands from bundles that come with
    # documentation
    from chimerax import app_dirs
    user_dir = os.path.join(app_dirs.user_cache_dir, 'docs', 'user')
    path = os.path.join(user_dir, 'index.html')
    if os.path.exists(path):
        return path
    os.makedirs(user_dir, exist_ok=True)

    from chimerax.core import toolshed
    ts = toolshed.get_toolshed()
    if ts is None:
        return None
    # Look for <div id="foobar">
    import lxml.html
    html = lxml.html.parse(source)
    for node in html.iterfind(".//div[@id]"):
        ident = node.attrib["id"]
        if ident == "clist":
            _update_list(ts, node, 'commands', _update_commands, logger)
        elif ident == "tlist":
            _update_list(ts, node, 'tools', _update_tools, logger)
    data = lxml.html.tostring(html)
    os.makedirs(user_dir, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    return path


def _update_list(toolshed, node, what, callback, logger):
    doc_ul = None    # ul node with documented stuff
    doc = OrderedDict()
    undoc_ul = None  # ul node with undocumented stuff
    undoc = OrderedDict()
    d = None
    errors = []
    for ul in node:
        if ul.tag != 'ul':
            continue
        if doc_ul is None:
            doc_ul = ul
            d = doc
        elif undoc_ul is None:
            undoc_ul = ul
            d = undoc
        else:
            errors.append("unexpected ul tag at line %d", ul.sourceline)
            continue
        # <li><a href="commands/alias.html"><b>alias</b></a>
        # &ndash; define a command alias (shortcut or composite action)</li>
        # <li><b>texture</b> &ndash; map image onto surface</li>
        # <li><a href="tools/basicactions.html"><b>Basic Actions</b></a></li>
        for li in ul:
            if li.tag != 'li':
                errors.append("unexpected node %r on line %d" % (li, li.sourceline))
                continue
            # inspect first child for name
            ab = li[0]
            t = li.text
            valid = t is None or t.strip() == ''  # should not have any text after <li>
            if ab.tag == 'b':
                name = ab.text.strip()
                if d == doc_ul:
                    errors.append("bad <b> on line", ab.sourceline)  # DEBUG
                    valid = False
            elif ab.tag == 'a':
                href = ab.attrib["href"]
                w, name = href.split('/')
                if w != what:
                    errors.append("didn't expect href to be to %s on line %d" % (
                        href, ab.sourceline))
                name = name.split('.')[0]
                if d == undoc_ul:
                    errors.append("bad <a> on line", ab.sourceline)  # DEBUG
                    valid = False
            if not valid:
                errors.append("expected %s tag as first part of <li> on line %d" % (
                    "<a>" if d == doc else "<b>", li.sourceline))
            d[name] = li
    # TODO: only report error in daily (non-production) builds
    if errors:
        from html import escape
        logger.info(
                '<p style="margin-bottom:0">'
                '<div style="font-size:small">'
                f'Developer warnings in user {what} index:'
                '<ul style="margin-top:0">'
                + ''.join('\n<li>%s' % escape(e) for e in errors) + "</ul></div>",
                is_html=True)
    # Currently, don't do anything with undocumented things
    callback(toolshed, doc_ul, doc)


def _update_commands(toolshed, doc_ul, doc):
    from lxml.html import builder as E
    missing = OrderedDict()
    for bi in toolshed.bundle_info(None):
        for cmd in bi.commands:
            words = cmd.name.split(maxsplit=2)
            name = words[0]
            if name in doc or name in missing:
                continue
            synopsis = cmd.synopsis
            if len(words) > 1:
                # synopsis not appropriate for multiword commands
                synopsis = bi.synopsis
            href = bi.get_path(os.path.join("docs", "user", "commands", "%s.html" % name))
            if href:
                missing[name] = ("commands/%s.html" % name, synopsis)
    names = list(doc)
    missing_names = list(missing)
    all_names = names + missing_names
    all_names.sort(key=str.casefold)
    for name in missing_names:
        i = all_names.index(name)
        href, synopsis = missing[name]
        if synopsis:
            synopsis = " &ndash; " + synopsis
        doc_ul.insert(
            i, E.LI(E.A(E.B(name), href=href), synopsis))


def _update_tools(toolshed, doc_ul, doc):
    from lxml.html import builder as E
    missing = OrderedDict()
    for bi in toolshed.bundle_info(None):
        for t in bi.tools:
            name = t.name
            if name in doc:
                continue
            pname = name.replace(' ', '_')
            href = bi.get_path(os.path.join("docs", "user", "tools", "%s.html" % pname))
            if href:
                missing[name] = ("tools/%s.html" % pname, t.synopsis)
    names = list(doc)
    missing_names = list(missing)
    all_names = names + missing_names
    all_names.sort(key=str.casefold)
    for name in missing_names:
        i = all_names.index(name)
        href, synopsis = missing[name]
        if synopsis:
            synopsis = " &ndash; " + synopsis
        doc_ul.insert(
            i, E.LI(E.A(E.B(name), href=href), synopsis))
