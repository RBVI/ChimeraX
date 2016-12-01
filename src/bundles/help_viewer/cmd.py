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


def help(session, topic=None, *, option=None, is_query=False, target=None):
    '''Display help

    Parameters
    ----------
    topic : string
        Show documentation for the specified topic.  If no topic is
        specified then the overview is shown.  Topics that are command names
        can be abbreviated.
    is_query : bool
        Instead of showing the documetation, return if it exists.
    '''
    if topic is None:
        if is_query:
            return True
        topic = 'help:user'
    if topic.startswith(('file:', 'http:')):
        if is_query:
            return False
        url = topic
    elif topic.startswith('cxcmd:'):
        from urllib.parse import unquote
        cmd = unquote(topic.split(':', 1)[1])
        from chimerax.cmd_line.tool import CommandLine
        ti = CommandLine.get_singleton(session, create=False)
        if ti:
            ti.cmd_replace(cmd)
            ti.execute()
        else:
            # no command line?!?
            run(session, cmd)
        return
    elif topic.startswith('help:'):
        # Help URLs are rooted at base_dir
        import os
        import sys
        from chimerax import app_data_dir
        base_dir = os.path.join(app_data_dir, 'docs')
        from urllib.parse import urlparse, urlunparse, quote, unquote
        from urllib.request import url2pathname, pathname2url
        (_, _, url_path, _, _, fragment) = urlparse(topic)
        url_path = quote(url_path)
        path = url2pathname(url_path)
        # make sure path is a relative path
        if os.path.isabs(path):
            if sys.platform.startswith('win'):
                path = os.path.relpath(path, os.path.splitdrive(path)[0])
            else:
                path = os.path.relpath(path, '/')
        path = os.path.join(base_dir, path)
        if not os.path.exists(path):
            # TODO: check if http url is within ChimeraX docs
            # TODO: handle missing doc -- redirect to web server
            if is_query:
                return False
            session.logger.error("No help found for '%s'" % topic)
            return
        if is_query:
            return True
        if os.path.isdir(path):
            path += '/index.html'
        url = urlunparse(('file', '', pathname2url(path), '', '', fragment))
    elif '/' in topic:
        import os
        path = os.path.expanduser(topic)
        path = os.path.abspath(path)
        if not os.path.exists(path):
            if is_query:
                return False
            session.logger.error("Help file not found: '%s'" % path)
            return
        from urllib.parse import urlunparse
        from urllib.request import pathname2url
        url = urlunparse(('file', '', pathname2url(path), '', '', ''))
    else:
        cmd_name = topic
        while 1:
            try:
                url = cli.command_url(cmd_name)
            except ValueError:
                session.logger.error("No help found for '%s'" % topic)
                return
            if url:
                if is_query:
                    return True
                return help(session, url, option=option, is_query=is_query)
            alias = cli.expand_alias(cmd_name)
            if not alias:
                break
            #if not is_query:
            #    run(session, "usage %s" % cmd_name, log=False)
            alias_words = alias.split()
            for i in range(len(alias_words)):
                try:
                    cmd_name = ' '.join(alias_words[0:i + 1])
                    cli.command_url(cmd_name)
                except ValueError:
                    cmd_name = ' '.join(alias_words[0:i])
                    break
        if is_query:
            return False
        run(session, "usage %s" % topic, log=False)
        return

    if session.ui.is_gui:
        new = option == 'new_viewer'
        from .tool import HelpUI
        if new:
            target = topic
        help_viewer = HelpUI.get_viewer(session, target)
        help_viewer.show(url)
    else:
        import webbrowser
        webbrowser.open(url)

help_desc = CmdDesc(
    optional=[
        ('option',
         Or(EnumOf(['new_viewer'], abbreviations=False), EmptyArg)),
        ('topic', RestOfLine)
    ],
    non_keyword=('option', 'topic'),
    synopsis='display help'
)
