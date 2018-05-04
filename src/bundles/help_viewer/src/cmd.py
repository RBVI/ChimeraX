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


def help(session, topic=None, *, option=None):
    '''Display help

    Parameters
    ----------
    topic : string
        Show documentation for the specified topic.  If no topic is
        specified then the overview is shown.  Topics that are command names
        can be abbreviated.
    '''
    url = None
    html = None
    if topic is None:
        topic = 'help:index.html'
    if topic.startswith('help:'):
        # Help URLs are rooted at base_dir
        import os
        import sys
        from chimerax import app_data_dir
        base_dir = os.path.join(app_data_dir, 'docs')
        from urllib.parse import urlparse, urlunparse, quote
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
            session.logger.error("No help found for '%s'" % topic)
            return
        if os.path.isdir(path):
            path += '/index.html'
        # TODO: if path == 'user/index.html':
        # TODO:     html = merged_user_index()
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
