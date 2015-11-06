# vim: set expandtab shiftwidth=4 softtabstop=4:
from chimera.core.commands import CmdDesc, Or, EnumOf, EmptyArg, RestOfLine, Command, run, cli


def help(session, topic=None, *, option=None, is_query=False):
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
    else:
        path = ""
        fragment = ""
        if topic.startswith('help:'):
            import os
            base_dir = os.path.join(session.app_data_dir, 'docs')
            from urllib.parse import urlparse
            from urllib.request import url2pathname
            (_, _, url_path, _, _, fragment) = urlparse(topic)
            path = os.path.join(base_dir, url2pathname(url_path))
            path = os.path.expanduser(path)
            if not os.path.exists(path):
                if is_query:
                    return False
                session.logger.error("No help found for '%s'" % topic)
                return
            if is_query:
                return True
            if os.path.isdir(path):
                path += '/index.html'
        else:
            import os
            base_dir = os.path.join(session.app_data_dir, 'docs')

            # check if topic matches a command name
            cmd = Command(None)
            cmd.current_text = topic
            cmd._find_command_name(True)
            is_alias = False
            if cmd.command_name is not None:
                alias = cli.expand_alias(cmd.command_name)
                while alias:
                    is_alias = True
                    cmd = Command(None)
                    cmd.current_text = alias
                    cmd._find_command_name(True)
                    if cmd.command_name is None:
                        break
                    alias = cli.expand_alias(cmd.command_name)
            if cmd.command_name is None:
                if is_query:
                    return False
                session.logger.error("No help found for '%s'" % topic)
                return
            # handle multi word command names
            #  -- use first word for filename and rest for #fragment
            if ' ' not in cmd.command_name:
                cmd_name = cmd.command_name
                fragment = ""
            else:
                cmd_name, fragment = cmd.command_name.split(maxsplit=1)
            path = os.path.join(base_dir, 'user', 'commands',
                                '%s.html' % cmd_name)
            if is_alias:
                run(session, "usage %s" % topic, log=False)
            if not os.path.exists(path):
                if is_query:
                    return False
                run(session, "usage %s" % cmd_name, log=False)
                return
            if is_query:
                return True
        from urllib.parse import urlunparse
        from urllib.request import pathname2url
        url = urlunparse(('file', '', pathname2url(path), '', '', fragment))

    if session.ui.is_gui:
        from .gui import HelpUI
        help_viewer = HelpUI.get_singleton(session)
        help_viewer.show(url, set_home=option == 'sethome')
    else:
        import webbrowser
        webbrowser.open(url)

help_desc = CmdDesc(
    optional=[
        ('option',
         Or(EnumOf(['sethome'], abbreviations=False), EmptyArg)),
        ('topic', RestOfLine)
    ],
    synopsis='display help'
)
