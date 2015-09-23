# vi: set expandtab shiftwidth=4 softtabstop=4:
from chimera.core.commands import CmdDesc, Or, EnumOf, EmptyArg, RestOfLine, Command


def help(session, topic=None, *, option=None):
    '''Display help

    Parameters
    ----------
    topic : string
        Show documentation for the specified topic.  If no topic is
        specified then the overview is shown.  Topics that are command names
        can be abbreviated.
    '''
    is_query = option == 'query'
    if topic is None:
        if is_query:
            return True
        topic = 'help:user'
    if topic.startswith(('file:', 'http:')):
        if is_query:
            return False
        url = topic
    else:
        if topic.startswith('help:'):
            import os
            base_dir = os.path.join(session.app_data_dir, 'docs')
            path = os.path.join(base_dir, topic[5:])
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
            cmd._find_command_name(True, no_aliases=True)
            if cmd.amount_parsed != len(cmd.current_text):
                if is_query:
                    return False
                session.logger.error("No help found for '%s'" % topic)
                return
            # handle multi word command names
            #  -- use first word for filename and rest for #fragment
            if ' ' not in cmd.current_text:
                cmd_name = cmd.current_text
                fragment = ""
            else:
                cmd_name, fragment = cmd.current_text.split(None, 1)
            path = os.path.join(base_dir, 'user', 'commands',
                                '%s.html' % cmd_name)
            if not os.path.exists(path):
                if is_query:
                    return False
                from chimera.core.commands import run
                run(session, "usage %s" % topic, log = False)
                return
            if is_query:
                return True
        from urllib.parse import urlunparse
        from urllib.request import pathname2url
        url = urlunparse(('file', '', pathname2url(path), '', '', fragment))

    if session.ui.is_gui:
        from .gui import get_singleton
        help_viewer = get_singleton(session)
        help_viewer.show(url, set_home=option == 'sethome')
    else:
        import webbrowser
        webbrowser.open(url)

help_desc = CmdDesc(
    required=[
        ('option',
         Or(EnumOf(['query', 'sethome'], abbreviations=False), EmptyArg))
    ],
    optional=[('topic', RestOfLine)],
    synopsis='display help'
)
