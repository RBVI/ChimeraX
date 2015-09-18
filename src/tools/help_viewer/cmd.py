# vi: set expandtab shiftwidth=4 softtabstop=4:
from chimera.core.commands import CmdDesc, Or, EnumOf, EmptyArg, RestOfLine, Command


def help(session, set_home=False, topic=None):
    '''Display help

    Parameters
    ----------
    topic : string
        Show documentation for the specified topic.  If no topic is
        specified then the overview is shown.  Topics that are command names
        can be abbreviated.
    '''
    if topic is not None and topic.startswith(('file:', 'http:')):
        url = topic
    else:
        import os
        base_dir = os.path.join(session.app_data_dir, 'docs')

        if topic is None:
            topic = 'user'

        path = os.path.join(base_dir, topic)
        if os.path.exists(path):
            if os.path.isdir(path):
                path += '/index.html'
        else:
            # check if topic matches a command name
            cmd = Command(None)
            cmd.current_text = topic
            cmd._find_command_name(True, no_aliases=True)
            if cmd._ci and cmd.amount_parsed == len(cmd.current_text):
                path = os.path.join(base_dir, 'user', 'commands',
                                    '%s.html' % cmd.current_text)
                if not os.path.exists(path):
                    from chimera.core.commands import run
                    run(session, "usage %s" % topic)
                    return
            else:
                session.logger.error("No help found for '%s'" % topic)
                return
        from urllib.parse import quote
        url = 'file://' + quote(path)

    if session.ui.is_gui:
        from .gui import get_singleton
        help_viewer = get_singleton(session)
        help_viewer.show(url, set_home=set_home)
    else:
        import webbrowser
        webbrowser.open(url)

help_desc = CmdDesc(
    required=[('set_home',
               Or(EnumOf(['sethome'], abbreviations=False), EmptyArg))],
    optional=[('topic', RestOfLine)],
    synopsis='display help'
)
