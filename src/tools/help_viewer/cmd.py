# vi: set expandtab shiftwidth=4 softtabstop=4:
from chimera.core.commands import CmdDesc, RestOfLine, Command


def help(session, topic=None):
    '''Display help

    Parameters
    ----------
    topic : string
        Show documentation for the specified topic.  If no topic is
        specified then the overview is shown.  Topics that are command names
        can be abbreviated.
    '''
    if not session.ui.is_gui:
        session.logger.warning("help viewer is only available if GUI is present")
        return

    if topic is not None and topic.startswith(('file:', 'http:')):
        help_file = topic
    else:
        import os
        base_dir = os.path.join(session.app_data_dir, 'docs')

        if topic is None:
            topic = 'user'

        tmp = os.path.join(base_dir, topic)
        if os.path.exists(tmp):
            help_file = tmp
        else:
            # check if topic matches a command name
            cmd = Command(None)
            cmd.current_text = topic
            cmd._find_command_name(True, no_aliases=True)
            if cmd._ci and cmd.amount_parsed == len(cmd.current_text):
                help_file = os.path.join(base_dir, 'user', 'commands',
                                         '%s.html' % cmd.current_text)
            else:
                session.logger.error("No help found for '%s'" % topic)
                return
        if os.path.isdir(help_file):
            help_file += '/index.html'
        from urllib.parse import quote
        help_file = 'file://' + quote(help_file)

    from .gui import get_singleton
    help_viewer = get_singleton(session)
    help_viewer.show(help_file, set_home=False)

help_desc = CmdDesc(
    optional=[('topic', RestOfLine)],
    synopsis='display help'
)
