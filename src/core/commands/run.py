# vi: set expandtab shiftwidth=4 softtabstop=4:

def run(session, text, downgrade_errors=False):
    """execute a textual command

    Parameters
    ----------
    text : string
        The text of the command to execute.
    downgrade_errors : bool
        True if errors in the command should be logged as informational.
    """
    command = cli.Command(session)
    from ..errors import UserError
    try:
        command.parse_text(text, final=True)
        command.execute()
    except UserError as err:
        if downgrade_errors:
            session.logger.info(str(err))
        else:
            session.logger.error(str(err))

def register_command(session):
    from . import cli
    desc = cli.CmdDesc(required=[('text', cli.StringArg)],
                       optional=[('downgrade_errors', cli.BoolArg)],
                       synopsis='indirectly run a command')
    cli.register('run', desc, run)
