# vim: set expandtab shiftwidth=4 softtabstop=4:


def run(session, text, *, log=True, downgrade_errors=False):
    """execute a textual command

    Parameters
    ----------
    text : string
        The text of the command to execute.
    log : bool
        Print the command text to the reply log.
    downgrade_errors : bool
        True if errors in the command should be logged as informational.
    """

    from . import cli
    from ..errors import UserError
    command = cli.Command(session)
    try:
        command.parse_text(text, final=True)
        return command.execute(log = log)
    except UserError as err:
        if downgrade_errors:
            session.logger.info(str(err))
        else:
            session.logger.error(str(err))


def register_command(session):
    from . import CmdDesc, register, StringArg, BoolArg
    desc = CmdDesc(required=[('text', StringArg)],
                   optional=[('log', BoolArg),
                             ('downgrade_errors', BoolArg),
                         ],
                   synopsis='indirectly run a command')
    register('run', desc, run)
