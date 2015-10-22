# vi: set expandtab shiftwidth=4 softtabstop=4:


def echo(session, text=''):
    '''Echo text to the log.

    Parameters
    ----------
    text : string
        The text to log.
    '''
    tokens = []
    from . import cli
    while text:
        token, chars, rest = cli.next_token(text)
        tokens.append(token)
        m = cli._whitespace.match(rest)
        rest = rest[m.end():]
        text = rest
    text = ' '.join(tokens)
    session.logger.info(text)


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(optional=[('text', cli.RestOfLine)],
                       non_keyword=['text'],
                       synopsis='show text in log')
    cli.register('echo', desc, echo)
