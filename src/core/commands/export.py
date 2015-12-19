# vim: set expandtab shiftwidth=4 softtabstop=4:


def export(session, filename, **kw):
    try:
        from .. import io
        return io.export(session, filename, **kw)
    except OSError as e:
        from ..errors import UserError
        raise UserError(e)


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(required=[('filename', cli.SaveFileNameArg)],
                       synopsis='export data in format matching filename suffix')
    cli.register('export', desc, export)
