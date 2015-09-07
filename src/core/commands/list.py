# vi: set expandtab shiftwidth=4 softtabstop=4:

def list(session):
    '''List the open model ids and names.'''
    models = session.models.list()
    if len(models) == 0:
        session.logger.status("No open models.")
        return

    def id_str(id):
        if isinstance(id, int):
            return str(id)
        return '.'.join(str(x) for x in id)
    ids = [m.id for m in models]
    ids.sort()
    from .cli import commas
    info, suffix = commas([id_str(id) for id in ids], ' and')
    session.logger.info("Open model%s: %s" % (suffix, info))

def register_command(session):
    from . import cli
    desc = cli.CmdDesc(synopsis='list open model ids')
    cli.register('list', desc, list)
