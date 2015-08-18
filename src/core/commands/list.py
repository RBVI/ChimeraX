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
    info = "Open models: "
    if len(models) > 1:
        info += ", ".join(id_str(id) for id in ids[:-1]) + " and"
    info += " %s" % id_str(ids[-1])
    session.logger.info(info)

def register_command(session):
    from . import cli
    desc = cli.CmdDesc(synopsis='list open model ids')
    cli.register('list', desc, list)
