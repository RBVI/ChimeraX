# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===


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
    from . import cli
    id_names = [id_str(id) for id in ids]
    info = cli.commas(id_names, ' and')
    noun = cli.plural_form(id_names, 'model')
    session.logger.info("Open %s: %s" % (noun, info))


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(synopsis='list open model ids')
    cli.register('list', desc, list)
