# vim: set expandtab shiftwidth=4 softtabstop=4:


def info(session, models=None):
    '''
    Report state of models, such as whether they are displayed, color, number of children,
    number of instances...

    Parameters
    ----------
    models : list of models
    '''
    m = session.models
    if models is None:
        models = m.list()
    
    lines = []
    msort = list(models)
    msort.sort(key = lambda m: m.id)
    for m in msort:
        line = '#%s, %s' % (m.id_string(), m.name)
        npos = len(m.positions)
        if npos > 1:
            line += ', %d instances' % npos
        lines.append(line)     
    msg = '%d models\n' % len(models) + '\n'.join(lines)
    session.logger.info(msg)

def register_command(session):
    from . import CmdDesc, register, ModelsArg
    desc = CmdDesc(optional=[('models', ModelsArg)],
                   synopsis='report info about models')
    register('info', desc, info)
