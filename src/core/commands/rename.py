# vim: set expandtab shiftwidth=4 softtabstop=4:

def rename(session, models, name):
    '''
    Rename a model.  Might extend this command in the future to rename chains, residues, atoms...

    Parameters
    ----------
    models : list of models
    name : string
    '''
    for m in models:
        m.name = name

def register_command(session):
    from . import CmdDesc, register, TopModelsArg, StringArg
    desc = CmdDesc(required=[('models', TopModelsArg),
                             ('name', StringArg)],
                   synopsis='rename a model')
    register('rename', desc, rename)
