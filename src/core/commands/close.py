# vi: set expandtab shiftwidth=4 softtabstop=4:

def close(session, model_ids=None):
    '''
    Close models.

    Parameters
    ----------
    model_ids : list of model ids
        These models and any submodels are closed.  If no model ids are specified then all models are closed.
    '''
    m = session.models
    if model_ids is None:
        mlist = m.list()
    else:
        try:
            mlist = sum((m.list(model_id) for model_id in model_ids), [])
        except ValueError as e:
            from ..errors import UserError
            raise UserError(e)
    m.close(mlist)

def register_command(session):
    from . import cli
    desc = cli.CmdDesc(optional=[('model_ids', cli.ListOf(cli.ModelIdArg))],
                          synopsis='close models')
    cli.register('close', desc, close)
