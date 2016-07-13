# vim: set expandtab shiftwidth=4 softtabstop=4:


def close(session, models=None):
    '''
    Close models.

    Parameters
    ----------
    models : list of models
        These models and any submodels are closed.  If models is none all models are closed.
    '''
    m = session.models
    if models is None:
        models = m.list()

    # Avoid closing grouping models if not all child models are closed.
    # This is so that "close ~#1.1" does not close grouping model #1.
    hc = have_all_child_models(models)
    cmodels = [m for m in models if m in hc]

    print ('close', ', '.join(m.id_string() for m in cmodels))
    m.close(cmodels)

def have_all_child_models(models):
    '''
    Return a set containing those models in the given models that have all
    child and descendant models in the given models.
    '''
    contains = set()
    mset = set(models)
    for m in models:
        _contains_model_tree(m, mset, contains)
    return contains

def _contains_model_tree(m, mset, contains):
    if not m in mset:
        return

    cmodels = m.child_models()
    for c in cmodels:
        _contains_model_tree(c, mset, contains)

    for c in cmodels:
        if not c in contains:
            return

    contains.add(m)
        
def register_command(session):
    from . import CmdDesc, register, ModelsArg
    desc = CmdDesc(optional=[('models', ModelsArg)],
                   synopsis='close models')
    register('close', desc, close)
