# vim: set expandtab shiftwidth=4 softtabstop=4:


def hide(session, objects=None, what=None):
    '''Hide specified atoms, bonds or models.

    Parameters
    ----------
    objects : Objects or None
        Atoms, bonds or models to hide. If None then all are hidden.
    what : 'atoms', 'bonds', 'pseudobonds', 'pbonds', 'cartoons', 'ribbons', 'models' or None
        What to hide.  If None then 'atoms' if any atoms specified otherwise 'models'.
    '''
    if objects is None:
        from . import atomspec
        objects = atomspec.all_objects(session)

    if what is None:
        what = 'atoms' if objects.atoms else 'models'

    if what == 'atoms':
        objects.atoms.displays = False
    elif what == 'bonds':
        atoms = objects.atoms
        atoms.inter_bonds.displays = False
    elif what in ('pseudobonds', 'pbonds'):
        from .. import atomic
        pbonds = atomic.interatom_pseudobonds(objects.atoms, session)
        pbonds.displays = False
    elif what == 'cartoons' or what == 'ribbons':
        res = objects.atoms.unique_residues
        res.ribbon_displays = False
    elif what == 'models':
        for m in objects.models:
            m.display = False


def register_command(session):
    from . import CmdDesc, register, ObjectsArg, EnumOf, EmptyArg, Or, create_alias
    what_arg = EnumOf(('atoms', 'bonds', 'pseudobonds', 'pbonds',
                       'cartoons', 'ribbons', 'models'))
    desc = CmdDesc(
        optional=[('objects', Or(ObjectsArg, EmptyArg)),
                  ('what', what_arg)],
        url='help:user/commands/show.html',
        synopsis='hide specified objects')
    register('hide', desc, hide)
    create_alias('~show', 'hide $*')
    create_alias('~display', 'hide $*')
