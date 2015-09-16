# vi: set expandtab shiftwidth=4 softtabstop=4:


def view(session, atoms=None):
    '''Move camera so the displayed models fill the graphics window.'''
    v = session.main_view
    if atoms is None:
        v.view_all()
    elif len(atoms) == 0:
        from ..errors import UserError
        raise UserError('No atoms specified.')
    else:
        from .. import geometry
        b = geometry.sphere_bounds(atoms.scene_coords, atoms.radii)
        shift = v.camera.view_all(b.center(), b.width())
        v.translate(-shift)


def register_command(session):
    from . import CmdDesc, register, AtomsArg
    desc = CmdDesc(
        optional=[('atoms', AtomsArg)],
        synopsis='reset view so everything is visible in window')
    register('view', desc, view)
