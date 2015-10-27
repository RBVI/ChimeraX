# vi: set expandtab shiftwidth=4 softtabstop=4:


def cofr(session, method=None, atoms=None, pivot=None, coordinate_system=None):
    '''
    Set center of rotation method to "front center" or "fixed".  For fixed can
    specify the pivot point as the center of specified atoms, or as a 3-tuple of numbers
    and optionally a model whose coordinate system is used for 3-tuples.

    Parameters
    ----------
    method : string
      "front center" or "fixed" specifies how the center of rotation point is defined.
    atoms : Atoms
      Set the method to "fixed" and use the center of the bounding box of these atoms
      as the pivot point.
    pivot : 3 floats
      Set the method to "fixed" and used the specified point as center of rotation.
    coordinate_system : Model
      The pivot argument is given in the coordinate system of this model.  If this
      option is not specified then the pivot is in scene coordinates.
    '''
    v = session.main_view
    if not method is None:
        v.center_of_rotation_method = method
    if not atoms is None:
        if len(atoms) == 0:
            from ..errors import UserError
            raise UserError('No atoms specified.')
        from .. import geometry
        b = geometry.sphere_bounds(atoms.scene_coords, atoms.radii)
        v.center_of_rotation = b.center()
    if not pivot is None:
        p = pivot if coordinate_system is None else coordinate_system.scene_position * pivot
        from numpy import array, float32
        v.center_of_rotation = array(p, float32)
    if method is None and atoms is None and pivot is None:
        msg = 'Center of rotation: %.5g %.5g %.5g' % tuple(v.center_of_rotation)
        log = session.logger
        log.status(msg)
        log.info(msg)
        
def register_command(session):
    from . import CmdDesc, register, EnumOf, EmptyArg, AtomsArg, Or, Float3Arg, ModelArg
    desc = CmdDesc(
        optional=[('method', Or(EnumOf(('front center', 'fixed')), EmptyArg)),
                  ('atoms', Or(AtomsArg, EmptyArg)),
                  ('pivot', Float3Arg)],
        keyword=[('coordinate_system', ModelArg)],
        synopsis='set center of rotation method')
    register('cofr', desc, cofr)
