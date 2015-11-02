# vim: set expandtab shiftwidth=4 softtabstop=4:


def cofr(session, method=None, objects=None, pivot=None, coordinate_system=None):
    '''
    Set center of rotation method to "front center" or "fixed".  For fixed can
    specify the pivot point as the center of specified displayed objects,
    or as a 3-tuple of numbers and optionally a model whose coordinate system
    is used for 3-tuples.

    Parameters
    ----------
    method : string
      "front center" or "fixed" specifies how the center of rotation point is defined.
    objects : AtomSpecResults
      Set the method to "fixed" and use the center of the bounding box of these objects
      as the pivot point.
    pivot : 3 floats
      Set the method to "fixed" and used the specified point as center of rotation.
    coordinate_system : Model
      The pivot argument is given in the coordinate system of this model.  If this
      option is not specified then the pivot is in scene coordinates.
    '''
    v = session.main_view
    if not method is None:
        if method == 'frontCenter':
            method = 'front center'
        v.center_of_rotation_method = method
    if not objects is None:
        if objects.empty():
            from ..errors import UserError
            raise UserError('No objects specified.')
        disp = objects.displayed()
        b = disp.bounds()
        if b is None:
            from ..errors import UserError
            raise UserError('No displayed objects specified.')
        v.center_of_rotation = b.center()
    if not pivot is None:
        p = pivot if coordinate_system is None else coordinate_system.scene_position * pivot
        from numpy import array, float32
        v.center_of_rotation = array(p, float32)
    if method is None and objects is None and pivot is None:
        msg = 'Center of rotation: %.5g %.5g %.5g' % tuple(v.center_of_rotation)
        log = session.logger
        log.status(msg)
        log.info(msg)

def uncofr(session):
    '''
    Set center of rotation method to the default "front center" method.
    '''
    v = session.main_view
    v.center_of_rotation_method = 'front center'
        
def register_command(session):
    from . import CmdDesc, register, EnumOf, EmptyArg, ObjectsArg, Or, Float3Arg, ModelArg
    desc = CmdDesc(
        optional=[('method', Or(EnumOf(('front center', 'frontCenter', 'fixed')), EmptyArg)),
                  ('objects', Or(ObjectsArg, EmptyArg)),
                  ('pivot', Float3Arg)],
        keyword=[('coordinate_system', ModelArg)],
        synopsis='set center of rotation method')
    register('cofr', desc, cofr)
    udesc = CmdDesc(synopsis='set center of rotation method to front center')
    register('~cofr', udesc, uncofr)
