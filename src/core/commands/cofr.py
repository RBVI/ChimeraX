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
    objects : Objects
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
        elif method == 'centerOfView':
            method = 'center of view'
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
        msg = 'Center of rotation: %.5g %.5g %.5g  %s' % (tuple(v.center_of_rotation) + (v.center_of_rotation_method,))
        log = session.logger
        log.status(msg)
        log.info(msg)
        
def register_command(session):
    from . import CmdDesc, register, EnumOf, EmptyArg, ObjectsArg, Or, Float3Arg, ModelArg, create_alias
    desc = CmdDesc(
        optional=[('method', Or(EnumOf(('front center', 'frontCenter', 'fixed', 'centerOfView')), EmptyArg)),
                  ('objects', Or(ObjectsArg, EmptyArg)),
                  ('pivot', Float3Arg)],
        keyword=[('coordinate_system', ModelArg)],
        synopsis='set center of rotation method')
    register('cofr', desc, cofr, logger=session.logger)
    create_alias('~cofr', 'cofr frontCenter')
