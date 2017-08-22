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

def texture(session, models, image_file = None, coords = None):
    '''
    Color surfaces using images read from files and specify texture coordinates
    that define how to map the image onto the surface.

    This is an experimental command.

    Parameters
    ----------
    models : list of Model
        Act on these models and their child drawings.
    image_file : string
        Path to image file, e.g. PNG or JPEG image.
    coords : "sphere"
        Defines how to map the image axes onto the surface.  "sphere" assigns
        texture coordinates to each surface vertex based on longitude and lattitude.
        The image x axis is longitude (texture coordinate u), and image y axis is
        lattitude (texture coordinate v).
    '''
    
    rgba = None if image_file is None or image_file == 'none' else image_file_as_rgba(image_file)
    dlist = drawings(models)
    for d in dlist:
        if coords == 'sphere':
            set_sphere_texture_coordinates(d)
        if image_file == 'none':
            d.texture = None
        elif rgba is not None:
            from chimerax.core.graphics import Texture
            if d.texture is not None:
                # TODO: Need a better mechanism to make sure discarded textures are released.
                session.main_view.render.make_current()
                d.texture.delete_texture()
            d.texture = Texture(rgba)
            d.opaque_texture = True

    msg = 'Textured %d drawings: %s' % (len(dlist), ', '.join(d.name for d in dlist))
    session.logger.info(msg)
    
def drawings(models):
    dlist = []
    for m in models:
        for d in m.all_drawings():
            if not d.triangles is None and len(d.triangles) > 0:
                dlist.append(d)
    return dlist

def image_file_as_rgba(path):
    from PyQt5.QtGui import QImage
    qi = QImage(path)
    from chimerax.core.graphics import qimage_to_numpy
    rgba = qimage_to_numpy(qi)
    return rgba

def set_sphere_texture_coordinates(drawing):
    d = drawing
    v = d.vertices
    nv = len(v)
    from numpy import arctan2, sqrt, empty, float32
    uv = empty((nv,2), float32)
    x,y,z = v[:,0], v[:,1], v[:,2]
    from math import pi
    uv[:,0] = arctan2(y, x)	# Phi, -pi to pi
    uv[:,0] /= 2*pi
    uv[:,0] += 0.5
    dxy = sqrt(x*x + y*y)
    uv[:,1] = arctan2(dxy, z)	# Psi, 0 to pi
    uv[:,1] /= pi
    d.texture_coordinates = uv

def register_texture_command(logger):
    from chimerax.core.commands import CmdDesc, register, ModelsArg, StringArg, EnumOf
    desc = CmdDesc(
        required = [('models', ModelsArg)],
        keyword = [('image_file', StringArg),
                   ('coords', EnumOf(('sphere',)))],
        synopsis = 'Color surfaces with images from files'
    )
    register('texture', desc, texture, logger=logger)
