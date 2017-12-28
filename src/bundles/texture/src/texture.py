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
        Path to image file, e.g. PNG or JPEG image.  If the value is the string "none" then
        any texture is removed from the surface.
    coords : "sphere", "pole", "south"
        Defines how to map the image axes onto the surface.  "sphere" assigns
        texture coordinates to each surface vertex based on longitude and lattitude.
        The image x axis is longitude (texture coordinate u), and image y axis is
        lattitude (texture coordinate v).  "pole" has the north pole at the middle
        of the image and the south pole on a circle touching the image edges.
        "south" is like pole only centered on the south pole.
    '''
    
    rgba = None if image_file is None or image_file == 'none' else image_file_as_rgba(image_file)
    dlist = drawings(models)
    for d in dlist:
        if coords is not None:
            uv = texture_coordinates(d, coords)
            d.texture_coordinates = uv
        if image_file == 'none':
            remove_texture(session, d)
        elif rgba is not None:
            remove_texture(session, d)
            from chimerax.core.graphics import Texture
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
    from os.path import expanduser, isfile
    p = expanduser(path)
    if not isfile(p):
        from chimerax.core.errors import UserError
        raise UserError('texture image file "%s" does not exist' % p)
    from PyQt5.QtGui import QImage
    qi = QImage(p)
    from chimerax.core.graphics import qimage_to_numpy
    rgba = qimage_to_numpy(qi)
    return rgba

def texture_coordinates(drawing, geometry = 'sphere'):
    d = drawing
    v = d.vertices
    nv = len(v)
    from numpy import arctan2, sqrt, empty, float32, pi, sin, cos
    uv = empty((nv,2), float32)
    x,y,z = v[:,0], v[:,1], v[:,2]
    phi = arctan2(y, x)	# Phi, -pi to pi
    dxy = sqrt(x*x + y*y)
    psi = arctan2(dxy, z)	# Psi, 0 to pi
    if geometry == 'sphere':
        uv[:,0] = phi
        uv[:,0] /= 2*pi
        uv[:,0] += 0.5
        uv[:,1] = psi
        uv[:,1] /= pi
    elif geometry == 'pole':
        r = psi/(2*pi)
        uv[:,0] = r*cos(phi) + 0.5
        uv[:,1] = r*sin(phi) + 0.5
    elif geometry == 'south':
        r = (pi-psi)/(2*pi)
        uv[:,0] = r*cos(phi) + 0.5
        uv[:,1] = r*sin(phi) + 0.5
    return uv

def remove_texture(session, drawing):
    d = drawing
    if d.texture is not None:
        # TODO: Need a better mechanism to make sure discarded textures are released.
        session.main_view.render.make_current()
        d.texture.delete_texture()
        d.texture = None

def register_texture_command(logger):
    from chimerax.core.commands import CmdDesc, register, ModelsArg, StringArg, EnumOf
    desc = CmdDesc(
        required = [('models', ModelsArg)],
        keyword = [('image_file', StringArg),
                   ('coords', EnumOf(('sphere','pole','south')))],
        synopsis = 'Color surfaces with images from files'
    )
    register('texture', desc, texture, logger=logger)
