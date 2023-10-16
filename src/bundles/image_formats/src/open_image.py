# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def open_image(session, path, width=None, height=None, pixel_size=None):
    '''
    Open an image and display it on a rectangle.
    '''
    from PIL import Image
    image = Image.open(path)
    from numpy import array
    rgba = array(image)
    ih, iw = rgba.shape[:2]

    if width is not None:
        w = width
        if height is None:
            h = (ih * width / iw) if iw > 0 else 0
        else:
            h = height
    elif height is not None:
        w = (iw * height / ih) if ih > 0 else 0
        h = height
    elif pixel_size is not None:
        w,h = iw*pixel_size, ih*pixel_size
    else:
        w,h = iw,ih
        
    # Create a rectangle and texture image onto it.
    from os.path import basename
    name = basename(path)
    model = ImageSurface(session, name, rgba, w, h)

    msg = 'Opened image %s' % name
    
    return [model], msg

from chimerax.core.models import Surface
class ImageSurface(Surface):
    def __init__(self, session, name, rgba, width, height):
        self._image_array = rgba
        self._width = width
        self._height = height
        
        Surface.__init__(self, name, session)

        w2,h2 = 0.5*width, 0.5*height
        from numpy import array, float32, int32
        va = array(((-w2,-h2,0),(w2,-h2,0),(w2,h2,0),(-w2,h2,0)), float32)
        na = array(((0,0,1),(0,0,1),(0,0,1),(0,0,1)), float32)
        ta = array(((0,1,2),(0,2,3)), int32)
        tc = array(((0,1),(1,1),(1,0),(0,0)), float32)
        self.set_geometry(va, na, ta)

        self.use_lighting = False
        self.allow_depth_cue = False
        self.color = (255,255,255,255)
        from chimerax.graphics import Texture
        self.texture = Texture(rgba)
        transparent = (len(rgba.shape) == 3 and rgba.shape[2] == 4 and rgba[:,:,3].min() < 255)
        self.opaque_texture = not transparent
        self.texture_coordinates = tc
        
    def take_snapshot(self, session, flags):
        data = {
            'surface state': Surface.take_snapshot(self, session, flags),
            'width': self._width,
            'height': self._height,
            'image_array': self._image_array,
            'version': 1,
        }
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        m = ImageSurface(session, 'image', data['image_array'], data['width'], data['height'])
        Surface.set_state_from_snapshot(m, session, data['surface state'])
        return m
