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

# The following are intentionally exported
__all__ = [
    'Drawing', 'Pick',
    'Camera', 'MonoCamera', 'OrthographicCamera',
    'StereoCamera', 'SplitStereoCamera',
    'Mono360Camera', 'Stereo360Camera',
    'CrossFade', 'MotionBlur',
    'Texture', 'Lighting', 'Material',
    'View', 'OpenGLContext',
]

from .drawing import Drawing, Pick, PickedTriangle, PickedTriangles, qimage_to_numpy

from .camera import Camera, MonoCamera, OrthographicCamera, StereoCamera, SplitStereoCamera
from .camera360 import Mono360Camera, Stereo360Camera

from .crossfade import CrossFade, MotionBlur

from .opengl import Texture, Lighting, Material
from .opengl import OffScreenRenderingContext, OpenGLContext
from .opengl import Render, OpenGLError, OpenGLVersionError

from .view import View, ClipPlane
