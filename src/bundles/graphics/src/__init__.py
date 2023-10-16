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

# The following are intentionally exported
__all__ = [
    'Drawing', 'Pick',
    'Camera', 'MonoCamera', 'OrthographicCamera',
    'StereoCamera', 'SplitStereoCamera',
    'Mono360Camera', 'Stereo360Camera', 'DomeCamera',
    'CrossFade', 'MotionBlur',
    'Texture', 'Lighting', 'Material',
    'View', 'OpenGLContext',
]

# Make sure _graphics can runtime link shared library libarrays.
import chimerax.arrays

from .drawing import Drawing, Pick, PickedTriangle, PickedTriangles
from .drawing import text_image_rgba, qimage_to_numpy
from .drawing import concatenate_geometry

from .camera import Camera, MonoCamera, OrthographicCamera, StereoCamera, SplitStereoCamera
from .camera360 import Mono360Camera, Stereo360Camera, DomeCamera

from .crossfade import CrossFade, MotionBlur

from .opengl import Texture, Lighting, Material
from .opengl import OffScreenRenderingContext, OpenGLContext
from .opengl import remember_current_opengl_context, restore_current_opengl_context
from .opengl import Render, OpenGLError, OpenGLVersionError

from .view import View
from .clipping import SceneClipPlane, CameraClipPlane, ClipPlane

from chimerax.core.toolshed import BundleAPI
class _GraphicsBundleAPI(BundleAPI):
    _classes = {
        'View': View,
        'MonoCamera': MonoCamera,
        'OrthographicCamera': OrthographicCamera,
        'Lighting': Lighting,
        'Material': Material,
        'ClipPlane': ClipPlane,
        'SceneClipPlane': SceneClipPlane,
        'CameraClipPlane': CameraClipPlane,
        'Drawing': Drawing,
        }
    @staticmethod
    def get_class(class_name):
        return _GraphicsBundleAPI._classes.get(class_name)

bundle_api = _GraphicsBundleAPI()
