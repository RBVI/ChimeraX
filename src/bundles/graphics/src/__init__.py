# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
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

_exports = {
    **dict.fromkeys(('Drawing', 'Pick', 'PickedTriangle', 'PickedTriangles',
        'text_image_rgba', 'qimage_to_numpy', 'concatenate_geometry'), 'drawing'),
    **dict.fromkeys(('Camera', 'MonoCamera', 'OrthographicCamera', 'StereoCamera',
        'SplitStereoCamera'), 'camera'),
    **dict.fromkeys(('Mono360Camera', 'Stereo360Camera', 'DomeCamera'), 'camera360'),
    **dict.fromkeys(('CrossFade', 'MotionBlur'), 'crossfade'),
    **dict.fromkeys(('Texture', 'Lighting', 'Material', 'OffScreenRenderingContext',
        'OpenGLContext', 'remember_current_opengl_context', 'restore_current_opengl_context',
        'Render', 'OpenGLError', 'OpenGLVersionError'), 'opengl'),
    'View': 'view',
    **dict.fromkeys(('SceneClipPlane', 'CameraClipPlane', 'ClipPlane'), 'clipping'),
}


def __getattr__(name):
    try:
        module_name = _exports[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module
    value = getattr(import_module('.' + module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_exports))

from chimerax.core.toolshed import BundleAPI
class _GraphicsBundleAPI(BundleAPI):
    _classes = {'View', 'MonoCamera', 'OrthographicCamera', 'Lighting', 'Material',
                'ClipPlane', 'SceneClipPlane', 'CameraClipPlane', 'Drawing'}
    @staticmethod
    def get_class(class_name):
        if class_name in _GraphicsBundleAPI._classes:
            from importlib import import_module
            return getattr(import_module(__name__), class_name)

bundle_api = _GraphicsBundleAPI()
