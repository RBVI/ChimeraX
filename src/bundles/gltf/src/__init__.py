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

from .gltf import read_gltf, write_gltf

from chimerax.core.toolshed import BundleAPI

class _gltfBundle(BundleAPI):

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'gltfModel':
            from . import gltf
            return gltf.gltfModel
        return None

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class Info(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    from . import gltf
                    return gltf.read_gltf(session, data, file_name)
        else:
            from chimerax.save_command import SaverInfo
            class Info(SaverInfo):
                def save(self, session, path, models=None, **kw):
                    from . import gltf
                    gltf.write_gltf(session, path, models, **kw)

                @property
                def save_args(self):
                    from chimerax.core.commands import BoolArg, ModelsArg, Float3Arg, \
                        FloatArg, Or
                    return {
                        'center': Or(BoolArg, Float3Arg),
                        'float_colors': BoolArg,
                        'models': ModelsArg,
                        'preserve_transparency': BoolArg,
                        'prune_vertex_colors': BoolArg,
                        'short_vertex_indices': BoolArg,
                        'texture_colors': BoolArg,
                        'metallic_factor': FloatArg,
                        'roughness_factor': FloatArg,
                        'flat_lighting': BoolArg,
                        'backface_culling': BoolArg,
                        'instancing': BoolArg,
                        'size': FloatArg,
                    }
                    
        return Info()


bundle_api = _gltfBundle()
