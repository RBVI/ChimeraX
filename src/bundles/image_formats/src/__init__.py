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

from chimerax.core.toolshed import BundleAPI

class _ImageFormatsBundleAPI(BundleAPI):
    
    @staticmethod
    def run_provider(session, name, mgr):
        from chimerax.save_command import SaverInfo
        class ImageInfo(SaverInfo):
            def save(self, session, path, format_name=name, **kw):
                from .save import save_image
                save_image(session, path, format_name, **kw)

            @property
            def save_args(self, _name=name):
                from chimerax.core.commands import PositiveIntArg, FloatArg, BoolArg, Bounded, IntArg
                args = {
                    'height': PositiveIntArg,
                    'pixel_size': FloatArg,
                    'supersample': PositiveIntArg,
                    'transparent_background': BoolArg,
                    'width': PositiveIntArg,
                }
                if _name == "JPEG image":
                    args['quality'] = Bounded(IntArg, min=0, max=100)
                return args

            def save_args_widget(self, session):
                from .gui import SaveOptionsWidget
                return SaveOptionsWidget(session)

            def save_args_string_from_widget(self, widget):
                return widget.options_string()

        return ImageInfo()

bundle_api = _ImageFormatsBundleAPI()
