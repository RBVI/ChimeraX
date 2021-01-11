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

from .save import save_image

from chimerax.core.toolshed import BundleAPI

class _ImageFormatsBundleAPI(BundleAPI):
    
    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class OpenImageInfo(OpenerInfo):
                def open(self, session, path, file_name, **kw):
                    from .open_image import open_image
                    return open_image(session, path, **kw)
                @property
                def open_args(self):
                    from chimerax.core.commands import FloatArg
                    return {
                        'width': FloatArg,
                        'height': FloatArg,
                        'pixel_size': FloatArg,
                    }
            return OpenImageInfo()
        else:
            # convert formal format name (e.g. Portable Network Graphics) to "punchy" name usable
            # by PIL (e.g. PNG)
            PIL_name = session.data_formats[name].nicknames[0].upper()
            from chimerax.save_command import SaverInfo
            class SaveImageInfo(SaverInfo):
                def save(self, session, path, format_name=PIL_name, **kw):
                    from .save import save_image
                    save_image(session, path, format_name, **kw)

                @property
                def save_args(self, _name=PIL_name):
                    from chimerax.core.commands import PositiveIntArg, FloatArg, BoolArg, Bounded, IntArg
                    args = {
                        'height': PositiveIntArg,
                        'pixel_size': FloatArg,
                        'supersample': PositiveIntArg,
                        'transparent_background': BoolArg,
                        'width': PositiveIntArg,
                    }
                    if _name == "JPEG":
                        args['quality'] = Bounded(IntArg, min=0, max=100)
                    return args

                def save_args_widget(self, session, _name=PIL_name):
                    from .gui import SaveOptionsWidget
                    return SaveOptionsWidget(session, _name)

                def save_args_string_from_widget(self, widget):
                    return widget.options_string()

            return SaveImageInfo()

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'ImageSurface':
            from .open_image import ImageSurface
            return ImageSurface
        return None

bundle_api = _ImageFormatsBundleAPI()
