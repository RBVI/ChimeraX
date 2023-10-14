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

from chimerax.core.toolshed import BundleAPI

class StdCommandsAPI(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        # 'initialize' is called by the toolshed on start up
        if session.ui.is_gui:
            from . import coordset_gui
            coordset_gui.register_mousemode(session)

    @staticmethod
    def get_class(class_name):
        if class_name in ['NamedView', 'NamedViews']:
            from . import view
            return getattr(view, class_name)
        if class_name in ['CoordinateSetSlider']:
            from . import coordset_gui
            return getattr(coordset_gui, class_name)

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        tilde = command_name[0] == '~'
        check_name = command_name[1:] if tilde else command_name
        if check_name.startswith("colour"):
            check_name = "color" + check_name[6:]
        name_remapping = {
            'colordef': 'colorname',
            'color delete': 'colorname',
            'color list': 'colorname',
            'color name': 'colorname',
            'color show': 'colorname',
            'lighting model': 'lighting',
            'quit': 'exit',
            'redo': 'undo',
            'select zone': 'zonesel'
        }
        if tilde:
            name_remapping['show'] = name_remapping['display'] = 'hide'
        else:
            name_remapping['display'] = 'show'
        if check_name in name_remapping:
            mod_name = name_remapping[check_name]
        elif check_name.startswith('measure '):
            mod_name = check_name.replace(' ', '_')
        else:
            if ' ' in check_name:
                mod_name, remainder = check_name.split(None, 1)
            else:
                mod_name = check_name
        from importlib import import_module
        mod = import_module(".%s" % mod_name, __package__)
        mod.register_command(logger)

    @staticmethod
    def register_selector(selector_name, logger):
        # 'register_selector' is lazily called when selector is referenced
        from .selectors import register_selectors
        register_selectors(logger)

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class DefattrInfo(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    from .defattr import defattr
                    if 'models' in kw:
                        kw['restriction'] = kw.pop('models')
                    try:
                        defattr(session, data, file_name=file_name, **kw)
                    except SyntaxError as e:
                        from chimerax.core.errors import UserError
                        raise UserError(str(e))
                    return [], ""

                @property
                def open_args(self):
                    from chimerax.core.commands import BoolArg
                    from chimerax.atomic import StructuresArg
                    return {
                        'log': BoolArg,
                        'models': StructuresArg
                    }
        else:
            from chimerax.save_command import SaverInfo
            class DefattrInfo(SaverInfo):
                def save(self, session, path, **kw):
                    from .defattr import write_defattr
                    write_defattr(session, path, **kw)

                @property
                def save_args(self):
                    from chimerax.core.commands import StringArg, BoolArg, EnumOf
                    from chimerax.atomic import StructuresArg
                    from .defattr import match_modes
                    return {
                        'attr_name': StringArg,
                        'model_ids': BoolArg,
                        'match_mode': EnumOf(match_modes),
                        'selected_only': BoolArg,
                        'models': StructuresArg,
                    }

                def save_args_widget(self, session):
                    from .defattr_gui import SaveOptionsWidget
                    return SaveOptionsWidget(session)

                def save_args_string_from_widget(self, widget):
                    return widget.options_string()

        return DefattrInfo()

bundle_api = StdCommandsAPI()

def register_commands(session):
    mod_names = ['alias', 'align', 'angle', 'camera', 'cartoon', 'cd', 'clip', 'close', 'cofr', 'colorname', 'color', 'coordset_gui', 'coordset', 'crossfade', 'defattr_gui', 'defattr', 'delete', 'dssp', 'exit', 'fly', 'getcrd', 'graphics', 'hide', 'lighting', 'material', 'measure_buriedarea', 'measure_center', 'measure_convexity', 'measure_correlation', 'measure_inertia', 'measure_length', 'measure_rotation', 'measure_symmetry', 'move', 'palette', 'perframe', 'pwd', 'rainbow', 'rename', 'ribbon','rmsd', 'rock', 'roll', 'runscript', 'select', 'setattr', 'set', 'show', 'size', 'split', 'stop', 'style', 'sym', 'tile', 'time', 'transparency', 'turn', 'undo', 'usage', 'version', 'view', 'wait', 'windowsize', 'wobble', 'zonesel', 'zoom']

    if not session.ui.is_gui:
        # Remove commands that require Qt to import
        gui_mod_names = ['coordset_gui', 'defattr_gui']
        for mod_name in gui_mod_names:
            mod_names.remove(mod_name)

    # Run command registration for each command.
    from importlib import import_module
    for mod_name in mod_names:
        mod = import_module(".%s" % mod_name, __package__)
        mod.register_command(session.logger)
