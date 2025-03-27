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

from chimerax.core.toolshed import BundleAPI

class _MDCrdsBundleAPI(BundleAPI):
    
    from chimerax.atomic import StructureArg

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            if name in ("psf", "data"):
                class MDInfo(OpenerInfo):
                    def open(self, session, data, file_name, *, slider=True, format_name=name, **kw):
                        if format_name == "psf":
                            from .read_psf import read_psf as read_topology
                        else:
                            from .read_lammps import read_data as read_topology
                        models, status = read_topology(session, data, file_name, **kw)
                        if slider and session.ui.is_gui:
                            from chimerax.std_commands.coordset import coordset_slider
                            coordset_slider(session, models)
                        return models, status
                    @property
                    def open_args(self):
                        from chimerax.core.commands import BoolArg, OpenFileNameArg, PositiveIntArg
                        return {
                            'auto_style': BoolArg,
                            'coords': OpenFileNameArg,
                            'end': PositiveIntArg,
                            'slider': BoolArg,
                            'start': PositiveIntArg,
                            'step': PositiveIntArg,
                        }
            elif name == "gro":
                class MDInfo(OpenerInfo):
                    def open(self, session, data, file_name, **kw):
                        from .read_gro import read_gro
                        return read_gro(session, data, file_name, **kw)
                    @property
                    def open_args(self):
                        from chimerax.core.commands import BoolArg
                        return { 'auto_style': BoolArg }
            else:
                class MDInfo(OpenerInfo):
                    def open(self, session, data, file_name, *, structure_model=None,
                            md_type=name, replace=True, slider=True, start=1, step=1, end=None, **kw):
                        if structure_model is None:
                            from chimerax.core.errors import UserError, CancelOperation
                            from chimerax.atomic import Structure
                            structures = [s for s in session.models if isinstance(s, Structure)]
                            if len(structures) == 0:
                                raise UserError("No atomic models open to read the coordinates into!")
                            elif len(structures) == 1:
                                structure_model = structures[0]
                            else:
                                if session.ui.is_gui and not session.in_script:
                                    from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QDialog, \
                                        QDialogButtonBox as qbbox
                                    class GetStructureDialog(QDialog):
                                        def __init__(self, session):
                                            super().__init__()
                                            self.setWindowTitle("Choose Structure for Coordinates")
                                            self.setSizeGripEnabled(True)
                                            from chimerax.atomic.widgets import StructureMenuButton
                                            layout = QVBoxLayout()
                                            chooser_layout = QHBoxLayout()
                                            from Qt.QtCore import Qt
                                            chooser_layout.addWidget(QLabel("Structure:"),
                                                alignment=Qt.AlignRight)
                                            self.structure_button = StructureMenuButton(session)
                                            chooser_layout.addWidget(self.structure_button,
                                                alignment=Qt.AlignLeft)
                                            layout.addLayout(chooser_layout)

                                            bbox = qbbox(qbbox.Ok | qbbox.Cancel)
                                            bbox.accepted.connect(self.accept)
                                            bbox.rejected.connect(self.reject)
                                            layout.addWidget(bbox)
                                            self.setLayout(layout)

                                        @property
                                        def model(self):
                                            return self.structure_button.value

                                    dlg = GetStructureDialog(session)
                                    okayed = dlg.exec()
                                    if not okayed or dlg.model is None:
                                        raise CancelOperation("No atomic structure specified")
                                    structure_model = dlg.model
                                else:
                                    raise UserError("Must specify an atomic model to read the coordinates"
                                        " into")
                        from .read_coords import read_coords
                        num_coords = read_coords(session, data, structure_model, md_type,
                            replace=replace, start=start, step=step, end=end)
                        if slider and session.ui.is_gui:
                            from chimerax.std_commands.coordset import coordset_slider
                            coordset_slider(session, [structure_model])
                        if replace:
                            return [], "Replaced existing frames of %s with %d new frames" \
                                % (structure_model, num_coords)
                        return [], "Added %d frames to %s" % (num_coords, structure_model)

                    @property
                    def open_args(self):
                        from chimerax.atomic import StructureArg
                        from chimerax.core.commands import BoolArg, PositiveIntArg
                        return {
                            'end': PositiveIntArg,
                            'replace': BoolArg,
                            'slider': BoolArg,
                            'start': PositiveIntArg,
                            'step': PositiveIntArg,
                            'structure_model': StructureArg,
                        }
            return MDInfo()
        if mgr == session.save_command:
            from chimerax.save_command import SaverInfo
            class MDInfo(SaverInfo):
                def save(self, session, path, *, models=None, **kw):
                    from chimerax import atomic
                    if models is None:
                        models = atomic.all_structures(session)
                    else:
                        models = [m for m in models if isinstance(m, atomic.Structure)]
                    if len(models) == 0:
                        from chimerax.core.errors import UserError
                        raise UserError("Must specify models to write DCD coordinates")
                    # Check that all models have same number of atoms.
                    nas = set(m.num_atoms for m in models)
                    if len(nas) != 1:
                        from chimerax.core.errors import UserError
                        raise UserError("Models have different number of atoms")
                    from .write_coords import write_coords
                    write_coords(session, path, "dcd", models)

                @property
                def save_args(self):
                    from chimerax.core.commands import ModelsArg
                    return { 'models': ModelsArg }

                def save_args_widget(self, session):
                    from .gui import SaveOptionsWidget
                    return SaveOptionsWidget(session)

                def save_args_string_from_widget(self, widget):
                    return widget.options_string()

            return MDInfo()

        # MD plotting manager
        if name == "distance":
            a1, a2 = kw['atoms']
            from chimerax.geometry import distance
            values = {}
            for cs_id in kw['structure'].coordset_ids:
                values[cs_id] = distance(a1.get_coordset_coord(cs_id), a2.get_coordset_coord(cs_id))
            return values
        elif name == "angle":
            from chimerax.geometry import angle
            values = {}
            for cs_id in kw['structure'].coordset_ids:
                values[cs_id] = angle(*[a.get_coordset_coord(cs_id) for a in kw['atoms']])
            return values
        elif name == "torsion":
            from chimerax.geometry import dihedral
            values = {}
            for cs_id in kw['structure'].coordset_ids:
                values[cs_id] = dihedral(*[a.get_coordset_coord(cs_id) for a in kw['atoms']])
            return values
        raise ValueError("Unknown plotting type: %s" % name)


bundle_api = _MDCrdsBundleAPI()
