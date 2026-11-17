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

class _PhenixBundle(BundleAPI):

    @staticmethod
    def get_class(class_name):
        from . import tool
        return getattr(tool, class_name)

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(logger)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        from chimerax.open_command import OpenerInfo
        class ProbeInfo(OpenerInfo):
            def open(self, session, data, file_name, *, structure_model=None, **kw):
                if structure_model is None:
                    from chimerax.core.errors import UserError, CancelOperation
                    from chimerax.atomic import Structure
                    structures = [s for s in session.models if isinstance(s, Structure)]
                    if len(structures) == 0:
                        raise UserError("No atomic models open to read the probe dots into!")
                    elif len(structures) == 1:
                        structure_model = structures[0]
                    else:
                        if session.ui.is_gui and not session.in_script:
                            from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QDialog, \
                                QDialogButtonBox as qbbox
                            class GetStructureDialog(QDialog):
                                def __init__(self, session):
                                    super().__init__()
                                    self.setWindowTitle("Choose Structure for Probe Dots")
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
                            raise UserError("Must specify an atomic model to read the probe dots into")
                from .validation import read_probe_dots
                read_probe_dots(session, data, file_name, structure_model, **kw)
                return [], "Read %d Probe annotations for %d atom pairs" % (sum([len(ld)
                    for p2 in structure_model.probe_dot_data.values() for ld in p2.values()]) / 2,
                    len(structure_model.probe_dot_data) / 2)

            @property
            def open_args(self):
                from chimerax.atomic import StructureArg
                return { 'structure_model': StructureArg }
        return ProbeInfo()

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == 'Water Placement':
            from .tool import LaunchDouseTool
            return LaunchDouseTool(session, tool_name)
        if tool_name == 'Local EM Fitting':
            from .tool import LaunchEmplaceLocalTool
            return LaunchEmplaceLocalTool(session, tool_name)
        if tool_name == 'Fit Loops':
            from .tool import LaunchFitLoopsTool
            return LaunchFitLoopsTool(session, tool_name)
        if tool_name == 'Fit Ligand':
            from .tool import LaunchLigandFitTool
            return LaunchLigandFitTool(session, tool_name)
        if tool_name == 'AlphaFold2 Barbed Wire':
            from .tool import LaunchBarbedWireAnalysisTool
            return LaunchBarbedWireAnalysisTool(session, tool_name)
        if tool_name == 'Find Reference Chain':
            from .tool import LaunchFindReferenceTool
            return LaunchFindReferenceTool(session, tool_name)

bundle_api = _PhenixBundle()
