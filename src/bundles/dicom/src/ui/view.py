from chimerax.core.commands import StringArg, CmdDesc, register, BoolArg
from chimerax.core.models import REMOVE_MODELS
from chimerax.map import Volume
from chimerax.nifti import NiftiGrid
from chimerax.nrrd import NRRDGrid
from ..dicom import DicomGrid
from .orthoplanes import PlaneViewer, orthoplane_triggers, Axis

medical_types = [DicomGrid, NiftiGrid, NRRDGrid]

_2x2OrthoLayout = None
_orthoplane_top_left = None
_orthoplane_bottom_left = None
_orthoplane_bottom_right = None
_currentLayout = "default"

from Qt.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QLabel

def dicom_view(session, arg, force = False):
    global _2x2OrthoLayout, _currentLayout, _orthoplane_top_left, _orthoplane_bottom_left, _orthoplane_bottom_right
    # TODO: Enable for NIfTI and NRRD as well
    open_volumes = [v for v in session.models if type(v) is Volume]
    medical_volumes = [m for m in open_volumes if type(m.data) in medical_types]
    if not force and not medical_volumes:
        session.logger.error("No medical images open")
        return
    if arg == "default" and _currentLayout != "default":
        if _2x2OrthoLayout:
            session.ui.main_window.restore_default_main_view()
            _orthoplane_top_left.delete()
            _orthoplane_bottom_left.delete()
            _orthoplane_bottom_right.delete()
            del _orthoplane_top_left
            del _orthoplane_bottom_left
            del _orthoplane_bottom_right
            del _2x2OrthoLayout
            _orthoplane_top_left = None
            _orthoplane_bottom_left = None
            _orthoplane_bottom_right = None
            _2x2OrthoLayout = None
            _currentLayout = "default"
        for trigger in orthoplane_triggers:
            session.triggers.delete_trigger(trigger)
    elif arg == "orthoplanes" and _currentLayout != "orthoplanes":
        if _2x2OrthoLayout is None:
            for trigger in orthoplane_triggers:
                session.triggers.add_trigger(trigger)

            newCentralWidget = QWidget()
            newMWLayout = QGridLayout()

            newCentralWidget.setLayout(newMWLayout)

            _orthoplane_top_left = PlaneViewer(session, Axis.AXIAL, parent=newCentralWidget)
            _orthoplane_bottom_left = PlaneViewer(session, Axis.CORONAL, parent=newCentralWidget)
            _orthoplane_bottom_right = PlaneViewer(session, Axis.SAGGITAL, parent=newCentralWidget)

            newMWLayout.addWidget(_orthoplane_top_left, 0, 0)
            newMWLayout.addWidget(session.ui.main_window.graphicsArea(), 0, 1)
            newMWLayout.addWidget(_orthoplane_bottom_left, 1, 0)
            newMWLayout.addWidget(_orthoplane_bottom_right, 1, 1)

            newMWLayout.setContentsMargins(0, 0, 0, 0)
            newMWLayout.setSpacing(0)

            _2x2OrthoLayout = newCentralWidget
            session.ui.main_window.main_view = _2x2OrthoLayout
        _currentLayout = "orthoplanes"

def _reset_view(session):
    pass

def _check_rapid_access(*args):
    global _currentLayout
    if _currentLayout != "default":
        session = args[1][0].session
        dicom_view(session, "default", force=True)

dicom_view_desc = CmdDesc(
    required = [("arg", StringArg)],
    optional = [("force", BoolArg)],
    synopsis = "Set the view window to a grid of orthoplanes or back to the default"
)

def register_cmds(logger):
    register("dicom view", dicom_view_desc, dicom_view, logger=logger)
    logger.session.triggers.add_handler(REMOVE_MODELS, _check_rapid_access)
