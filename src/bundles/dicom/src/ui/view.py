from chimerax.core.commands import StringArg, CmdDesc, register, BoolArg
from chimerax.core.models import REMOVE_MODELS
from chimerax.map import Volume
from chimerax.nifti import NiftiGrid
from chimerax.nrrd import NRRDGrid
from ..dicom import DicomGrid
from .orthoplanes import PlaneViewer, PlaneViewerManager, orthoplane_triggers, Axis

medical_types = [DicomGrid, NiftiGrid, NRRDGrid]

from Qt.QtWidgets import QWidget, QGridLayout


class FourUpView(QWidget):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self._graphics_area = session.ui.main_window.graphicsArea()

        self._newMWLayout = QGridLayout(parent=self)

        self._orthoplane_manager = PlaneViewerManager(session)

        self._orthoplane_top_left = PlaneViewer(self, self._orthoplane_manager, session, Axis.AXIAL)
        self._orthoplane_bottom_left = PlaneViewer(self, self._orthoplane_manager, session, Axis.CORONAL)
        self._orthoplane_bottom_right = PlaneViewer(self, self._orthoplane_manager, session, Axis.SAGGITAL)

        self._newMWLayout.addWidget(self._orthoplane_top_left.container, 0, 0)
        self._newMWLayout.addWidget(session.ui.main_window.graphicsArea(), 0, 1)
        self._newMWLayout.addWidget(self._orthoplane_bottom_left.container, 1, 0)
        self._newMWLayout.addWidget(self._orthoplane_bottom_right.container, 1, 1)

        self._newMWLayout.setContentsMargins(0, 0, 0, 0)
        self._newMWLayout.setSpacing(0)

        self.setLayout(self._newMWLayout)

    @property
    def graphics_area(self):
        return self._graphics_area


def dicom_view(session, arg, force = False):
    # TODO: Enable for NIfTI and NRRD as well
    open_volumes = [v for v in session.models if type(v) is Volume]
    medical_volumes = [m for m in open_volumes if type(m.data) in medical_types]
    if not force and not medical_volumes:
        session.logger.error("No medical images open")
        return
    if arg == "default" and session.ui.main_window.view_layout != "default":
        session.ui.main_window.restore_default_main_view()
        for trigger in orthoplane_triggers:
            session.triggers.delete_trigger(trigger)
    elif arg == "orthoplanes" and session.ui.main_window.view_layout != "fourup":
        for trigger in orthoplane_triggers:
            session.triggers.add_trigger(trigger)
        session.ui.main_window.main_view = FourUpView(session)
        session.ui.main_window.view_layout = "fourup"

def _check_rapid_access(*args):
    session = args[1][0].session
    if session.ui.main_window.view_layout != "default":
        # TODO: Only do this if all the models have been removed.
        dicom_view(session, "default", force=True)

dicom_view_desc = CmdDesc(
    required = [("arg", StringArg)],
    optional = [("force", BoolArg)],
    synopsis = "Set the view window to a grid of orthoplanes or back to the default"
)

def register_cmds(logger):
    register("dicom view", dicom_view_desc, dicom_view, logger=logger)
    logger.session.triggers.add_handler(REMOVE_MODELS, _check_rapid_access)
