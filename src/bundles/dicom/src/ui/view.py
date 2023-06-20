from chimerax.core.commands import StringArg, CmdDesc, register, BoolArg
from chimerax.core.models import REMOVE_MODELS
from chimerax.map import Volume
from chimerax.nifti import NiftiGrid
from chimerax.nrrd import NRRDGrid
from ..dicom import DicomGrid
from .orthoplanes import PlaneViewer, PlaneViewerManager, Axis
from Qt.QtCore import Qt
from Qt.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QSplitter, QSizePolicy

medical_types = [DicomGrid, NiftiGrid, NRRDGrid]
views = ["fourup", "sidebyside", "overunder"]

class FourPanelView(QWidget):
    def __init__(self, session, layout: str = "fourup"):
        super().__init__()
        self.layout = layout
        self.session = session
        self._graphics_area = session.ui.main_window.graphicsArea()
        self._orthoplane_manager = PlaneViewerManager(session)
        self._axial_orthoplane = PlaneViewer(self, self._orthoplane_manager, session, Axis.AXIAL)
        self._coronal_orthoplane = PlaneViewer(self, self._orthoplane_manager, session, Axis.CORONAL)
        self._sagittal_orthoplane = PlaneViewer(self, self._orthoplane_manager, session, Axis.SAGITTAL)

        if self.layout == "sidebyside":
            self._construct_side_by_side()
        elif self.layout == "overunder":
            self._construct_over_under()
        else:
            self._construct_fourup()

    @property
    def graphics_area(self):
        return self._graphics_area

    def register_segmentation_tool(self, tool):
        self._orthoplane_manager.register_segmentation_tool(tool)

    def clear_segmentation_tool(self):
        self._orthoplane_manager.clear_segmentation_tool()

    def toggle_guidelines(self):
        self._orthoplane_manager.toggle_guidelines()

    def add_segmentation(self, seg):
        self._orthoplane_manager.add_segmentation(seg)

    def segmentation_tool_open(self):
        return self._orthoplane_manager.have_seg_tool

    def convert_to_layout(self, layout: str = None):
        if self.layout == layout:
            return
        if layout == "fourup":
            self._convert_to_fourup()
        elif layout == "overunder":
            self._convert_to_over_under()
        elif layout == "sidebyside":
            self._convert_to_side_by_side()

    def _convert_to_over_under(self):
        ...

    def _convert_to_side_by_side(self):
        ...

    def _convert_to_fourup(self):
        ...

    def _construct_fourup(self):
        self._newMWLayout = QGridLayout(parent=self)

        self._newMWLayout.addWidget(self._axial_orthoplane.container, 0, 0)
        self._newMWLayout.addWidget(self.session.ui.main_window.graphicsArea(), 0, 1)
        self._newMWLayout.addWidget(self._coronal_orthoplane.container, 1, 0)
        self._newMWLayout.addWidget(self._sagittal_orthoplane.container, 1, 1)

        self._newMWLayout.setContentsMargins(0, 0, 0, 0)
        self._newMWLayout.setSpacing(0)
        self.setLayout(self._newMWLayout)

    def _construct_side_by_side(self):
        self._main_layout = QVBoxLayout()
        self._main_widget = QSplitter()

        self._viewContainerWidget = QWidget(parent=self)
        self._viewContainerLayout = QVBoxLayout()

        self._viewContainerLayout.addWidget(self._axial_orthoplane.container)
        self._viewContainerLayout.addWidget(self._coronal_orthoplane.container)
        self._viewContainerLayout.addWidget(self._sagittal_orthoplane.container)
        self._viewContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._viewContainerLayout.setSpacing(0)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        self._viewContainerWidget.setLayout(self._viewContainerLayout)

        self._main_layout.addWidget(self._main_widget)
        self._main_widget.addWidget(self.session.ui.main_window.graphicsArea())
        self._main_widget.addWidget(self._viewContainerWidget)
        self._main_widget.setSizes([1000,300])

        self.setLayout(self._main_layout)

    def _construct_over_under(self):
        self._main_layout = QVBoxLayout()
        self._main_widget = QSplitter(Qt.Orientation.Vertical)

        self._viewContainerWidget = QWidget(parent=self)
        self._viewContainerLayout = QHBoxLayout()

        self._viewContainerLayout.addWidget(self._axial_orthoplane.container)
        self._viewContainerLayout.addWidget(self._coronal_orthoplane.container)
        self._viewContainerLayout.addWidget(self._sagittal_orthoplane.container)
        self._viewContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._viewContainerLayout.setSpacing(0)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        self._viewContainerWidget.setLayout(self._viewContainerLayout)

        self._main_layout.addWidget(self._main_widget)
        self._main_widget.addWidget(self.session.ui.main_window.graphicsArea())
        self._main_widget.addWidget(self._viewContainerWidget)
        self._main_widget.setSizes([100,2])
        self.setLayout(self._main_layout)
