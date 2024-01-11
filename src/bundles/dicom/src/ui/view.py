from chimerax.core.commands import StringArg, CmdDesc, register, BoolArg
from chimerax.core.models import REMOVE_MODELS
from chimerax.map import Volume
from chimerax.nifti import NiftiGrid
from chimerax.nrrd import NRRDGrid
from ..dicom import DicomGrid
from .orthoplanes import PlaneViewer, PlaneViewerManager, Axis
import PyQt6.sip
from Qt.QtCore import Qt
from Qt.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QSplitter, QSizePolicy

medical_types = [DicomGrid, NiftiGrid, NRRDGrid]
views = ["fourup", "sidebyside", "overunder"]

class FourPanelView(QWidget):
    def __init__(self, session, layout: str = "fourup"):
        super().__init__()
        self._view_layout = layout
        self.session = session
        self._graphics_area = session.ui.main_window.graphicsArea()
        self._orthoplane_manager = PlaneViewerManager(session)
        self._axial_orthoplane = PlaneViewer(self, self._orthoplane_manager, session, Axis.AXIAL)
        self._coronal_orthoplane = PlaneViewer(self, self._orthoplane_manager, session, Axis.CORONAL)
        self._sagittal_orthoplane = PlaneViewer(self, self._orthoplane_manager, session, Axis.SAGITTAL)

        if self._view_layout == "sidebyside":
            self._construct_side_by_side()
        elif self._view_layout == "overunder":
            self._construct_over_under()
        else:
            self._construct_fourup()

    def view_layout(self):
        return self._view_layout

    @property
    def graphics_area(self):
        return self._graphics_area

    def set_guideline_visibility(self, visible: bool) -> None:
        if visible:
            self._orthoplane_manager.show_guidelines()
        else:
            self._orthoplane_manager.hide_guidelines()

    def register_segmentation_tool(self, tool):
        self._orthoplane_manager.register_segmentation_tool(tool)

    def clear_segmentation_tool(self):
        self._orthoplane_manager.clear_segmentation_tool()

    def toggle_guidelines(self):
        self._orthoplane_manager.toggle_guidelines()

    def add_segmentation(self, seg):
        self._orthoplane_manager.add_segmentation(seg)

    def remove_segmentation(self, seg):
        self._orthoplane_manager.remove_segmentation(seg)

    def update_displayed_model(self, model):
        self._orthoplane_manager.update_displayed_model(model)

    def redraw_all(self):
        self._orthoplane_manager.redraw_all()

    def segmentation_tool_open(self):
        return self._orthoplane_manager.have_seg_tool

    def segmentation_tool(self):
        return self._orthoplane_manager.segmentation_tool()

    def convert_to_layout(self, layout: str = None):
        if self._view_layout == layout:
            return
        if layout == "fourup":
            self._convert_to_fourup()
        elif layout == "overunder":
            self._convert_to_over_under()
        elif layout == "sidebyside":
            self._convert_to_side_by_side()

    def _clean_fourup(self):
        self._main_layout.removeWidget(self._axial_orthoplane.container)
        self._main_layout.removeWidget(self._coronal_orthoplane.container)
        self._main_layout.removeWidget(self._sagittal_orthoplane.container)
        self._main_layout.removeWidget(self.session.ui.main_window.graphicsArea())

    def _clean_over_under(self):
        self._viewContainerLayout.removeWidget(self._axial_orthoplane.container)
        self._viewContainerLayout.removeWidget(self._coronal_orthoplane.container)
        self._viewContainerLayout.removeWidget(self._sagittal_orthoplane.container)
        self._main_layout.removeWidget(self.session.ui.main_window.graphicsArea())

    def _clean_side_by_side(self):
        self._clean_over_under()

    def _convert_to_over_under(self):
        if self._view_layout == "fourup":
            self._clean_fourup()
        elif self._view_layout == "overunder":
            self._clean_side_by_side()
        PyQt6.sip.delete(self.layout())
        self._construct_over_under()
        self._view_layout = "overunder"

    def _convert_to_side_by_side(self):
        if self._view_layout == "fourup":
            self._clean_fourup()
        elif self._view_layout == "overunder":
            self._clean_over_under()
        PyQt6.sip.delete(self.layout())
        self._construct_side_by_side()
        self._view_layout = "sidebyside"

    def _convert_to_fourup(self):
        self._clean_side_by_side()
        PyQt6.sip.delete(self.layout())
        self._construct_fourup()
        self._view_layout = "fourup"

    def _construct_fourup(self):
        self._main_layout = QGridLayout(parent=self)

        self._main_layout.addWidget(self._axial_orthoplane.container, 0, 0)
        self._main_layout.addWidget(self.session.ui.main_window.graphicsArea(), 0, 1)
        self._main_layout.addWidget(self._coronal_orthoplane.container, 1, 0)
        self._main_layout.addWidget(self._sagittal_orthoplane.container, 1, 1)

        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        self.setLayout(self._main_layout)

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

    def clean_up(self) -> None:
        for orthoplane in [self._axial_orthoplane, self._coronal_orthoplane, self._sagittal_orthoplane]:
            orthoplane.close()
        if self._view_layout == "fourup":
            self._clean_fourup()
        elif self._view_layout == "overunder":
            self._clean_over_under()
        elif self._view_layout == "sidebyside":
            self._clean_side_by_side()