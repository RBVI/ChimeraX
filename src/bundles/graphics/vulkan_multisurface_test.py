#!/usr/bin/env python3

import sys
import time
# Will not work on PyQt6 because it does not expose QApplication.QNativeInterface
from Qt.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget, QFrame
from Qt.QtGui import QWindow, QSurface, QSurfaceFormat
from Qt.QtCore import Qt, QTimer
import numpy as np

from src import _vulkan

class GraphicsWindow(QWindow):
    def __init__(self):
        super().__init__()

class GraphicsWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.left_graphics_window = GraphicsWindow()
        self.right_graphics_window = GraphicsWindow()
        self.left_graphics_container = QWidget.createWindowContainer(
            self.left_graphics_window, self
        )
        self.right_graphics_container = QWidget.createWindowContainer(
            self.right_graphics_window, self
        )

        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.VLine)

        layout = QHBoxLayout(self)
        layout.addWidget(self.left_graphics_container)
        layout.addWidget(self.separator)
        layout.addWidget(self.right_graphics_container)
        layout.setContentsMargins(0, 0, 0, 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL Triangle - Press V for Vulkan, G for OpenGL")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = GraphicsWidget()
        self.setCentralWidget(self.central_widget)

        self._vulkan_context = _vulkan.VulkanContext()

    def exposeEvent(self, event):
        super().exposeEvent(event)
        # Window is exposed (visible) - NOW we can init Vulkan!
        if self.isExposed() and not self.renderer:
            self._initializeVulkan()
            self.renderer = self._vulkan_renderer


        self._vulkan_renderer = _vulkan.VulkanRenderer(self._vulkan_context)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q:
            self.close()

    def render_frame(self):
        if self.renderer and self.isExposed():
            try:
                self.renderer.drawFrame()  # Or clearToBlack() for testing
            except Exception as e:
                print(f"Render error: {e}")
                self.render_timer.stop()

    def _initializeVulkan(self):
        # Get the window handle
        window_id = int(self.winId())
        app = QApplication.instance()
        platform = app.platformName()
        self._vulkan_context = _vulkan.VulkanContext()
        self.renderer = _vulkan.VulkanRenderer(self._vulkan_context)
        # This will not work until Qt 6.10, see PySide bug #2787
        if platform == "wayland":
            pni = app.nativeInterface()
            ws_connection = pni.display()
            self.renderer.setWindowSystemType(_vulkan.SurfaceBackend.Wayland)
        elif platform == "xcb":
            pni = app.nativeInterface()
            ws_connection = pni.connection()
            self.renderer.setWindowSystemType(_vulkan.SurfaceBackend.Xcb)

        # Arbitrarily use the left window to pick the physical device
        left_window_id = self.central_widget.left_graphics_window.winId()
        self.renderer.createSurface(left_window_id)

        # Create and initialize renderer
        self._vulkan_renderer = _vulkan.VulkanRenderer(self._vulkan_context)
        self._vulkan_renderer.initVulkan()
        self._vulkan_renderer.clearToBlack()



def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
