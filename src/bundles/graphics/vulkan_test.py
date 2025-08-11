#!/usr/bin/env python3

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QWindow, QSurface
from PySide6.QtCore import Qt, QTimer

from src import _vulkan


class VulkanWindow(QWindow):
    def __init__(self):
        super().__init__()
        self.setSurfaceType(QSurface.SurfaceType.VulkanSurface)
        self.renderer = None
        self.render_timer = None

    def exposeEvent(self, event):
        super().exposeEvent(event)

        # Window is exposed (visible) - NOW we can init Vulkan!
        if self.isExposed() and not self.renderer:
            self.initializeVulkan()

    def initializeVulkan(self):
        print("Window exposed, initializing Vulkan...")

        # Get the window handle
        window_id = int(self.winId())
        print(f"Window ID: {window_id:#x}")

        # Set it globally (for your current architecture)
        _vulkan.set_window_id(window_id)

        # Create and initialize renderer
        self.renderer = _vulkan.VulkanRenderer()
        self.renderer.initVulkan()

        # Try clearing once
        print("Clearing to black...")
        self.renderer.clearToBlack()

        # Set up render loop
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.render_frame)
        self.render_timer.start(16)  # ~60 FPS

    def render_frame(self):
        if self.renderer:
            try:
                self.renderer.drawFrame()  # Or clearToBlack() for testing
            except Exception as e:
                print(f"Render error: {e}")
                self.render_timer.stop()


class VulkanWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.vulkan_window = VulkanWindow()
        self.vulkan_container = QWidget.createWindowContainer(self.vulkan_window, self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.vulkan_container)
        layout.setContentsMargins(0, 0, 0, 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vulkan Canvas Test")
        self.setGeometry(100, 100, 800, 600)

        central_widget = VulkanWidget()
        self.setCentralWidget(central_widget)


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
