#!/usr/bin/env python3

import sys
import time
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QWindow, QSurface, QSurfaceFormat, QOpenGLContext, QNativeInterface
from PySide6.QtCore import Qt, QTimer

import numpy as np
from OpenGL import GL

from src import _vulkan

opengl_vertex_shader = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;
out vec3 fragColor;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    fragColor = color;
}
"""

opengl_fragment_shader = """
#version 330 core
in vec3 fragColor;
out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
"""

class OpenGLRenderer:
    def __init__(self, surface, context):
        self.surface = surface
        self.context = context
        self.context.makeCurrent(self.surface)

        # Compile shaders
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, opengl_vertex_shader)
        GL.glCompileShader(vs)

        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, opengl_fragment_shader)
        GL.glCompileShader(fs)

        # Link program
        self.program = GL.glCreateProgram()
        GL.glAttachShader(self.program, vs)
        GL.glAttachShader(self.program, fs)
        GL.glLinkProgram(self.program)

        # Triangle data (position + color)
        vertices = np.array([
            # x,    y,     r,   g,   b
            -0.5, -0.5,   1.0, 0.0, 0.0,  # Bottom left - red
             0.5, -0.5,   0.0, 1.0, 0.0,  # Bottom right - green
             0.0,  0.5,   0.0, 0.0, 1.0,  # Top - blue
        ], dtype=np.float32)

        # Create and bind VAO
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Create and bind VBO
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

        ## Set up vertex attributes
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 20, GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 20, GL.ctypes.c_void_p(8))
        GL.glEnableVertexAttribArray(1)


    def drawFrame(self):
        """Placeholder for OpenGL rendering logic"""
        # Clear and draw
        self.context.makeCurrent(self.surface)
        GL.glClearColor(0.2, 0.2, 0.2, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self.program)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

        self.context.swapBuffers(self.surface)

    def cleanup(self):
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(1, [self.vbo])
        GL.glDeleteProgram(self.program)
        self.context.doneCurrent()


class GraphicsWindow(QWindow):
    def __init__(self):
        super().__init__()

        self._opengl_renderer = None
        self._vulkan_context = _vulkan.VulkanContext();
        self._vulkan_renderer = None

        self.renderer = None
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.render_frame)
        self.render_timer.start(16)
        self.initialized = False

    def _initializeOpenGLSurface(self):
        self.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)
        self._opengl_surface_format = QSurfaceFormat()
        self._opengl_surface_format.setVersion(3,3)
        self._opengl_surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self._opengl_surface_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        self._context = QOpenGLContext()
        self._context.setFormat(self._opengl_surface_format)
        self._context.create()


    def switch_to_vulkan(self):
        self.render_timer.stop()
        time.sleep(0.1)  # give time for the last frame to be drawn
        self._opengl_renderer.cleanup()
        del self._opengl_renderer
        del self._context
        self.renderer = self._vulkan_renderer
        self.setSurfaceType(QSurface.SurfaceType.VulkanSurface)
        _vulkan.set_window_id(self.winId())
        self._vulkan_renderer.recreateSurface()
        self.render_timer.start(16)

    def switch_to_opengl(self):
        self.render_timer.stop()
        time.sleep(0.1)  # give time for the last frame to be drawn
        self._initializeOpenGLSurface()
        self._opengl_renderer = OpenGLRenderer(self, self._context)
        self.renderer = self._opengl_renderer
        self.render_timer.start(16)

    def exposeEvent(self, event):
        super().exposeEvent(event)
        # Window is exposed (visible) - NOW we can init Vulkan!
        if self.isExposed() and not self.renderer:
            self._initializeVulkan()
            self.renderer = self._vulkan_renderer

    def _initializeVulkan(self):
        self.setSurfaceType(QSurface.SurfaceType.MetalSurface)

        # Get the window handle
        window_id = int(self.winId())
        app = QApplication.instance()
        platform = app.platformName()
        if platform == "wayland":
            pni = app.nativeInterface()
            display = pni.display()
            _vulkan.set_window_system_type(_vulkan.SurfaceBackend.Wayland)
        elif platform == "xcb":
            pni = app.nativeInterface()
            display_id = pni.connection()
            _vulkan.set_window_system_type(_vulkan.SurfaceBackend.Xcb)

        _vulkan.set_window_id(window_id)
        _vulkan.set_display_id(display_id)

        # Create and initialize renderer
        self._vulkan_renderer = _vulkan.VulkanRenderer(self._vulkan_context)
        self._vulkan_renderer.initVulkan()
        self._vulkan_renderer.clearToBlack()

    def render_frame(self):
        if self.renderer and self.isExposed():
            try:
                self.renderer.drawFrame()  # Or clearToBlack() for testing
            except Exception as e:
                print(f"Render error: {e}")
                self.render_timer.stop()

    def keyPressEvent(self, event):
        print("key pressed")
        if event.key() == Qt.Key.Key_V:
            self.switch_to_vulkan()
        elif event.key() == Qt.Key.Key_G:
            self.switch_to_opengl()

class GraphicsWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.graphics_window = GraphicsWindow()
        self.graphics_container = QWidget.createWindowContainer(self.graphics_window, self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.graphics_container)
        layout.setContentsMargins(0, 0, 0, 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL Triangle - Press V for Vulkan, G for OpenGL")
        self.setGeometry(100, 100, 800, 600)

        central_widget = GraphicsWidget()
        self.setCentralWidget(central_widget)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q:
            self.close()


def main():
    app = QApplication(sys.argv)
    print(app.platformName())

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
