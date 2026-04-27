#!/usr/bin/env python3

import sys
import time
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import (
    QWindow,
    QSurface,
    QSurfaceFormat,
    QOpenGLContext,
    QNativeInterface,
)
from PySide6.QtCore import Qt, QTimer

import numpy as np
from OpenGL import GL


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
        vertices = np.array(
            [
                # x,    y,     r,   g,   b
                -0.5,
                -0.5,
                1.0,
                0.0,
                0.0,  # Bottom left - red
                0.5,
                -0.5,
                0.0,
                1.0,
                0.0,  # Bottom right - green
                0.0,
                0.5,
                0.0,
                0.0,
                1.0,  # Top - blue
            ],
            dtype=np.float32,
        )

        # Create and bind VAO
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Create and bind VBO
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW
        )

        ## Set up vertex attributes
        GL.glVertexAttribPointer(
            0, 2, GL.GL_FLOAT, GL.GL_FALSE, 20, GL.ctypes.c_void_p(0)
        )
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            1, 3, GL.GL_FLOAT, GL.GL_FALSE, 20, GL.ctypes.c_void_p(8)
        )
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

        self._context = None
        self._opengl_renderer = None

        self.renderer = None
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.render_frame)
        self.initialized = False

    def _initializeOpenGLSurface(self):
        self.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)
        self._opengl_surface_format = QSurfaceFormat()
        self._opengl_surface_format.setVersion(3, 3)
        self._opengl_surface_format.setProfile(
            QSurfaceFormat.OpenGLContextProfile.CoreProfile
        )
        self._opengl_surface_format.setRenderableType(
            QSurfaceFormat.RenderableType.OpenGL
        )
        self._context = QOpenGLContext()
        self._context.setFormat(self._opengl_surface_format)
        self._context.create()

    def exposeEvent(self, event):
        super().exposeEvent(event)
        # Window is exposed (visible) - NOW we can init Vulkan!
        if self.isExposed() and not self.renderer:
            self._initializeOpenGLSurface()
            self.renderer = OpenGLRenderer(self, self._context)
            self.render_timer.start(16)

    def render_frame(self):
        if self.renderer and self.isExposed():
            try:
                self.renderer.drawFrame()  # Or clearToBlack() for testing
            except Exception as e:
                print(f"Render error: {e}")
                self.render_timer.stop()


class GraphicsWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.graphics_window = GraphicsWindow()
        self.graphics_container = QWidget.createWindowContainer(
            self.graphics_window, self
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.graphics_container)
        layout.setContentsMargins(0, 0, 0, 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL Triangle")
        self.setGeometry(100, 100, 800, 600)

        central_widget = GraphicsWidget()
        self.setCentralWidget(central_widget)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q:
            self.close()


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
