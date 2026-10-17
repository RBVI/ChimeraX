from os.path import dirname, join

from chimerax.graphics import opengl


def test_opengl_shader_sources_are_ascii():
    # Shader.compile_shader opens these with the locale's default encoding.
    # When ChimeraX starts without a UTF-8 locale that is ASCII, and a
    # non-ASCII character (e.g. an em dash in a comment) breaks the OpenGL
    # status bar.
    for name in ("vertexShader.txt", "fragmentShader.txt"):
        with open(join(dirname(opengl.__file__), name), "rb") as f:
            for lineno, line in enumerate(f, start=1):
                assert line.isascii(), f"{name}:{lineno} has non-ASCII text"
