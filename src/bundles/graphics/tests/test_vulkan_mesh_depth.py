"""Ordinary Vulkan meshes must not write gl_FragDepth.

Any depth write in a compiled fragment shader disables early depth
rejection, so hidden surface fragments were fully lit (including every
multishadow lookup) before failing the depth test.
"""
import os
import re

from chimerax.graphics import vulkan


def _mesh_frag():
    path = os.path.join(os.path.dirname(vulkan.__file__), "shaders", "mesh.frag")
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_mesh_depth_writes_only_in_custom_depth_variant():
    depth = 0          # nesting inside #ifdef CUSTOM_FRAG_DEPTH
    stack = []
    for line in _mesh_frag().split("\n"):
        stripped = line.strip()
        if stripped.startswith("#if"):
            custom = stripped == "#ifdef CUSTOM_FRAG_DEPTH"
            stack.append(custom)
            depth += custom
        elif stripped.startswith("#endif"):
            depth -= stack.pop()
        elif re.search(r"\bgl_FragDepth\s*=", line) and not stripped.startswith("//"):
            assert depth > 0, f"unguarded depth write: {stripped}"


def test_define_is_inserted_after_version_line():
    src = "// header\n#version 450\nvoid main() {}\n"
    out = vulkan._define_after_version(src, "CUSTOM_FRAG_DEPTH")
    assert out.split("\n")[:3] == [
        "// header", "#version 450", "#define CUSTOM_FRAG_DEPTH"]
