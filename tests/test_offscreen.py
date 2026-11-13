import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="--offscreen is supported only on Linux")
def test_offscreen_renders_image(ensure_chimerax_initialized, tmp_path):
    import chimerax

    image = tmp_path / "sphere.png"
    subprocess.run(
        [
            os.path.join(chimerax.app_bin_dir, "ChimeraX"),
            "--offscreen",
            "--exit",
            "--cmd",
            f"set bgColor black; shape sphere radius 10 color white; view; save {image} width 64 height 64",
        ],
        check=True,
        timeout=300,
    )
    from PIL import Image

    # A drawn sphere, not just the background
    assert Image.open(image).convert("L").getextrema()[1] > 128
