import os
import sys


def test_chimerax_module_attrs(ensure_chimerax_initialized):
    import chimerax

    bin_dir = chimerax.app_bin_dir
    if sys.platform == "win32":
        assert os.path.exists(os.path.join(bin_dir, "ChimeraX.exe"))
    else:
        assert os.path.exists(os.path.join(bin_dir, "ChimeraX"))
