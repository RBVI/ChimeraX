import os
import sys
import chimerax

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import chimerax_test_utils

chimerax_test_utils.ensure_chimerax_initialized()


def test_chimerax_module_attrs():
    bin_dir = chimerax.app_bin_dir
    if sys.platform == "win32":
        assert os.path.exists(os.path.join(bin_dir, "ChimeraX.exe"))
    else:
        assert os.path.exists(os.path.join(bin_dir, "ChimeraX"))
