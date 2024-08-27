
import glob
import os

import pytest

test_data_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "test-data"
)

test_files = [
    os.path.basename(file) for file in glob.glob(os.path.join(test_data_dir, "*.*"))
]

import pytest


@pytest.mark.parametrize("file", test_files)
def test_imports(test_production_session, file):
    from chimerax.core.commands import run

    run(test_production_session, "open %s" % os.path.join(test_data_dir, file))
