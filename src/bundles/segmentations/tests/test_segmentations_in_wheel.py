import pytest

import chimerax.segmentations


@pytest.mark.wheel
def test_public_api_visibility():
    assert not hasattr(chimerax.segmentations, "bundle_api")
