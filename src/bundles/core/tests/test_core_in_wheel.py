import pytest


@pytest.mark.wheel
def test_runtime_env_in_wheel():
    import chimerax.core

    assert not chimerax.core.runtime_env_is_chimerax_app()
