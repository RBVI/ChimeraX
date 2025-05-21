import pytest
import sys


@pytest.mark.parametrize(
    "directory",
    ["app_dirs", "app_dirs_unversioned", "app_bin_dir", "app_data_dir", "app_lib_dir"],
)
def test_appdirs_defined_in_global_namespace(test_production_session, directory):
    _ = test_production_session
    import chimerax

    assert getattr(chimerax, directory) is not None


def test_env_is_chimerax_app():
    import chimerax.core

    assert chimerax.core.runtime_env_is_chimerax_app()
