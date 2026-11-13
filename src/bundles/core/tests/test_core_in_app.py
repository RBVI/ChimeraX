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


def test_built_app_has_install_timestamp(test_production_session):
    # Without it the installed bundle cache is never valid, so every start
    # rebuilds it.  Builds that install bundles without ChimeraX write it.
    _ = test_production_session
    import os
    import chimerax
    from chimerax.core.toolshed import _TIMESTAMP

    assert os.path.isfile(os.path.join(chimerax.app_data_dir, _TIMESTAMP))
