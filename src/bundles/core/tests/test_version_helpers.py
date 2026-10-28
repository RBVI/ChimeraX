"""Startup reads version numbers without importing packaging.version."""

import pytest

VERSIONS = ['1.13', '1.13.dev202610280700', '1.12rc1', '1.12.1', '1.12.1.post3', '2', '1!3.4',
            'v1.10', '1.11b2+local.1', ' 1.9 ', '1.13.0a1.dev5', '0.9', '1.0.post1.dev3',
            '1.0+dev', '1.0dev', '1.0.DEV4', '1.0-dev_5', '1.0rc1.post1.dev0', '1.0+abc.dev3']


@pytest.mark.parametrize('version', VERSIONS)
def test_version_major_minor_matches_packaging(version):
    from packaging.version import Version
    from chimerax.core.utils import version_major_minor
    v = Version(version)
    assert version_major_minor(version) == (v.major, v.minor)
    assert version_major_minor(v) == (v.major, v.minor)


def test_version_major_minor_rejects_invalid_versions():
    from packaging.version import InvalidVersion
    from chimerax.core.utils import version_major_minor
    with pytest.raises(InvalidVersion):
        version_major_minor('')


@pytest.mark.parametrize('version', VERSIONS)
def test_is_daily_build_matches_packaging(version, monkeypatch):
    from packaging.version import Version
    import chimerax.core
    monkeypatch.setattr(chimerax.core, 'version', version)
    assert chimerax.core.is_daily_build() == Version(version).is_devrelease
