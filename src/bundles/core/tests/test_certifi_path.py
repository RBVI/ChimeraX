import os

from chimerax.core.__main__ import _certifi_where


def test_ca_bundle_matches_certifi():
    import certifi
    assert os.path.samefile(_certifi_where(), certifi.where())


def test_missing_plain_bundle_falls_back_to_certifi(monkeypatch):
    import certifi
    real_isfile = os.path.isfile
    monkeypatch.setattr(os.path, 'isfile',
                        lambda path: False if path.endswith('cacert.pem') else real_isfile(path))
    monkeypatch.setattr(certifi, 'where', lambda: 'from certifi')
    assert _certifi_where() == 'from certifi'
