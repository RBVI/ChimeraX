"""GNOME appearance comes from GSettings instead of Qt's GTK3 platform theme."""

import pytest

from chimerax.ui import desktop_settings


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(desktop_settings, "_settings", None)
    monkeypatch.delenv("QT_QPA_PLATFORMTHEME", raising=False)
    monkeypatch.setattr(desktop_settings.sys, "platform", "linux")
    return monkeypatch


SETTINGS = {"color-scheme": "default", "gtk-theme": "Yaru", "icon-theme": "Yaru",
            "font-name": "Ubuntu Sans 11"}


@pytest.mark.parametrize("desktop", ["GNOME", "ubuntu:GNOME", "Unity"])
def test_gnome_desktops_use_the_gnome_theme(clean, desktop):
    clean.setenv("XDG_CURRENT_DESKTOP", desktop)
    clean.setattr(desktop_settings, "_read_settings", lambda: dict(SETTINGS))
    assert desktop_settings.select_platform_theme()
    assert desktop_settings.active()
    import os
    assert os.environ["QT_QPA_PLATFORMTHEME"] == "gnome"


@pytest.mark.parametrize("desktop", ["KDE", "XFCE", "X-Cinnamon", ""])
def test_other_desktops_keep_their_theme(clean, desktop):
    clean.setenv("XDG_CURRENT_DESKTOP", desktop)
    clean.setattr(desktop_settings, "_read_settings", lambda: dict(SETTINGS))
    assert not desktop_settings.select_platform_theme()
    assert not desktop_settings.active()


def test_a_chosen_platform_theme_is_kept(clean):
    clean.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    clean.setenv("QT_QPA_PLATFORMTHEME", "gtk3")
    clean.setattr(desktop_settings, "_read_settings", lambda: dict(SETTINGS))
    assert not desktop_settings.select_platform_theme()


def test_unreadable_settings_keep_the_default_theme(clean):
    clean.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    clean.setattr(desktop_settings, "_read_settings", lambda: None)
    assert not desktop_settings.select_platform_theme()

    def fail():
        raise OSError("no libgio")
    clean.setattr(desktop_settings, "_read_settings", fail)
    assert not desktop_settings.select_platform_theme()


@pytest.mark.parametrize("scheme, gtk_theme, dark", [
    ("prefer-dark", "Yaru", True),
    ("prefer-light", "Yaru-dark", False),
    ("default", "Yaru-blue-dark", True),
    ("default", "Adwaita", False),
])
def test_dark_preference(clean, scheme, gtk_theme, dark):
    clean.setattr(desktop_settings, "_settings", dict(SETTINGS, **{"color-scheme": scheme, "gtk-theme": gtk_theme}))
    assert desktop_settings.prefers_dark() is dark


def test_no_dark_preference_without_settings(clean):
    assert desktop_settings.prefers_dark() is False


@pytest.mark.parametrize("description, family, size", [
    ("Ubuntu Sans 11", "Ubuntu Sans", 11.0),
    ("Cantarell 10.5", "Cantarell", 10.5),
    ("DejaVu Sans, 9", "DejaVu Sans", 9.0),
    ("Sans", "Sans", None),
])
def test_parse_font(description, family, size):
    assert desktop_settings.parse_font(description) == (family, size)


def test_read_settings_from_this_system():
    # Exercises the ctypes calls where GSettings has the GNOME schema.
    import sys
    if not sys.platform.startswith("linux"):
        pytest.skip("GSettings is used on Linux")
    try:
        settings = desktop_settings._read_settings()
    except OSError:
        pytest.skip("no libgio")
    if settings is None:
        pytest.skip("no org.gnome.desktop.interface schema")
    assert set(settings) >= {"icon-theme", "font-name"}
    assert all(isinstance(v, str) for v in settings.values())
