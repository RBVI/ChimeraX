# SPDX-License-Identifier: MIT
# Copyright 2026 Samuel Lobo

from chimerax.core.settings import Settings


class _FastHydroMapSettings(Settings):
    EXPLICIT_SAVE = {
        "fasthydromap_install_location": "",
    }


def _fasthydromap_settings(session):
    settings = getattr(session, "_fasthydromap_settings", None)
    if settings is None:
        settings = _FastHydroMapSettings(session, "fasthydromap")
        session._fasthydromap_settings = settings
    return settings
