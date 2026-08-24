# SPDX-License-Identifier: MIT
# Copyright 2026 Samuel Lobo

from chimerax.core.toolshed import BundleAPI

__version__ = "0.1.1"


class _FastHydroMapAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        if command_name == "fasthydromap":
            from .cmd import register_fasthydromap_command
            register_fasthydromap_command(logger)
        elif command_name == "fasthydromap install":
            from .install import register_fasthydromap_install_command
            register_fasthydromap_install_command(logger)


bundle_api = _FastHydroMapAPI()
