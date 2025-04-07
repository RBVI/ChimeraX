# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
import glob
import logging
import os
import shutil
import zipfile

log = logging.getLogger()

from .bundle_builder_toml import Bundle, read_toml


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    bundle = Bundle(log, read_toml("pyproject.toml"))
    wheel = bundle.build_wheel()
    return os.path.basename(wheel)


def build_sdist(sdist_directory, config_settings=None) -> str:
    bundle = Bundle(log, read_toml("pyproject.toml"))
    return os.path.basename(bundle.build_sdist())


def build_editable(
    wheel_directory, config_settings=None, metadata_directory=None
) -> str:
    bundle = Bundle(log, read_toml("pyproject.toml"))
    return os.path.basename(bundle.build_editable(config_settings))


def get_requires_for_build_wheel(config_settings=None) -> None:
    toml_file = read_toml("pyproject.toml")
    return toml_file["build-system"]["requires"]


get_requires_for_build_sdist = get_requires_for_build_wheel
get_requires_for_build_editable = get_requires_for_build_wheel

