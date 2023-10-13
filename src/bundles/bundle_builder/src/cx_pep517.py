# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
import glob
import logging
import os
import shutil
import tomli
import zipfile

log = logging.getLogger()

from .bundle_builder_toml import Bundle, read_toml

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None) -> None:
    bundle = Bundle(log, read_toml('pyproject.toml'))
    return bundle.build_wheel()

def build_sdist(sdist_directory, config_settings=None) -> None:
    bundle = Bundle(log, read_toml('pyproject.toml'))
    return bundle.build_sdist()

def build_editable(wheel_directory, config_settings=None, metadata_directory=None) -> None:
    bundle = Bundle(log, read_toml('pyproject.toml'))
    return bundle.build_editable(config_settings)

def get_requires_for_build_wheel(config_settings = None) -> None:
    toml_file = read_toml('pyproject.toml')
    return toml_file['build-system']['requires']

get_requires_for_build_sdist = get_requires_for_build_wheel
get_requires_for_build_editable = get_requires_for_build_wheel

# A bug in pip requires that we specify prepare_metadata_for_build_editable, even
# though it should take the dist-info directory of build_editable if it's not present.
#
# So because we don't want to teach bundle builder to make a dist info dir, since setuptools
# will do it for us just fine, we'll just build the wheel and return its dist info directory
# until pip is smart enough to do it on its own.
#
# This process is negligibly slower than building the wheel, because we discard the original
# wheel for setuptools's new two-file editable wheel.
def prepare_metadata_for_build_editable(metadata_directory, config_settings=None) -> None:
    bundle = Bundle(log, read_toml('pyproject.toml'))
    path_to_wheel = bundle.build_wheel_for_build_editable()
    dist_dir = os.path.dirname(path_to_wheel)
    with zipfile.ZipFile(path_to_wheel, 'r') as f:
        f.extractall(dist_dir)
    # Remove the old wheel and the chimerax folder, leaving only the dist-info directory.
    os.remove(path_to_wheel)
    shutil.rmtree(os.path.join(dist_dir, 'chimerax'))
    try:
        dist_info_filename = glob.glob(os.path.join(dist_dir, "*.dist-info"))[0]
    except IndexError:
        raise ValueError("No dist info dir found")
    return os.path.join(dist_dir, dist_info_filename)
