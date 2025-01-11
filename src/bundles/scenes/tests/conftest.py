# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# You can also
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


import warnings

import pytest


@pytest.fixture(scope="function")
def ensure_chimerax_initialized():
    _ensure_chimerax_initialized()


# You can import this hidden one if you need to run code that needs ChimeraX initialized before
# the tests even run, as in amber_info, which tries to access chimerax.app_bin_dir when you import
# it.
def _ensure_chimerax_initialized():
    import chimerax

    if not getattr(chimerax, "app_bin_dir", None):
        import chimerax.core.__main__

        chimerax.core.__main__.init(["dummy", "--nogui", "--safemode", "--exit"])


_test_session = None


def get_test_session():
    global _test_session
    if _test_session is None:
        from chimerax.core import get_minimal_test_session

        with warnings.catch_warnings(action="ignore"):
            _test_session = get_minimal_test_session()
    return _test_session


@pytest.fixture(scope="function")
def test_production_session():
    session = get_test_session()
    yield session
    session.reset()


def pytest_configure(config):
    markexpr = config.getoption("markexpr")
    if "not wheel" in markexpr:
        # Initialize the test session before tests are even collected, because
        # pytest's usual schtick of importing modules BEFORE the tests are collected
        # totally breaks code that modifies __all__s at runtime. We need ChimeraX to
        # always be the first thing that runs in any tool.
        _ = get_test_session()
