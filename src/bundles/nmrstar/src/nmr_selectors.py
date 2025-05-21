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

# -----------------------------------------------------------------------------
#
def _register_distance_constraint_selectors(logger):
    'Add selector for satisfied and long constraints for easy hiding.'
    from chimerax.core.commands import register_selector
    register_selector("nmr-satisfied", _satisfied_constraint_selector, logger, desc='satisfied NMR distance restraints')
    register_selector("nmr-long", _long_constraint_selector, logger, desc='long NMR distance restraints')

# -----------------------------------------------------------------------------
#
def _satisfied_constraint_selector(session, models, results):
    pblist = []
    from chimerax.atomic import Pseudobonds, PseudobondGroup
    for pbg in models:
        if isinstance(pbg, PseudobondGroup):
            for pbond in pbg.pseudobonds:
                if hasattr(pbond, 'nmr_max_distance') and pbond.length <= pbond.nmr_max_distance:
                    pblist.append(pbond)
    pbonds = Pseudobonds(pblist)
    results.add_pseudobonds(pbonds)
    for m in pbonds.unique_groups:
        results.add_model(m)

# -----------------------------------------------------------------------------
#
def _long_constraint_selector(session, models, results):
    pblist = []
    from chimerax.atomic import Pseudobonds, PseudobondGroup
    for pbg in models:
        if isinstance(pbg, PseudobondGroup):
            for pbond in pbg.pseudobonds:
                if hasattr(pbond, 'nmr_max_distance') and pbond.length > pbond.nmr_max_distance:
                    pblist.append(pbond)
    pbonds = Pseudobonds(pblist)
    results.add_pseudobonds(pbonds)
    for m in pbonds.unique_groups:
        results.add_model(m)
