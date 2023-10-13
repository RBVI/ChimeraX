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

# ensure atomic_libs C++ shared libs are linkable by us
import chimerax.atomic_lib

from ._dssp import compute_ss as _compute_ss

from chimerax.core.toolshed import BundleAPI

def compute_ss(structure, **kw):
    """Compute/assign secondary structure using Kabsch & Sander DSSP algorithm

    Parameters
    ----------
    structure : :py:class:`~chimerax.atomic.AtomicStructure`
        The structure to use.
    energy_cutoff : float, optional
        hbond energy cutoff (default -0.5).
    min_helix_len : int, optional
        minimum helix length (default 3).
    min_strand_len : int, optional
        minimum strand length (default 3).
    report : bool, optional
        whether to log computed values (default false).
    return_values : bool, optional
        whether to return computed values (default false).

    Returns
    -------
    None or dict
        Optionally returns dictonary of computed values.  The values include:

        ``strands``: a list of :py:class:`~chimerax.atomic.Residue` pairs.
        ``sheets``: a list of sets of indices into strands.
        ``strand_parallel``: a dict of index pairs into strands to true if parallel.
        ``helix_info``: a list of pair of :py:class:`~chimerax.atomic.Residue` with
        the dssp report character as an integer.
    """
    return _compute_ss(structure._c_pointer.value, **kw)

class _DsspBundle(BundleAPI):
    pass

bundle_api = _DsspBundle()
