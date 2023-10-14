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

# $Id: __init__.py 41155 2016-06-30 23:18:29Z pett $


def call_c_plus_plus(cpp_func, structures, return_collection, *args):
    import os
    num_cpus = os.cpu_count()
    if num_cpus is None:
        num_cpus = 1

    from chimerax.atomic import Atom, Atoms, AtomicStructure
    groups = []
    for structure in structures:
        if not isinstance(structure, AtomicStructure):
            continue
        cpp_args = args + (num_cpus, return_collection)
        grps = cpp_func(structure.cpp_pointer, *cpp_args)
        if return_collection:
            # accumulate the numpy arrays to later be concatenated and turned into a Collection
            groups.append(grps)
        else:
            groups.extend(grps)
    if return_collection:
        if groups:
            import numpy
            groups = Atoms(numpy.concatenate(groups))
        else:
            groups = Atoms()
    return groups

def collate_results(results, return_collection):
    if return_collection:
        from chimerax.atomic import concatenate
        return concatenate(results)
    ret_val = []
    for result in results:
        ret_val.extend(result)
    return ret_val
