# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# $Id: __init__.py 41155 2016-06-30 23:18:29Z pett $


def call_c_plus_plus(cpp_func, structures, return_collection, *args):
	import os
	num_cpus = os.cpu_count() if not None else 1

	from chimerax.core.atomic import Atom, Atoms, AtomicStructure
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
			from chimerax.core.atomic.molobject import object_map
			groups.extend([[object_map(ptr, Atom) for ptr in ptrs] for ptrs in grps])
	if return_collection:
		if groups:
			import numpy
			groups = Atoms(numpy.concatenate(groups))
		else:
			groups = Atoms()
	return groups

def collate_results(results, return_collection):
	if return_collection:
		from chimerax.core.atomic.molarray import concatenate
		return concatenate(results)
	ret_val = []
	for result in results:
		ret_val.extend(result)
	return ret_val
