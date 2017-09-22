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

from .settings import defaults

def find_clashes(session, test_atoms,
        assumed_max_vdw=2.1,
        attr_name=defaults["attr_name"],
		bond_separation=defaults["bond_separation"],
        clash_threshold=defaults["clash_threshold"],
        coordset=None,
        distance_only=None,
        group_name=defaults["group_name"],
		hbond_allowance=defaults["clash_hbond_allowance"],
		inter_model=True,
		inter_submodel=False,
        intra_res=False,
        intra_mol=True,
        test="others"):
	"""Detect steric clashes/contacts

	   'test_atoms' should be an Atoms collection.

	   If 'test' is 'others' then non-bonded clashes between atoms in
	   'test_atoms' and non-'test_atoms' atoms will be found.  If 'test'
       is 'self' then non-bonded clashes within 'test_atoms' atoms will be
       found.  Otherwise 'test' should be a list of atoms to test against.
	   The "clash value" is the sum of the VDW radii minus the distance,
	   keeping only the maximal clash (which must exceed 'clash_threshold').

	   'hbond_allowance' is how much the clash value is reduced if one
	   atom is a donor and the other an acceptor.

       If 'distance_only' is set (in which case it must be a positive numeric
       value), then both VDW radii and hbond_allowance are ignored and the
       center-center distance between the atoms must <= the given value.

	   Atom pairs are eliminated from consideration if they are less than
	   or equal to 'bond_separation' bonds apart.

	   Intra-residue clashes are ignored unless intra_res is True.
	   Intra-molecule (covalently connected fragment) clashes are ignored
	   unless intra_mol is True.
	   Inter-submodel clashes are ignored unless inter_submodel is True.
       Inter-model clashes are ignored unless inter_model is True.

	   If 'coordset' is specified, that coordset will be used for the
	   atoms.  Obviously, all the atoms have to be in the same model
	   in such a case.

	   Returns a dictionary keyed on atoms, with values that are
	   dictionaries keyed on clashing atom with value being the clash value.
	"""

	# use the fast _closepoints module to cut down candidate atoms if we
	# can (since _closepoints doesn't know about "non-bonded" it isn't as
	# useful as it might otherwise be)
	if test == "others":
        if inter_model:
            from chimera.core.atomic import all_atoms
            universe_atoms = all_atoms(session)
        else:
            from chimera.core.atomic import structure_atoms
            universe_atoms = structure_atoms(test_atoms.unique_structures)
        other_atoms = universe_atoms.subtract(test_atoms)
		if len(other_atoms) == 0:
			from chimerax.core.errors import UserError
			raise UserError("All atoms are in test set: no others available to test against")
		cutoff = 2.0 * assumed_max_vdw - clash_threshold
		t_close, o_close = find_close_points(test_atoms.scene_coords,
                                other_atoms.scene_coords, cutoff)
		test_atoms = test_atoms[t_close]
		search_atoms = other_atoms[o_close]
	elif not isinstance(test, basestring):
		search_atoms = test
	else:
		search_atoms = test_atoms

    #TODO (remember that 'inter-model' logic needs to be added)

	from chimera.misc import atomSearchTree
	tree = atomSearchTree(list(search_atoms), xformed=not crdSet, crdSet=crdSet)
	clashes = {}
	for a in test_atoms:
		cutoff = a.radius + assumed_max_vdw - clash_threshold
		if crdSet:
			nearby = tree.searchTree(a.coord(crdSet).data(), cutoff)
		else:
			nearby = tree.searchTree(a.xformCoord().data(), cutoff)
		if not nearby:
			continue
		needExpansion = a.allLocations()
		exclusions = set(needExpansion)
		for i in range(bond_separation):
			nextNeed = []
			for expand in needExpansion:
				for n in expand.neighbors:
					if n in exclusions:
						continue
					exclusions.add(n)
					nextNeed.append(n)
			needExpansion = nextNeed
		for nb in nearby:
			if nb in exclusions:
				continue
			if not intra_res and a.residue == nb.residue:
				continue
			if not intra_mol and a.molecule.rootForAtom(a,
					True) == nb.molecule.rootForAtom(nb, True):
				continue
			if a in clashes and nb in clashes[a]:
				continue
			if not inter_submodel \
			and a.molecule.id == nb.molecule.id \
			and a.molecule.subid != nb.molecule.subid:
				continue
			if test == "model" and a.molecule != nb.molecule:
				continue
			if crdSet:
				clash = a.radius + nb.radius - a.coord(crdSet).distance(nb.coord(crdSet))
			else:
				clash = a.radius + nb.radius - a.xformCoord().distance(
							nb.xformCoord())
			if hbond_allowance:
				if (_donor(a) and _acceptor(nb)) or (
				_donor(nb) and _acceptor(a)):
					clash -= hbond_allowance
			if clash < clash_threshold:
				continue
			clashes.setdefault(a, {})[nb] = clash
			clashes.setdefault(nb, {})[a] = clash
	return clashes

"""
hyd = chimera.Element(1)
negative = set([chimera.Element(sym) for sym in ["N", "O", "S"]])
from chimera.idatm import typeInfo
def _donor(a):
	if a.element == hyd:
		if a.neighbors and a.neighbors[0].element in negative:
			return True
	elif a.element in negative:
		try:
			if len(a.bonds) < typeInfo[a.idatmType].substituents:
				# implicit hydrogen
				return True
		except KeyError:
			pass
		for nb in a.neighbors:
			if nb.element == hyd:
				return True
	return False

def _acceptor(a):
	try:
		info = typeInfo[a.idatmType]
	except KeyError:
		return False
	return info.substituents < info.geometry
"""
