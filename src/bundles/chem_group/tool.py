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

"""Find chemical groups in a structure"""

import chimerax
from chimerax.core.atomic.idatm import type_info, tetrahedral, planar, linear, single
#TODO: from miscFind import findAromatics, findAliAmines, findAroAmines
findAromatics = findAliAmines = findAroAmines = None

# R is a shorthand for alkyl group
# X is a shorthand for 'any halide'
# None matches anything (and nothing)
from chimerax.core.atomic import Element
N = Element.get_element('N').number
C = Element.get_element('C').number
O = Element.get_element('O').number
H = Element.get_element('H').number
R = (C, H)
X = ('F', 'Cl', 'Br', 'I')
single_bond = (H, {'geometry':tetrahedral}, {'geometry':single})
heavy =	{'not_type': ['H', 'HC', 'D', 'DC']}
non_oxygen_single_bond = (H, {'geometry':tetrahedral, 'not_type': ['O', 'O3', 'O2', 'O3-', 'O2-', 'Oar', 'Oar+']}, {'geometry':single})

# 'R'/None and H must be found only in a list of descendents (not as the key
# element of a group or subgroup) and they must be arranged to be trailing
# components of a group description
#
# H can match nothing (and therefore R can match nothing)
#
# Note that 'R' in the textual group description is interpreted to mean:
# "C or H, unless H would produce a different functional group (then just C)"
#
#---
#
# For algorithmic group descriptions that aren't themselves functions,
# the following applies:
#
#	Atoms are indicated as either strings [idatm types] or numbers/
#	symbolic constants [element number].  Tuples indicate the atom
#	can be any one of the alternatives in the tuple.
#
#	A list where an atom is expected is always two elements:  an atom,
#	and a list of what is bonded to that atom.
#
#	Dictionaries indicate facets of the idatm type (as per the
#	chimera.idatm.type_info dictionaries).  The keys of the dictionary
#	correspond to type_info dictionaries' keys, and the values are what
#	attribute value the atom must have to match.  The dictionary can
#	have a special key named 'default'.  The value of True for 'default'
#	means that atom types not in the type_info dictionary will match,
#	False means they won't.  The default for 'default' [urg] is False.  The
#	dictionary can also have a special 'not_type' key, whose value is a
#	list of idatm types that aren't allowed to match.
#
#	Lastly, an "atom" can be an instance of RingAtom, which is initialized
#	with two arguments, the first of which is any of the preceding
#	forms of atom specification, and the second is the number of
#	intraresidue rings that the atom participates in 
#
#---
#
# The last component of the group description indicates which atoms are
# considered "principal" and are returned as part of the atom group.  This
# typically prunes the 'R's from a group.  A '1' indicates the atom is
# principal, a '0' that it is not.  Due to the algorithm for pruning, 
# 'optional' atoms (R and H) must appear at the end of the group description,
# with all R's before any H's.
#
# This last component is also used to indicate ring closures.  A value that
# is not '0' or '1' is assumed to the first of several such values.  Each
# atom with that value must be identical or the group will be pruned.  All
# atoms but the first with that value will be pruned from the returned group.
#

class RingAtom:
	def __init__(self, atom_desc, num_rings):
		self.atom_desc = atom_desc
		self.num_rings = num_rings

group_info = {
	"acyl halide":	("R(C=O)X",	['C2', [X, 'O2', single_bond]], [1,1,1,0]),
	"adenine":	("6-aminopurine",
			['Npl', [['Car', ['Car', ['N2', [['Car', [['N2', [['Car', ['Car', ['Npl', ['C3', ['Car', [['N2', ['Car']], H]]]]]]]], H]]]]]], H, H]],
					[1,1,2,1,1,1,1,2,1,0,1,1,2,1,1,1,1]),
	"aldehyde":	("R(C=O)H",	['C2', ['O2', single_bond, H]], [1,1,0,1]),
	"amide":	("R(C=O)NR2",	['C2', ['O2', 'Npl', None]], [1,1,1,0]),
	"amine":	("RxNHy",	lambda m: findAliAmines(m) + findAroAmines(m, 0), None),
	"aliphatic amine": ("RxNHy",	findAliAmines, None),
	"aliphatic primary amine": ("RNH2",	[('N3','N3+'), ['C3', H, H, H]], [1,0,1,1,1]),
	"aliphatic secondary amine": ("R2NH",	[('N3','N3+'), ['C3', 'C3', H, H]], [1,0,0,1,1]),
	"aliphatic tertiary amine": ("R3N",		[('N3','N3+'), ['C3', 'C3', 'C3', H]], [1,0,0,0,1]),
	"aliphatic quaternary amine": ("R4N+",	['N3+', ['C3', 'C3', 'C3', 'C3']], [1,0,0,0,0]),
	"aromatic amine": ("RxNHy",	lambda m: findAroAmines(m, 0), None),
	"aromatic primary amine": ("RNH2",	lambda m: findAroAmines(m, 1), None),
	"aromatic secondary amine": ("R2NH",	lambda m: findAroAmines(m, 2), None),
	"aromatic tertiary amine": ("R3N",		lambda m: findAroAmines(m, 3), None),
	"aromatic ring":("aromatic",	findAromatics, None),
	"carbonyl":	("R2C=O",		['O2', [C]], [1,1]),
	"carboxylate":	("RCOO-",	['Cac', [['O2-', []], ['O2-', []], single_bond]], [1,1,1,0]),
	"cytosine":	("2-oxy-4-aminopyrimidine",
					['Npl', [['C2', ['N2', ['C2', [['C2', [['Npl', ['C3', ['C2', ['N2', 'O2']]]], H]], H]]]]], H, H],
					[1,1,2,1,1,1,0,1,2,1,1,1,1,1]),
	"disulfide":	("RSSR",	['S3', [['S3', [single_bond]], single_bond]], [1,1,0,0]),
	"ester":	("R(C=O)OR",	['C2', ['O2', [O, [C]], single_bond]], [1,1,1,0,0]),
	"ether O":	("ROR",		['O3', [C, C]], [1,0,0]),
	"guanine":	("2-amino-6-oxypurine",
		['Npl', [['C2', [['N2', ['Car']], ['Npl', [['C2', ['O2', ['Car', ['Car', ['N2', [['Car', [['Npl', ['C3', 'Car']], H]]]]]]]], H]]]], H, H]],
					[1,1,1,2,1,1,1,1,2,1,1,1,0,2,1,1,1,1]),
	"halide":	("RX",		[X, [heavy]], [1,0]),
	"hydroxyl":	("COH or NOH",	['O3', [(C, N), H]], [1,0,1]),
	"imine":	("R2C=NR",	['C2', [['N2', [single_bond]], single_bond, single_bond]], [1,1,0,0,0]),
	"ketone":	("R2C=O",	['C2', ['O2', C, C]], [1,1,0,0]),
	"methyl":	("RCH3",	['C3', [heavy, H, H, H]], [1,0,1,1,1]),
	"nitrile":	("RC*N",	['C1', ['N1', single_bond]], [1,1,0]),
	"nitro":	("RNO2",	['Ntr', ['O2-', 'O2-', non_oxygen_single_bond]], [1,1,1,0]),
	"phosphate":	("PO4",		['Pac', [O, O, O, O]], [1,1,1,1,1]),
	"phosphinyl":	("R2PO2-",	['Pac', [['O3-', []], ['O3-', []], R,R]], [1,1,1,0,0]),
	"phosphonate":	("RPO3-",	['Pac', ['O3-', 'O3-', 'O3-', C]], [1,1,1,1,0]),
	"purine":	("purine",	['Car', ['Car', RingAtom(N,1), [RingAtom(N,1), [[RingAtom('Car',1), [[RingAtom(N,1), [[RingAtom('Car',2), [RingAtom('Car',2), [RingAtom('Car',1), [[RingAtom(N,1), [[RingAtom('Car',1), [RingAtom(N,1), None]], None]], None]]]], None]], None]], None]]]],
					[3,2,4,1,1,1,2,3,1,1,1,4,0,0,0,0,0,0]),
	"pyrimidine":	("pyrimidine",	[RingAtom(N,1), [RingAtom('Car',1), [RingAtom('Car',1), [[RingAtom(N,1), [[RingAtom('Car',1), [[RingAtom('Car',1), [RingAtom('Car',1), None]], None]], None]], None]], None]],
					[1,2,1,1,1,1,2,0,0,0,0,0]),
	"sulfate":	("SO4",		['Sac', [O, O, O, O]], [1,1,1,1,1]),
	"sulfonamide":	("RSO2NR2",	['Son', ['O2', 'O2', 'Npl', non_oxygen_single_bond]], [1,1,1,0,0]),
	"sulfonate":	("RSO3-",	['Sac', ['O3-', 'O3-', 'O3-', C]], [1,1,1,1,0]),
	"sulfone":	("R2SO2",	['Son', ['O2', 'O2', C, C]], [1,1,1,0,0]),
	"sulfonyl":	("R2SO2",	['Son', ['O2', 'O2', non_oxygen_single_bond, non_oxygen_single_bond]],
					[1,1,1,0,0]),
	"thiocarbonyl":	("C=S",		['S2', [C]], [1,1]),
	"thioether":	("RSR",		['S3', [C, C]], [1,0,0]),
	"thiol":	("RSH",		['S3', [C, H]], [1,0,1]),
	"thymine":	("5-methyl-2,4-dioxypyrimidine",
				['C3', [['C2', ['C2', ['C2', ['O2', ['Npl', [['C2', ['O2', ['Npl', ['C3', ['C2', ['C2', H]]]]]], H]]]]]], H, H, H]],
					[1,2,3,1,1,1,1,1,1,0,3,2,1,1,1,1,1]),
	"uracil":	("2,4-dioxypyrimidine",
					['O2', [['C2', ['Npl', ['Npl', [['C2', ['O2', ['C2', [['C2', [['Npl', ['C2', 'C3']], H]], H]]]], H]]]]]],
					[1,2,3,1,1,1,1,1,3,2,0,1,1,1])
}


def find_group(group_desc, structures):
	
	if isinstance(group_desc, str):
		try:
			group_formula, group_rep, group_principals = group_info[group_desc]
		except:
			raise KeyError("No known chemical group named '%s'" % group_desc)
	else:
		group_rep, group_principals = group_desc
	
	if callable(group_rep):
		return group_rep(structures)
	
	import os
	num_cpus = os.cpu_count() if not None else 1
	from ._chem_group import find_group as fg
	from chimerax.core.atomic import Atoms, AtomicStructure
	groups = []
	for structure in structures:
		if not isinstance(structure, AtomicStructure):
			continue
		"""
		groups.extend([Atoms(atom_ptrs) for atom_ptrs in
			fg(structure.cpp_pointer, group_rep, group_principals, RingAtom, num_cpus)])
		"""
		from time import time
		t0 = time()
		grps = fg(structure.cpp_pointer, group_rep, group_principals, RingAtom, num_cpus)
		t1 = time()
		from chimerax.core.atomic import molarray
		molarray.instrument_collection = True
		collections = [Atoms(atom_ptrs, guaranteed_live_pointers = True) for atom_ptrs in grps]
		molarray.instrument_collection = False
		t2 = time()
		groups.extend(collections)
		t3 = time()
		print("Call C++:", t1-t0)
		print("Form collections:", t2-t1)
		print(" 1:", molarray.accum_t[0])
		print(" 2:", molarray.accum_t[1])
		print(" 3:", molarray.accum_t[2])
		print(" 4:", molarray.accum_t[3])
		print(" 5:", molarray.accum_t[4])
		print(" 6:", molarray.accum_t[5])
		print("Extend groups:", t3-t2)
	return groups
'''
	groups = []
	for structure in structures:
		for atom in structure.atoms:
			found_groups = _trace_group(atom, None, group_rep)

			if found_groups == None:
				continue
			
			if not group_principals:
				groups = groups + found_groups
				continue
			
			for group in found_groups:
				principal_group = []
				rings_okay = True
				ring_closures = {}
				for i in range(len(group)):
					principal = group_principals[i]
					if principal:
						if ring_closures.has_key(principal):
							if group[i] !=  ring_closures[principal]:
								rings_okay = False
								break
							# don't add second
							# instance of this atom
							continue
						if principal != 1:
							ring_closures[principal] = group[i]
							
						principal_group.append(group[i])
				if rings_okay:
					groups.append(principal_group)
	return groups

def _trace_group(atom, parent, frag_rep):
	# see if 'atom' satisfies the requirements of 'frag_rep' without using
	# the 'parent' atom

	if isinstance(frag_rep, list):
		atom_target = frag_rep[0]
		descendents_target = frag_rep[1]
	else:
		atom_target = frag_rep
		descendents_target = None

	if not isinstance(atom_target, list):
		atom_matches = _simple_match(atom, atom_target)
	else:
		for target in atom_target:
			atom_matches = _simple_match(atom, target)
			if atom_matches:
				break
	
	if not atom_matches:
		return None
	
	if descendents_target == None:
		return [[atom]]
	
	# 'descendents_target' should be a sequence that matches the 
	# atoms connected to 'atom' (possibly skipping 'parent', if specified)

	# since 'R' and None may match 0 or 1 atoms, can't check for exact
	# match between number of bonds and number of descendents
	bonds_to_match = atom.num_bonds - (parent != None)
	possible_Hs = descendents_target.count(R) + \
		descendents_target.count(None) + descendents_target.count(H)
	if len(descendents_target) < bonds_to_match \
	or len(descendents_target) - possible_Hs > bonds_to_match:
		return None
	
	if len(descendents_target) == 0 and bonds_to_match == 0:
		return [[atom]]
	matches = _match_descendents(atom, atom.neighbors, parent, descendents_target[:], {})
	
	if matches == None:
		return None
	
	for i in range(len(matches)):
		matches[i] = [atom] + matches[i]

	return matches

def _simple_match(atom, target):
	if isinstance(target, basestring):
		if target == atom.idatm_type:
			return True
	elif isinstance(target, int):
		if target == atom.element.number:
			return True
	elif isinstance(target, dict):
		idatm_type = atom.idatm_type
		if not type_info.has_key(idatm_type):
			# uncommon type
			if target.has_key('default'):
				return target['default']
			return False
		if 'not_type' in target:
			if idatm_type in target['not_type']:
				return False
		info = type_info[idatm_type]
		for key, value in target.items():
			if key == 'default' or key == 'not_type':
				continue
			if getattr(info, key) != value:
				return False
		return True
	elif target is None:
		return True
	elif isinstance(target, RingAtom):
		return _simple_match(atom, target.atom_desc) \
				and len(atom.minimumRings()) == target.num_rings
	else:
		# tuple
		for subtarget in target:
			if _simple_match(atom, subtarget):
				return True
	return False

def _match_descendents(atom, neighbors, parent, descendents, prevAssigned):
	# prevAssigned is a dictionary that notes what assignments of atoms
	# types to fragment descriptions have previously occurred and is used
	# to try to avoid multiply matching indistinguishable fragments with
	# the same set of atoms
	matches = []
	target = descendents[0]
	alternatives = descendents[1:]
	possible_Hs = descendents.count(R) + descendents.count(H) \
						+ descendents.count(None)

	prevAssigned = prevAssigned.copy()
	bonds_to_match = len(neighbors) - (parent != None)

	if len(descendents) < bonds_to_match \
	  or len(descendents) - possible_Hs > bonds_to_match:
		return None

	for otherAtom in neighbors:
		if otherAtom == parent:
			continue
		if prevAssigned.has_key(otherAtom):
			skipAtom = False
			for fragment in prevAssigned[otherAtom]:
				if _fragCompare(fragment, target):
					skipAtom = True
					break
			if skipAtom:
				continue
		possibleMatches = _trace_group(otherAtom, atom, target)
		if possibleMatches != None:
			if len(alternatives) == 0:
				remainderMatches = [[]]
			else:
				remainNeighbors = neighbors[:]
				remainNeighbors.remove(otherAtom)
				remainderMatches = _match_descendents(atom,
				  remainNeighbors, parent,
				  alternatives, prevAssigned)
			if remainderMatches != None:
				for match1 in possibleMatches:
					for match2 in remainderMatches:
						matches.append(match1 + match2)
			if prevAssigned.has_key(otherAtom):
				# don't use append here!
				prevAssigned[otherAtom] = prevAssigned[otherAtom] + [target]
			else:
				prevAssigned[otherAtom] = [target]

	possible_Hs = alternatives.count(R) + alternatives.count(H) \
						+ alternatives.count(None)
	if (target == R or target == H or target == None) \
	  and len(alternatives) >= bonds_to_match \
	  and len(alternatives) - possible_Hs <= bonds_to_match:
		# since 'R'/None may be hydrogen, and hydrogen is missing
		# from the structure, check if the group matches while
		# omitting the 'R'/None (or H)
		if len(alternatives) == 0 and bonds_to_match == 0:
			matches.append([])
		else:
			remainderMatches = _match_descendents(atom, neighbors[:],
			  parent, alternatives, prevAssigned)
			if remainderMatches != None:
				for match in remainderMatches:
					matches.append(match)

	if len(matches) > 0:
		return matches
	
	return None

def _fragCompare(frag1, frag2):
	if type(frag1) != type(frag2):
		if isinstance(frag1, tuple) and frag2 in frag1:
			return True
		if isinstance(frag2, tuple) and frag1 in frag2:
			return True
		return False
	if isinstance(frag1, (tuple, list)):
		if len(frag1) != len(frag2):
			return False
		for i in range(len(frag1)):
			if isinstance(frag1[i], (tuple, list)):
				if not _fragCompare(frag1[i], frag2[i]):
					return False
			else:
				if frag1[i] != frag2[i]:
					return False
		return True
	return frag1 == frag2

# register selectors with selection manager...
selCategory = 'functional group'
nucSubcategory = 'nucleoside base'
nucBases = ('adenine', 'cytosine', 'guanine', 'thymine', 'uracil')
amineSubcategory = 'amine'
try:
	from chimera.selection.manager import selMgr, SortString
except ImportError:
	pass
else:
	nucSels = {}
	amineSels = {}
	for group in group_info.keys():
		selectorText = """\
import %s
from chimera.misc import bonds

all = []
for group in %s.find_group('%s',structures):
	# use 'bonds' instead of sel.addImplied 
	# to avoid adding bonds between adjacent groups
	all.extend(group)
	all.extend(bonds(group))
	sel.add(all)
""" % (__name__,  __name__, group)
		if group in nucBases:
			nucSels[SortString(group)] = selectorText
		elif group[-5:] == "amine":
			if group == "amine":
				key = "all"
			else:
				key = group[:-6]
			amineSels[key] = (selectorText, group_info[group][0])
		else:
			selMgr.addSelector(__name__, [selMgr.CHEMISTRY,
					selCategory, group], selectorText,
					description=group_info[group][0])
	selAll = None
	for ns in nucSels.values():
		if selAll is None:
			selAll = ns
		else:
			selAll = selMgr.doOp('EXTEND', selAll, ns)
	nucSels[SortString('all of the above', 1)] = selAll
	for group in nucSels.keys():
		if group_info.has_key(group):
			description = group_info[group][0]
		else:
			description = None
		selMgr.addSelector(__name__, [selMgr.CHEMISTRY, selCategory,
				nucSubcategory, group], nucSels[group],
				description=description)
	for amine in amineSels.keys():
		sortVal = 0
		if "aliphatic" in amine:
			sortVal += 1
		if "aromatic" in amine:
			sortVal += 6
		for add, substring in enumerate(["primary", "secondary",
					"tertiary", "quaternary"]):
			if substring in amine:
				sortVal += add + 1
		selMgr.addSelector(__name__, [selMgr.CHEMISTRY, selCategory,
			amineSubcategory, SortString(amine, sortVal)],
			amineSels[amine][0], description=amineSels[amine][1])
	selMgr.makeCallbacks()

	del group, nucSels, selAll, ns, description, selMgr, selectorText
	del amineSels, amine, SortString
'''
