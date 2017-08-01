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

def cmd_addh(session, structures=None, hbond=True, in_isolation=True, use_his_name=True,
    use_glu_name=True, use_asp_name=True, use_lys_name=True, use_cys_name=True):

    if structures is None:
        from chimerax.core.atomic import AtomicStructure
        structures = [m for m in session.models if isinstance(m, AtomicStructure)]

    #add_h_func = hbond_add_hydrogens if hbond else simple_add_hydrogens
    if hbond:
        from chimerax.core.errors import LimitationError
        raise LimitationError("'hbond true' option not yet implemented (use 'hbond false')")
    add_h_func = simple_add_hydrogens

    prot_schemes = {}
    for res_name in ['his', 'glu', 'asp', 'lys', 'cys']:
        use = eval('use_%s_name' % res_name)
        scheme_name = res_name + '_scheme'
        prot_schemes[scheme_name] = None if use else {}
    #if session.ui.is_gui:
    #   initiate_add_hyd...
    #else:
    #   add_f_func(...)
    add_h_func(session, structures, in_isolation=in_isolation, **prot_schemes)
#TODO: determine_terminii, complete_terminal_carboxylate, determine_naming_schemas
#TODO: _prep_add, _make_shared_data, simple.add_hydrogens, post_add, _delete_shared_data
#TODO: hbond_add_hydrogens
#TODO: initiate_add_hyd

def simple_add_hydrogens(session, structures, unknowns_info={}, in_isolation=False, **prot_schemes):
	"""Add hydrogens to given structures using simple geometric criteria

	Geometric info for atoms whose IDATM types don't themselves provide
	sufficient information can be passed via the 'unknowns_info' dictionary.
	The keys are atoms and the values are dictionaries specifying the
	geometry and number of substituents (not only hydrogens) for the atom.

	If 'in_isolation' is True, each model will ignore all others as it is
	protonated.

	'prot_schemes' can contain the following keywords for controlling amino acid
	protonation: his_scheme, glu_scheme, asp_scheme, lys_scheme, and cys_scheme.

	The 'his_scheme' keyword determines how histidines are handled.  If
	it is None (default) then the residue name is expected to be HIE,
	HID, HIP, or HIS indicating the protonation state is epsilon, delta,
	both, or unspecified respectively.  Otherwise the value is a dictionary:
	the keys are histidine residues and the values are HIE/HID/HIP/HIS
	indication of the protonation state.  Histidines not in the
	dictionary will be protonated based on the nitrogens' atom types.

	The other protonation schemes are handled in a similar manner.  The
	residue names are:

	glu: GLU (unprotonated), GLH (carboxy oxygen 2 protonated)
	asp: ASP (unprotonated), ASH (carboxy oxygen 2 protonated)
	lys: LYS (positively charged), LYN (neutral)
	cys: CYS (neutral), CYM (negatively charged)

	For asp/glu, the dictionary values are either 0, 1, or 2: indicating
	no protonation or carboxy oxygen #1 or #2 protonation respectively.

	This routine adds hydrogens immediately even if some atoms have
	unknown geometries.  To allow the user to intervene to specify
	geometries, use the 'initiate_add_hyd' function of the unknownsGUI
	module of this package.
	"""

	if in_isolation and len(structures) > 1:
		for struct in structures:
			simple_add_hydrogens(session, [struct], unknowns_info=unknowns_info,
				in_isolation=in_isolation, **prot_schemes)
		return
	from simple import add_hydrogens
	atoms, type_info_for_atom, naming_schemas, idatm_type, hydrogen_totals, his_Ns, coordinations, \
        fake_N, fake_C = _prepAdd(session, structures, unknowns_info, **prot_schemes)
	_make_shared_data(structures, in_isolation)
	invert_xforms = {}
	for atom in atoms:
		if not type_info_for_atom.has_key(atom):
			continue
		bonding_info = type_info_for_atom[atom]
		try:
			invert = invert_xforms[atom.structure]
		except KeyError:
			invert_xforms[atom.structure] = invert = atom.structure.scene_position.inverse()

		add_hydrogens(atom, bonding_info, (naming_schemas[atom.residue],
			naming_schemas[atom.structure]), hydrogen_totals[atom],
			idatm_type, invert, coordinations.get(atom, []))
	post_add(fake_N, fake_C)
	_delete_shared_data()
	session.logger.status("Hydrogens added")

class IdatmTypeInfo:
	def __init__(self, geometry, substituents):
		self.geometry = geometry
		self.substituents = substituents
type_info = {}
for element_num in range(1, Element.NUM_SUPPORTED_ELEMENTS):
    e = Element.get_element(element_num)
    if e.is_metal or e.is_halogen:
	type_info[e.name] = IdatmTypeInfo(single, 0)
from chimerax.core.atomic import idatm
type_info.update(idatm.type_info)

asp_prot_names, asp_res_names = ["OD1", "OD2"], ["ASP", "ASH"]
glu_prot_names, glu_res_names = ["OE1", "OE2"], ["GLU", "GLH"]
lys_prot_names, lys_res_names = ["NZ"], ["LYS", "LYN"]
cys_prot_names, cys_res_names = ["SG"], ["CYS", "CYM"]
def _prep_add(session, structures, unknowns_info, need_all=False, **prot_schemes):
	global _serial
	_serial = None
	atoms = []
	type_info_for_atom = {}
	naming_schemas = {}
	idatm_type = {} # need this later; don't want a recomp
	hydrogen_totals= {}

	# add missing OXTs of "real" C terminii;
	# delete hydrogens of "fake" N terminii after protonation
	# and add a single "HN" back on, using same dihedral as preceding residue;
	# delete extra hydrogen of "fake" C terminii after protonation
    logger = session.logger
	real_N, real_C, fake_N, fake_C = determine_terminii(structures)
	logger.info("Chain-initial residues that are actual N"
		" terminii: %s\n" % ", ".join([str(r) for r in real_N]))
	logger.info("Chain-initial residues that are not actual N"
		" terminii: %s\n" % ", ".join([str(r) for r in fake_N]))
	logger.info("Chain-final residues that are actual C"
		" terminii: %s\n" % ", ".join([str(r) for r in real_C]))
	logger.info("Chain-final residues that are not actual C"
		" terminii: %s\n" % ", ".join([str(r) for r in fake_C]))
	for rc in real_C:
		complete_terminal_carboxylate(rc)

	# ensure that N terminii are protonated as N3+ (since Npl will fail)
	for nter in real_N+fake_N:
        n = nter.find_atom("N")
        if not n:
            continue
		if not (n.residue.name == "PRO" and n.num_bonds >= 2):
			n.idatm_type = "N3+"

	coordinations = {}
    for struct in structures:
        pbg = struct.pseudobond_group(struct.PBG_METAL_COORDINATION, create_type=None)
        if not pbg:
            continue
		for pb in pbg.pseudobonds:
			for a in pb.atoms:
				if not need_all and a.structure not in structures:
					continue
				if not a.element.is_metal:
					coordinations.setdefault(a, []).append(pb.other_atom(a))

	for struct in structures:
		for atom in struct.atoms:
			if atom.element.number == 0:
				res = atom.residue
				struct.delete_atom(atom)
				if not res.atoms:
					struct.delete_residue(res)
		for atom in struct.atoms:
			idatm_type[atom] = atom.idatm_type
			if type_info.has_key(atom.idatm_type):
				# don't want to ask for idatm_type in middle
				# of hydrogen-adding loop (since that will
				# force a recomp), so remember here
				type_info_for_atom[atom] = type_info[atom.idatm_type]
				atoms.append(atom)
				# sulfonamide nitrogens coordinating a metal
				# get an additional hydrogen stripped
				if coordinations.get(atom, []) and atom.element.name == "N":
					if "Son" in [nb.idatm_type for nb in atom.neighbors]:
						from copy import copy
						ti = copy(type_info[atom.idatm_type])
						ti.substituents -= 1
						type_info_for_atom[atom] = ti
				continue
			if unknowns_info.has_key(atom):
				type_info_for_atom[atom] = unknowns_info[atom]
				atoms.append(atom)
				continue
			logger.info("Unknown hydridization for atom (%s) of residue type %s" %
					(atom.name, atom.residue.name))
		naming_schemas.update(determine_naming_schemas(struct, type_info_for_atom))

	if need_all:
        from chimerax.core.atomic import AtomicStructure
		for struct in [m for m in session.models if isinstance(m, AtomicStructure)]:
			if struct in structures:
				continue
			for atom in struct.atoms:
				idatm_type[atom] = atom.idatm_type
				if type_info.has_key(atom.idatm_type):
					type_info_for_atom[atom] = type_info[atom.idatm_type]

	for atom in atoms:
		if not type_info_for_atom.has_key(atom):
			continue
		bonding_info = type_info_for_atom[atom]
		totalHydrogens = bonding_info.substituents - atom.num_bonds
		for bonded in atom.neighbors:
			if bonded.element.number == 1:
				total_hydrogens += 1
		hydrogen_totals[atom] = total_hydrogens
    #TODO

	for schemeType, resNames, resCheck, typedAtoms in [
			('his', ["HID", "HIE", "HIP"], None, []),
			('asp', asp_res_names, _aspCheck, asp_prot_names),
			('glu', glu_res_names, _gluCheck, glu_prot_names),
			('lys', lys_res_names, _lysCheck, lys_prot_names),
			('cys', cys_res_names, _cysCheck, cys_prot_names) ]:
		scheme = prot_schemes.get(schemeType + 'Scheme', None)
		if scheme is None:
			byName = True
			scheme = {}
		else:
			byName = False
		if not scheme:
			for s in structures:
				for r in s.residues:
					if r.type in resNames and resCheck and resCheck(r):
						if byName:
							scheme[r] = r.type
						elif schemeType != 'his':
							scheme[r] = resNames[0]
						# unset any explicit typing...
						for ta in typedAtoms:
							for a in r.atomsMap[ta]:
								a.idatm_type = None
		else:
			for r in scheme.keys():
				if resCheck and not resCheck(r, scheme[r]):
					del scheme[r]
		exec("%sScheme = scheme" % schemeType)
	# create dictionary keyed on histidine residue with value of another
	# dictionary keyed on the nitrogen atoms with boolean values: True
	# equals should be protonated
	hisNs = {}
	for r, protonation in hisScheme.items():
		try:
			delta = r.atomsMap["ND1"][0]
			epsilon = r.atomsMap["NE2"][0]
		except KeyError:
			# find the ring, etc.
			rings = r.molecule.minimumRings()
			for ring in rings:
				if ring.atoms.pop().residue == r:
					break
			else:
				continue
			# find CG by locating CB-CG bond
			ringBonds = ring.bonds
			for ra in ring.atoms:
				if ra.element.name != "C":
					continue
				for ba, b in ra.bondsMap.items():
					if ba.element.name == "C" \
					and b not in ringBonds:
						break
				else:
					continue
				break
			else:
				continue
			nitrogens = [a for a in ring.atoms
						if a.element.name == "N"]
			if len(nitrogens) != 2:
				continue
			if ra in nitrogens[0].neighbors:
				delta, epsilon = nitrogens
			else:
				epsilon, delta = nitrogens
		if protonation == "HID":
			hisNs.update({ delta: True, epsilon: False })
		elif protonation == "HIE":
			hisNs.update({ delta: False, epsilon: True })
		elif protonation == "HIP":
			hisNs.update({ delta: True, epsilon: True })
		else:
			continue
	for n, doProt in hisNs.items():
		if doProt:
			type_info_for_atom[n] = type_info["Npl"]
			n.idatm_type = idatm_type[n] = "Npl"
		else:
			type_info_for_atom[n] = type_info["N2"]
			n.idatm_type = idatm_type[n] = "N2"

	for r, protonation in aspScheme.items():
		_handleAcidProtonationSchemeItem(r, protonation, asp_res_names,
			asp_prot_names, type_info, type_info_for_atom)

	for r, protonation in gluScheme.items():
		_handleAcidProtonationSchemeItem(r, protonation, glu_res_names,
			glu_prot_names, type_info, type_info_for_atom)

	for r, protonation in lysScheme.items():
		nzs = r.atomsMap["NZ"]
		if protonation == "LYS":
			it = 'N3+'
		else:
			it = 'N3'
		ti = type_info[it]
		for nz in nzs:
			type_info_for_atom[nz] = ti
			# avoid explicitly setting type if possible
			if nz.idatm_type != it:
				nz.idatm_type = it

	for r, protonation in cysScheme.items():
		sgs = r.atomsMap["SG"]
		if protonation == "CYS":
			it = 'S3'
		else:
			it = 'S3-'
		ti = type_info[it]
		for sg in sgs:
			type_info_for_atom[sg] = ti
			# avoid explicitly setting type if possible
			if sg.idatm_type != it:
				sg.idatm_type = it

	return atoms, type_info_for_atom, naming_schemas, idatm_type, \
			hydrogen_totals, hisNs, coordinations, fake_N, fake_C

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, AtomicStructuresArg, BoolArg
    desc = CmdDesc(
        keyword = [('structures', AtomicStructuresArg), ('hbond', BoolArg),
            ('in_isolation', BoolArg), ('use_his_name', BoolArg), ('use_glu_name', BoolArg),
            ('use_asp_name', BoolArg), ('use_lys_name', BoolArg), ('use_cys_name', BoolArg)],
        synopsis = 'Add hydrogens'
    )
    register('addh', desc, cmd_addh, logger=logger)
