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
#TODO: complete_terminal_carboxylate, determine_naming_schemas
#TODO: _handle_acid_protonation_scheme_item
#TODO: _make_shared_data, simple.add_hydrogens, post_add, _delete_shared_data
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

def complete_terminal_carboxylate(session, cter):
	from chimerax.core.atomic.bond_geom import bond_positions
    #TODO
	from chimera.molEdit import addAtom
	if cter.find_atom("OXT"):
		return
    c = cter.find_atom("C")
	if c:
		if c.num_bonds != 2:
			return
		loc = bond_positions(c.coord, 3, 1.229, [n.coord for n in c.neighbors])[0]
		oxt = addAtom("OXT", chimera.Element("O"), cter, loc,
								bondedTo=c)
        session.logger.info("Missing OXT added to C-terminal residue %s" % str(cter))

def determine_terminii(session, structs):
	real_N = []
	real_C = []
	fake_N = []
	fake_C = []
    logger = session.logger
	for s in structs:
		sr_res = set()
		for chain in s.chains:
			if chain.from_seqres:
				sr_res.update(chain.residues)
				rn, rc, fn, fc = terminii_from_seqres(chain)
				logger.info("Terminii for %s determined from SEQRES records" % chain.full_name)
			else:
				rn, rc, fn, fc = guessTerminii(chain)
				if chain.fromSeqres == None:
					logger.info("No SEQRES records for %s;" % chain.full_name, add_newline=False)
				else:
					logger.info("SEQRES records don't match %s;" % chain.full_name,
                        add_newline=False)
				replyobj.info(" guessing terminii instead")
			real_N.extend(rn)
			real_C.extend(rc)
			fake_N.extend(fn)
			fake_C.extend(fc)
		if sr_res:
			# Look for peptide terminii not in SEQRES records
            from chimerax.core.atomic import Sequence
            protein3to1 = Sequence.protein3to1
			for r in s.residues:
				if r in sr_res:
					continue
				if protein3to1(r.name) == 'X':
					continue
                ca = r.find_atom("CA")
                o = r.find_atom("O")
                n = r.find_atom("N")
                c = r.find_atom("C")
				if ca and o and n and c:
					for atom_name, terminii in [('N', real_N), ('C', real_C)]:
						for nb in r.find_atom(atom_name).neighbors:
							if nb.residue != r:
								break
						else:
							terminii.append(r)

	return real_N, real_C, fake_N, fake_C

def terminii_from_seqres(chain):
	real_N = []
	real_C = []
	fake_N = []
	fake_C = []
	if chain.residues[0]:
		real_N.append(chain.residues[0])
	if chain.residues[-1]:
		real_C.append(chain.residues[-1])

	last = chain.residues[0]
	for res in chain.residues[1:]:
		if res:
			if not last:
				fake_N.append(res)
		else:
			if last:
				fake_C.append(last)
		last = res
	return real_N, real_C, fake_N, fake_C


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
	real_N, real_C, fake_N, fake_C = determine_terminii(session, structures)
	logger.info("Chain-initial residues that are actual N"
		" terminii: %s" % ", ".join([str(r) for r in real_N]))
	logger.info("Chain-initial residues that are not actual N"
		" terminii: %s" % ", ".join([str(r) for r in fake_N]))
	logger.info("Chain-final residues that are actual C"
		" terminii: %s" % ", ".join([str(r) for r in real_C]))
	logger.info("Chain-final residues that are not actual C"
		" terminii: %s" % ", ".join([str(r) for r in fake_C]))
	for rc in real_C:
		complete_terminal_carboxylate(session, rc)

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

    schemes = {}
	for scheme_type, res_names, res_check, typed_atoms in [
			('his', ["HID", "HIE", "HIP"], None, []),
			('asp', asp_res_names, _aspCheck, asp_prot_names),
			('glu', glu_res_names, _gluCheck, glu_prot_names),
			('lys', lys_res_names, _lysCheck, lys_prot_names),
			('cys', cys_res_names, _cysCheck, cys_prot_names) ]:
		scheme = prot_schemes.get(scheme_type + '_scheme', None)
		if scheme is None:
			by_name = True
			scheme = {}
		else:
			by_name = False
		if not scheme:
			for s in structures:
				for r in s.residues:
					if r.name in res_names and res_check and res_check(r):
						if by_name:
							scheme[r] = r.name
						elif scheme_type != 'his':
							scheme[r] = res_names[0]
						# unset any explicit typing...
						for ta in typed_atoms:
                            a = r.find_atom(ta)
                            if a:
								a.idatm_type = None
		else:
			for r in scheme.keys():
				if res_check and not res_check(r, scheme[r]):
					del scheme[r]
        schemes[scheme_type] = scheme
	# create dictionary keyed on histidine residue with value of another
	# dictionary keyed on the nitrogen atoms with boolean values: True
	# equals should be protonated
	his_Ns = {}
	for r, protonation in schemes["his"].items():
        delta = r.find_atom("ND1")
        epsilon = r.find_atom("NE2")
        if delta is None or epsilon is None:
			# find the ring, etc.
			rings = r.structure.rings()
			for ring in rings:
                if r in rings.atoms.residues:
					break
			else:
				continue
			# find CG by locating CB-CG bond
			ring_bonds = ring.bonds
			for ra in ring.atoms:
				if ra.element.name != "C":
					continue
				for ba, b in zip(ra.neighbors, ra.bonds):
					if ba.element.name == "C" and b not in ring_bonds:
						break
				else:
					continue
				break
			else:
				continue
			nitrogens = [a for a in ring.atoms if a.element.name == "N"]
			if len(nitrogens) != 2:
				continue
			if ra in nitrogens[0].neighbors:
				delta, epsilon = nitrogens
			else:
				epsilon, delta = nitrogens
		if protonation == "HID":
			his_Ns.update({ delta: True, epsilon: False })
		elif protonation == "HIE":
			his_Ns.update({ delta: False, epsilon: True })
		elif protonation == "HIP":
			his_Ns.update({ delta: True, epsilon: True })
		else:
			continue
	for n, do_prot in his_Ns.items():
		if do_prot:
			type_info_for_atom[n] = type_info["Npl"]
			n.idatm_type = idatm_type[n] = "Npl"
		else:
			type_info_for_atom[n] = type_info["N2"]
			n.idatm_type = idatm_type[n] = "N2"

	for r, protonation in schemes["asp"].items():
		_handle_acid_protonation_scheme_item(r, protonation, asp_res_names,
			asp_prot_names, type_info, type_info_for_atom)

	for r, protonation in schemes["glu"].items():
		_handle_acid_protonation_scheme_item(r, protonation, glu_res_names,
			glu_prot_names, type_info, type_info_for_atom)

	for r, protonation in schemes["lys"].items():
		nz = r.find_atom("NZ")
		if protonation == "LYS":
			it = 'N3+'
		else:
			it = 'N3'
		ti = type_info[it]
		if nz is not None:
			type_info_for_atom[nz] = ti
			# avoid explicitly setting type if possible
			if nz.idatm_type != it:
				nz.idatm_type = it

	for r, protonation in schemes["cys"].items():
		sg = r.find_atom("SG")
		if protonation == "CYS":
			it = 'S3'
		else:
			it = 'S3-'
		ti = type_info[it]
		if sg is not None:
			type_info_for_atom[sg] = ti
			# avoid explicitly setting type if possible
			if sg.idatm_type != it:
				sg.idatm_type = it

	return atoms, type_info_for_atom, naming_schemas, idatm_type, \
			hydrogen_totals, his_Ns, coordinations, fake_N, fake_C

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, AtomicStructuresArg, BoolArg
    desc = CmdDesc(
        keyword = [('structures', AtomicStructuresArg), ('hbond', BoolArg),
            ('in_isolation', BoolArg), ('use_his_name', BoolArg), ('use_glu_name', BoolArg),
            ('use_asp_name', BoolArg), ('use_lys_name', BoolArg), ('use_cys_name', BoolArg)],
        synopsis = 'Add hydrogens'
    )
    register('addh', desc, cmd_addh, logger=logger)
