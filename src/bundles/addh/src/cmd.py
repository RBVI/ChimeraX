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

from .util import complete_terminal_carboxylate, determine_terminii, determine_naming_schemas
from chimerax.core.atomic import Element
from chimerax.core.atomic.struct_edit import add_atom
from chimerax.core.atomic.colors import element_colors

def cmd_addh(session, structures=None, hbond=True, in_isolation=True, use_his_name=True,
    use_glu_name=True, use_asp_name=True, use_lys_name=True, use_cys_name=True):

    if structures is None:
        from chimerax.core.atomic import AtomicStructure
        structures = [m for m in session.models if isinstance(m, AtomicStructure)]
        from chimerax.core.atomic import AtomicStructures
        struct_collection = AtomicStructures(structures)
    else:
        struct_collection = structures
        structures = list(structures)

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
    for structure in structures:
        structure.alt_loc_change_notify = False
    atoms = struct_collection.atoms
    num_pre_hs = len(atoms.filter(atoms.elements.numbers == 1))
    # at this time, Atom.scene_coord is *so* much slower then .coord (50x),
    # that we use this hack to use .coord if possible
    from chimerax.core.atomic import Atom
    Atom._addh_coord = Atom.coord if in_isolation else Atom.scene_coord
    try:
        add_h_func(session, structures, in_isolation=in_isolation, **prot_schemes)
    finally:
        delattr(Atom, "_addh_coord")
        for structure in structures:
            structure.alt_loc_change_notify = True
    atoms = struct_collection.atoms
    # If side chains are displayed, then the CA is _not_ hidden, so we
    # need to let the ribbon code update the hide bits so that the CA's
    # hydrogen gets hidden...
    atoms.update_ribbon_visibility()
    session.logger.info("%s hydrogens added" %
        (len(atoms.filter(atoms.elements.numbers == 1)) - num_pre_hs))
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
    from .simple import add_hydrogens
    atoms, type_info_for_atom, naming_schemas, idatm_type, hydrogen_totals, his_Ns, coordinations, \
        fake_N, fake_C = _prep_add(session, structures, unknowns_info, **prot_schemes)
    _make_shared_data(session, structures, in_isolation)
    invert_xforms = {}
    for atom in atoms:
        if atom not in type_info_for_atom:
            continue
        bonding_info = type_info_for_atom[atom]
        try:
            invert = invert_xforms[atom.structure]
        except KeyError:
            invert_xforms[atom.structure] = invert = atom.structure.scene_position.inverse()

        add_hydrogens(atom, bonding_info, (naming_schemas[atom.residue],
            naming_schemas[atom.structure]), hydrogen_totals[atom],
            idatm_type, invert, coordinations.get(atom, []))
    post_add(session, fake_N, fake_C)
    _delete_shared_data()
    session.logger.status("Hydrogens added")

class IdatmTypeInfo:
    def __init__(self, geometry, substituents):
        self.geometry = geometry
        self.substituents = substituents
from chimerax.core.atomic import idatm
type_info = {}
for element_num in range(1, Element.NUM_SUPPORTED_ELEMENTS):
    e = Element.get_element(element_num)
    if e.is_metal or e.is_halogen:
        type_info[e.name] = IdatmTypeInfo(idatm.single, 0)
type_info.update(idatm.type_info)

def post_add(session, fake_n, fake_c):
    # fix up non-"true" terminal residues (terminal simply because
    # next residue is missing)
    for fn in fake_n:
        n = fn.find_atom("N")
        ca = fn.find_atom("CA")
        c = fn.find_atom("C")
        if not n or not ca or not c:
            continue
        dihed = None
        for cnb in c.neighbors:
            if cnb.name == "N":
                pn = cnb
                break
        else:
            dihed = 0.0
        if dihed is None:
            pr = pn.residue
            pc = pr.find_atom("C")
            pca = pr.find_atom("CA")
            if pr.name == "PRO":
                ph = pr.find_atom("CD")
            else:
                ph = pr.find_atom("H")
            if not pc or not pca or not ph:
                dihed = 0.0
        add_nh = True
        for nb in n.neighbors:
            if nb.element.number == 1:
                if nb.name == "H":
                    add_nh = False
                else:
                    nb.structure.delete_atom(nb)
        if fn.name == "PRO":
            n.idatm_type = "Npl"
            continue
        if add_nh:
            if dihed is None:
                from chimerax.core.geometry import dihedral
                dihed = dihedral(pc.coord, pca.coord, pn.coord, ph.coord)
            session.logger.info("Adding 'H' to %s" % str(fn))
            from chimerax.core.atomic.struct_edit import add_dihedral_atom
            h = add_dihedral_atom("H", "H", n, ca, c, 1.01, 120.0, dihed, bonded=True)
            h.color = determine_h_color(n)
        # also need to set N's IDATM type, because if we leave it as
        # N3+ then the residue will be identified by AddCharge as
        # terminal and there will be no charge for the H atom
        n.idatm_type = "Npl"

    for fc in fake_c:
        c = fc.find_atom("C")
        if not c:
            continue
        for nb in c.neighbors:
            if nb.element.number == 1:
                session.logger.info("%s is not terminus, removing H atom from 'C'" % str(fc))
                nb.structure.delete_atom(nb)
        # the N proton may have been named 'HN'; fix that
        hn = fc.find_atom("HN")
        if not hn:
            continue
        n = hn.neighbors[0]
        h = add_atom("H", "H", fc, hn.coord, serial_number=hn.serial_number, bonded_to=n)
        h.color = determine_h_color(n)
        h.hide = n.hide
        fc.structure.delete_atom(hn)

def _acid_check(r, protonation, res_types, atom_names):
    if protonation == res_types[0]:
        protonation = 0
    elif protonation == res_types[1]:
        protonation = 2
    one_name, two_name = atom_names
    ones = [r.find_atom(one_name)]
    twos = [r.find_atom(two_name)]
    if not ones[0] or not twos[0]:
        return False
    if protonation == 1:
        prot_check = ones
        deprot_check = twos
    elif protonation == 2:
        prot_check = twos
        deprot_check = ones
    else:
        prot_check = []
        deprot_check = ones + twos
    can_prot = True
    for oxy in prot_check:
        if oxy.num_bonds != 1:
            can_prot = False
            break
    if not can_prot:
        return False
    can_deprot = True
    for oxy in deprot_check:
        if oxy.num_bonds != 1:
            can_deprot = False
            break
    if not can_deprot:
        return False
    return protonation, prot_check, deprot_check

def _asp_check(r, protonation=None):
    if protonation == None:
        protonation = r.name
    return _acid_check(r, protonation, asp_res_names, asp_prot_names)

def _glu_check(r, protonation=None):
    if protonation == None:
        protonation = r.name
    return _acid_check(r, protonation, glu_res_names, glu_prot_names)

def _lys_check(r, protonation=None):
    if protonation == None:
        protonation = r.name
    nz = r.find_atom("NZ")
    if not nz:
        return False
    prot_okay = True
    if len([nb for nb in nz.neighbors if nb.element.number > 1]) != 1:
        prot_okay = False
    elif protonation == "LYS" and nz.num_bonds > 3:
        prot_okay = False
    elif nz.num_bonds > 2:
        prot_okay = False
    return prot_okay

def _cys_check(r, protonation=None):
    if protonation == None:
        protonation = r.name
    sg = r.find_atom("SG")
    if not sg:
        return False
    prot_okay = True
    if len([nb for nb in sg.neighbors if nb.element.number > 1]) != 1:
        prot_okay = False
    elif sg.num_bonds > 1:
        prot_okay = False
    return prot_okay

def _handle_acid_protonation_scheme_item(r, protonation, res_types, atom_names,
        type_info, type_info_for_atom):
    protonation, prot_check, deprot_check = _acid_check(r, protonation, res_types, atom_names)
    tiO3 = type_info['O3']
    tiO2 = type_info['O2']
    tiO2minus = type_info['O2-']
    for oxy in prot_check:
        type_info_for_atom[oxy] = tiO3
    if protonation:
        deprot_type = tiO2
    else:
        deprot_type = tiO2minus
    for oxy in deprot_check:
        type_info_for_atom[oxy] = deprot_type

_tree_dist = 3.25
_metal_dist = 3.6
h_rad = 1.0
def _make_shared_data(session, protonation_models, in_isolation):
    from chimerax.core.geometry import AdaptiveTree, distance_squared
    # since adaptive search tree is static, it will not include
    # hydrogens added after this; they will have to be found by
    # looking off their heavy atoms
    global search_tree, _radii, _metals, ident_pos_models, _h_coloring
    _radii = {}
    xyzs = []
    vals = []
    metal_xyzs = []
    metal_vals = []
    # if we're adding hydrogens to unopen models, add those models to open models...
    pm_set = set(protonation_models)
    if in_isolation:
        models = pm_set
    else:
        from chimerax.core.atomic import AtomicStructure
        om_set = set([m for m in session.models if isinstance(m, AtomicStructure)])
        models = om_set | pm_set
    # consider only one copy of identically-positioned models...
    ident_pos_models = {}
    for pm in protonation_models:
        for m in models:
            if m == pm:
                continue
            if pm.num_atoms != m.num_atoms:
                continue
            for a1, a2 in zip(pm.atoms[:3], m.atoms[:3]):
                if distance_squared(a1._addh_coord, a2._addh_coord) > 0.00001:
                    break
            else:
                ident_pos_models.setdefault(pm, set()).add(m)

    for m in models:
        if m not in ident_pos_models:
            ident_pos_models[m] = set()
        for a in m.atoms:
            xyzs.append(a._addh_coord)
            vals.append(a)
            _radii[a] = a.radius
            if a.element.is_metal:
                metal_xyzs.append(a.coord)
                metal_vals.append(a)
    search_tree = AdaptiveTree(xyzs, vals, _tree_dist)
    _metals = AdaptiveTree(metal_xyzs, metal_vals, _metal_dist)
    from weakref import WeakKeyDictionary
    _h_coloring = WeakKeyDictionary()

def _delete_shared_data():
    global search_tree, _radii, _metals, ident_pos_models
    search_tree = radii = _metals = ident_pos_models = None

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
            if atom.idatm_type in type_info:
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
            if atom in unknowns_info:
                type_info_for_atom[atom] = unknowns_info[atom]
                atoms.append(atom)
                continue
            logger.info("Unknown hybridization for atom (%s) of residue type %s" %
                    (atom.name, atom.residue.name))
        naming_schemas.update(determine_naming_schemas(struct, type_info_for_atom))

    if need_all:
        from chimerax.core.atomic import AtomicStructure
        for struct in [m for m in session.models if isinstance(m, AtomicStructure)]:
            if struct in structures:
                continue
            for atom in struct.atoms:
                idatm_type[atom] = atom.idatm_type
                if atom.idatm_type in type_info:
                    type_info_for_atom[atom] = type_info[atom.idatm_type]

    for atom in atoms:
        if atom not in type_info_for_atom:
            continue
        bonding_info = type_info_for_atom[atom]
        total_hydrogens = bonding_info.substituents - atom.num_bonds
        for bonded in atom.neighbors:
            if bonded.element.number == 1:
                total_hydrogens += 1
        hydrogen_totals[atom] = total_hydrogens

    schemes = {}
    for scheme_type, res_names, res_check, typed_atoms in [
            ('his', ["HID", "HIE", "HIP"], None, []),
            ('asp', asp_res_names, _asp_check, asp_prot_names),
            ('glu', glu_res_names, _glu_check, glu_prot_names),
            ('lys', lys_res_names, _lys_check, lys_prot_names),
            ('cys', cys_res_names, _cys_check, cys_prot_names) ]:
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

def find_nearest(pos, atom, exclude, check_dist, avoid_metal_info=None):
    nearby = search_tree.search_tree(pos, check_dist)
    near_pos = n = near_atom = None
    exclude_pos = set([tuple(ex._addh_coord) for ex in exclude])
    exclude_pos.add(tuple(atom._addh_coord))
    from chimerax.core.geometry import distance
    for nb in nearby:
        n_pos = nb._addh_coord
        if tuple(n_pos) in exclude_pos:
            # excludes identical models also...
            continue
        if nb.structure != atom.structure and (
                atom.structure.id is None or
                (len(nb.structure.id) > 1 and (nb.structure.id[:-1] == atom.structure.id[:-1]))
                or nb.structure in ident_pos_models[atom.structure]):
            # (1) unopen models only "clash" with themselves
            # (2) don't consider atoms in sibling submodels
            continue
        if nb not in atom.neighbors:
            if avoid_metal_info and nb.element.is_metal and nb.structure == atom.structure:
                if metal_clash(nb._addh_coord, pos, atom._addh_coord, atom, avoid_metal_info):
                    return n_pos, 0.0, nb
            d = distance(n_pos, pos) - vdw_radius(nb)
            if near_pos is None or d < n:
                near_pos = n_pos
                n = d
                near_atom = nb
        # only heavy atoms in tree...
        for nbb in nb.neighbors:
            if nbb.element.number != 1:
                continue
            if tuple(nbb._addh_coord) in exclude_pos:
                continue
            n_pos = nbb._addh_coord
            d = distance(n_pos, pos) - h_rad
            if near_pos is None or d < n:
                near_pos = n_pos
                n = d
                near_atom = nbb
    return near_pos, n, near_atom

from chimerax.core.atomic.bond_geom import cos705 as cos70_5
from math import sqrt
sin70_5 = sqrt(1.0 - cos70_5 * cos70_5)
def find_rotamer_nearest(at_pos, idatm_type, atom, neighbor, check_dist):
    # find atom that approaches nearest to a methyl-type rotamer
    n_pos = neighbor._addh_coord
    v = at_pos - n_pos
    try:
        geom = type_info[idatm_type].geometry
    except KeyError:
        geom = 4
    bond_len = bond_with_H_length(atom, geom)
    from numpy.linalg import norm
    v *= cos70_5 * bond_len / norm(v)
    center = at_pos + v
    from chimerax.core.geometry import Plane
    plane = Plane(center, normal=v)
    radius = sin70_5 * bond_len
    check_dist += radius

    nearby = search_tree.search_tree(center, check_dist)
    near_pos = n = near_atom = None
    for nb in nearby:
        nb_sc = nb._addh_coord
        from numpy import allclose
        if allclose(nb_sc, at_pos) or allclose(nb_sc, n_pos):
            # exclude atoms from identical-copy structure also...
            continue
        if nb.structure != atom.structure and (
                atom.structure.id is None or
                (len(nb.structure.id) > 1 and (nb.structure.id[:-1] == atom.structure.id[:-1]))
                or nb.structure in ident_pos_models[atom.structure]):
            # (1) unopen models only "clash" with themselves
            # (2) don't consider atoms in sibling submodels
            continue
        candidates = [(nb, vdw_radius(nb))]
        # only heavy atoms in tree...
        for nbb in nb.neighbors:
            if nbb.element.number != 1:
                continue
            if nbb == neighbor:
                continue
            candidates.append((nbb, h_rad))

        for candidate, a_rad in candidates:
            c_pos = candidate._addh_coord
            # project into plane...
            proj = plane.nearest(c_pos)

            # find nearest approach of circle...
            cv = proj - center
            if not cv.any(): # all elements are zero
                continue
            cv *= radius / norm(cv)
            app = center + cv
            d = norm(c_pos - app) - a_rad
            if near_pos is None or d < n:
                near_pos = c_pos
                n = d
                near_atom = candidate
    return near_pos, n, near_atom

def roomiest(positions, attached, check_dist, atom_type_info):
    pos_info =[]
    for i in range(len(positions)):
        pos = positions[i]
        if isinstance(attached, list):
            atom = attached[i]
            val = (atom, [pos])
        else:
            atom = attached
            val = pos
        if callable(atom_type_info):
            info = atom_type_info(atom)
        else:
            info = atom_type_info
        near_pos, nearest, near_a = find_nearest(pos, atom, [], check_dist, avoid_metal_info=info)
        if nearest is None:
            nearest = check_dist
        pos_info.append((nearest, val))
    pos_info.sort(key=lambda a: a[0], reverse=True)
    # return a list of the values...
    return list(zip(*pos_info))[1]

def metal_clash(metal_pos, pos, parent_pos, parent_atom, parent_type_info):
    if parent_atom.element.valence < 5:
        # carbons, et al, can't coordinate metals
        return False
    if parent_type_info.geometry == 3 and len(
            [a for a in parent_atom.neighbors if a.element.number > 1]) < 2:
        return False
    from chimerax.core.geometry import distance, angle
    if distance(metal_pos, parent_pos) > _metal_dist:
        return False
    # 135.0 is not strict enough (see :1004.a in 1nyr)
    if angle(parent_pos, pos, metal_pos) > 120.0:
        return True
    return False

def vdw_radius(atom):
    # to avoid repeated IDATM computation in the middle of hydrogen addition
    return _radii.get(atom, h_rad)

N_H = 1.01
def bond_with_H_length(heavy, geom):
    element = heavy.element.name
    if element == "C":
        if geom == 4:
            return 1.09
        if geom == 3:
            return 1.08
        if geom == 2:
            return 1.056
    elif element == "N":
        return N_H
    elif element == "O":
        # can't rely on water being in chain "water" anymore...
        if heavy.num_bonds == 0 or heavy.num_bonds == 2 \
        and len([nb for nb in heavy.neighbors if nb.element.number > 1]) == 0:
            return 0.9572
        return 0.96
    elif element == "S":
        return 1.336
    return Element.bond_length(heavy.element, Element.get_element(1))

def new_hydrogen(parent_atom, h_num, total_hydrogens, naming_schema, pos, parent_type_info,
        alt_loc):
    global _serial, _metals
    nearby_metals = _metals.search_tree(pos, _metal_dist)
    for metal in nearby_metals:
        if metal.structure != parent_atom.structure:
            continue
        metal_pos = metal.coord
        parent_pos = parent_atom.coord
        if metal_clash(metal_pos, pos, parent_pos, parent_atom, parent_type_info):
            return
    new_h = add_atom(_h_name(parent_atom, h_num, total_hydrogens, naming_schema), "H",
        parent_atom.residue, pos, serial_number=_serial, bonded_to=parent_atom, alt_loc=alt_loc)
    _serial = new_h.serial_number + 1
    new_h.color = determine_h_color(parent_atom)
    new_h.hide = parent_atom.hide
    import sys
    return new_h

def determine_h_color(parent_atom):
    global _h_coloring
    res = parent_atom.residue
    if res.name in res.water_res_names:
        return element_colors(1)
    if parent_atom.structure in _h_coloring:
        color_scheme = _h_coloring[parent_atom.structure]
    else:
        parent_color = parent_atom.color
        from numpy import allclose
        if allclose(parent_color, element_colors(parent_atom.element.number)):
            color_scheme = "element"
        else:
            color_scheme = "parent"
        _h_coloring[parent_atom.structure] = color_scheme
    return parent_atom.color if color_scheme == "parent" else element_colors(1)

naming_exceptions = {
    'ATP': {
        "N6": ["HN61", "HN62"],
    },
    'ADP': {
        "N6": ["HN61", "HN62"],
    },
    'GTP': {
        "N1": ["HN1"],
        "N2": ["HN21", "HN22"]
    },
    'GDP': {
        "N1": ["HN1"],
        "N2": ["HN21", "HN22"]
    }
}

def _h_name(atom, h_num, total_hydrogens, naming_schema):
    res_name = atom.residue.name
    find_atom = atom.residue.find_atom

    res_schema, pdb_version = naming_schema
    if res_name in naming_exceptions and atom.name in naming_exceptions[res_name]:
        except_names = naming_exceptions[res_name][atom.name]
        for name in except_names:
            h_name = name
            if not find_atom(h_name):
                return h_name
        else:
            raise ValueError("All hydrogen names for %s taken!" % atom)
    elif res_schema == "prepend" or len(atom.name) < len(atom.element.name):
        h_name = "H" + atom.element.name
    elif atom in res_schema:
        h_name = "H" + atom.name
    else:
        h_name = "H" + atom.name[len(atom.element.name):]

    while len(h_name) + (total_hydrogens>1) > 4:
        if h_name.isalnum():
            h_name = h_name[:-1]
        else:
            h_name = "".join([x for x in h_name if x.isalnum()])

    if pdb_version == 2:
        if total_hydrogens > 1 or find_atom(h_name):
            while find_atom("%d%s" % (h_num, h_name)):
                h_num += 1
            h_name = "%d%s" % (h_num, h_name)
    elif h_name[-1] == "'" and len(h_name) + (total_hydrogens-1) <= 4:
        while find_atom(h_name):
            h_name += "'"
    elif total_hydrogens > 1 or find_atom(h_name):
        if total_hydrogens == 2 and len([nb for nb in atom.neighbors
                        if nb.element.number > 1]) == 2:
            h_num += 1
        while find_atom("%s%d" % (h_name, h_num)):
            h_num += 1
        h_name = "%s%d" % (h_name, h_num)
    return h_name

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, AtomicStructuresArg, BoolArg
    desc = CmdDesc(
        keyword = [('structures', AtomicStructuresArg), ('hbond', BoolArg),
            ('in_isolation', BoolArg), ('use_his_name', BoolArg), ('use_glu_name', BoolArg),
            ('use_asp_name', BoolArg), ('use_lys_name', BoolArg), ('use_cys_name', BoolArg)],
        synopsis = 'Add hydrogens'
    )
    register('addh', desc, cmd_addh, logger=logger)
