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

from .util import complete_terminal_carboxylate, determine_termini, determine_naming_schemas, \
    bond_with_H_length
from chimerax.atomic import Element, Residue, TmplResidue, Atom
from chimerax.atomic.struct_edit import add_atom
from chimerax.atomic.colors import element_color
from chimerax.atomic.bond_geom import linear

metal_dist_default = 3.95

# functions in .dock_prep may need updating if cmd_addh() call signature changes
def cmd_addh(session, structures, *, hbond=True, in_isolation=True, metal_dist=metal_dist_default,
    template=False, use_his_name=True, use_glu_name=True, use_asp_name=True, use_lys_name=True,
    use_cys_name=True):

    if structures is None:
        from chimerax.atomic import AtomicStructure
        structures = [m for m in session.models if isinstance(m, AtomicStructure)]
        from chimerax.atomic import AtomicStructures
        struct_collection = AtomicStructures(structures)
    else:
        struct_collection = structures
        structures = list(structures)
    if not structures:
        from chimerax.core.errors import UserError
        raise UserError("No structures specified")

    add_h_func = hbond_add_hydrogens if hbond else simple_add_hydrogens

    global _metal_dist
    _metal_dist = metal_dist

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
    Atom._addh_coord = Atom.coord if in_isolation else Atom.scene_coord
    from chimerax.core.logger import Collator
    with Collator(session.logger, "Summary of feedback from adding hydrogens to %s"
            % ("multiple structures" if len(structures) > 1 else structures[0])):
        session.logger.status("Adding hydrogens")
        try:
            add_h_func(session, structures, template=template, in_isolation=in_isolation, **prot_schemes)
        except BaseException:
            session.logger.status("")
            raise
        finally:
            delattr(Atom, "_addh_coord")
            for structure in structures:
                structure.alt_loc_change_notify = True
        session.logger.status("Hydrogens added")
        atoms = struct_collection.atoms
        # If side chains are displayed, then the CA is _not_ hidden, so we
        # need to let the ribbon code update the hide bits so that the CA's
        # hydrogen gets hidden...
        atoms.update_ribbon_backbone_atom_visibility()
        session.logger.info("%s hydrogens added" %
            (len(atoms.filter(atoms.elements.numbers == 1)) - num_pre_hs))
#TODO: initiate_add_hyd

def simple_add_hydrogens(session, structures, *, unknowns_info={}, in_isolation=False,
        template=False, **prot_schemes):
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
    cys: CYS (unspecified), CYM (negatively charged)

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
    atoms, type_info_for_atom, naming_schemas, idatm_type, hydrogen_totals, his_Ns, \
        coordinations, fake_N, fake_C, fake_5p, fake_3p = \
        _prep_add(session, structures, unknowns_info, template, **prot_schemes)
    _make_shared_data(session, structures, in_isolation)
    from chimerax.atomic import Atom
    invert_xforms = {}
    for atom in atoms:
        if atom not in type_info_for_atom:
            continue
        bonding_info = type_info_for_atom[atom]
        if Atom._addh_coord == Atom.coord:
            invert = None
        else:
            try:
                invert = invert_xforms[atom.structure]
            except KeyError:
                invert_xforms[atom.structure] = invert = atom.structure.scene_position.inverse()

        add_hydrogens(atom, bonding_info, (naming_schemas[atom.residue],
            naming_schemas[atom.structure]), hydrogen_totals[atom],
            idatm_type, invert, coordinations.get(atom, []))
    post_add(session, fake_N, fake_C, fake_5p, fake_3p)
    _delete_shared_data()

def hbond_add_hydrogens(session, structures, *, unknowns_info={}, in_isolation=False,
        template=False, **prot_schemes):
    """Add hydrogens to given structures, trying to preserve H-bonding

    Argument are similar to simple_add_hydrogens() except that for the
    'hisScheme' keyword, histidines not in the hisScheme dictionary the
    hydrogen-bond interactions determine the histidine protonation
    """

    if in_isolation and len(structures) > 1:
        for struct in structures:
            hbond_add_hydrogens(session, [struct], unknowns_info=unknowns_info,
                in_isolation=in_isolation, **prot_schemes)
        return
    from .hbond import add_hydrogens
    atoms, type_info_for_atom, naming_schemas, idatm_type, hydrogen_totals, his_Ns, \
        coordinations, fake_N, fake_C, fake_5p, fake_3p = \
        _prep_add(session, structures, unknowns_info, template, **prot_schemes)
    _make_shared_data(session, structures, in_isolation)
    add_hydrogens(session, atoms, type_info_for_atom, naming_schemas, hydrogen_totals,
        idatm_type, his_Ns, coordinations, in_isolation)
    post_add(session, fake_N, fake_C, fake_5p, fake_3p)
    _delete_shared_data()

class IdatmTypeInfo:
    def __init__(self, geometry, substituents):
        self.geometry = geometry
        self.substituents = substituents
from chimerax.atomic import idatm
type_info = {}
for element_num in range(1, Element.NUM_SUPPORTED_ELEMENTS):
    e = Element.get_element(element_num)
    if e.is_metal or e.is_halogen:
        type_info[e.name] = IdatmTypeInfo(idatm.single, 0)
type_info.update(idatm.type_info)

def post_add(session, fake_n, fake_c, fake_5p, fake_3p):
    # Add alt locs to parent atoms that wouldn't otherwise need them so that their
    # alt loc hydrogens can merge into the proper "alt loc pool".  Do it now instead
    # of "at the time" so that unneeded alt locs don't spread from the parent atom.
    for parent_atom, alt_loc_info in parent_alt_locs.items():
        for alt_loc, occupancy in alt_loc_info:
            parent_atom.set_alt_loc(alt_loc, True)
            parent_atom.occupancy = occupancy
    parent_alt_locs.clear()

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
                    if fn.name == "PRO":
                        nb.structure.delete_atom(nb)
                else:
                    nb.structure.delete_atom(nb)
        if fn.name == "PRO":
            n.idatm_type = "Npl"
            continue
        if add_nh:
            if dihed is None:
                from chimerax.geometry import dihedral
                dihed = dihedral(pc.coord, pca.coord, pn.coord, ph.coord)
            session.logger.info("Adding 'H' to %s" % str(fn))
            from chimerax.atomic.struct_edit import add_dihedral_atom
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

    for f5p in fake_5p:
        o5p = f5p.find_atom("O5'")
        if not o5p:
            continue
        for nb in o5p.neighbors:
            if nb.element.number == 1:
                session.logger.info("%s is not terminus, removing H atom from O5'" % str(f5p))
                nb.structure.delete_atom(nb)

    for f3p in fake_3p:
        o3p = f3p.find_atom("O3'")
        if not o3p:
            continue
        for nb in o3p.neighbors:
            if nb.element.number == 1:
                session.logger.info("%s is not terminus, removing H atom from O3'" % str(f3p))
                nb.structure.delete_atom(nb)

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
h_rad = 1.0
def _make_shared_data(session, protonation_models, in_isolation):
    from chimerax.geometry import distance_squared
    from chimerax.atom_search import AtomSearchTree
    # since adaptive search tree is static, it will not include
    # hydrogens added after this; they will have to be found by
    # looking off their heavy atoms
    global search_tree, _radii, _metals, ident_pos_models, _h_coloring, _solvent_atoms
    _radii = {}
    search_atoms = []
    metal_atoms = []
    # if we're adding hydrogens to unopen models, add those models to open models...
    pm_set = set(protonation_models)
    if in_isolation:
        models = pm_set
    else:
        from chimerax.atomic import AtomicStructure
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
            search_atoms.append(a)
            _radii[a] = a.radius
            if a.element.is_metal:
                metal_atoms.append(a)
    from chimerax.atomic import Atom
    use_scene_coords = Atom._addh_coord == Atom.scene_coord
    search_tree = AtomSearchTree(search_atoms, sep_val=_tree_dist, scene_coords=use_scene_coords)
    _metals = AtomSearchTree(metal_atoms, sep_val=max(_metal_dist, 1.0), scene_coords=use_scene_coords)
    from weakref import WeakKeyDictionary
    _h_coloring = WeakKeyDictionary()
    _solvent_atoms = WeakKeyDictionary()

def _delete_shared_data():
    global search_tree, _radii, _metals, ident_pos_model, _h_coloring
    search_tree = radii = _metals = ident_pos_models = _h_coloring = None

asp_res_names, asp_prot_names = ["ASP", "ASH"], ["OD1", "OD2"]
glu_res_names, glu_prot_names = ["GLU", "GLH"], ["OE1", "OE2"]
def _prep_add(session, structures, unknowns_info, template, need_all=False, **prot_schemes):
    global _serial, parent_alt_locs
    _serial = None
    atoms = []
    type_info_for_atom = {}
    naming_schemas = {}
    idatm_type = {} # need this later; don't want a recomp
    hydrogen_totals= {}
    parent_alt_locs = {}

    # add missing OXTs of "real" C termini;
    # delete hydrogens of "fake" N termini after protonation
    # and add a single "HN" back on, using same dihedral as preceding residue;
    # delete extra hydrogen of "fake" C termini after protonation
    logger = session.logger
    real_N, real_C, fake_N, fake_C, fake_5p, fake_3p = determine_termini(session, structures)
    logger.info("Chain-initial residues that are actual N"
        " termini: %s" % ", ".join([str(r) for r in real_N]))
    logger.info("Chain-initial residues that are not actual N"
        " termini: %s" % ", ".join([str(r) for r in fake_N]))
    logger.info("Chain-final residues that are actual C"
        " termini: %s" % ", ".join([str(r) for r in real_C]))
    logger.info("Chain-final residues that are not actual C"
        " termini: %s" % ", ".join([str(r) for r in fake_C]))
    if fake_5p:
        logger.info("Chain-initial residues that are not actual 5'"
            " termini: %s" % ", ".join([str(r) for r in fake_5p]))
    for rc in real_C:
        complete_terminal_carboxylate(session, rc)

    # ensure that normal N termini are protonated as N3+ (since Npl will fail)
    from chimerax.atomic import Sequence
    for nter in real_N+fake_N:
        n = nter.find_atom("N")
        if not n:
            continue
        # if residue wasn't templated, leave atom typing alone
        if Sequence.protein3to1(n.residue.name) == 'X':
            continue
        # if multiple heavy-atom bond partners then this is an unusual N terminus
        # (e.g. FME in 3fil, or any proline)
        if len([nb for nb in n.neighbors if nb.element.number > 1]) < 2:
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

    remaining_unknowns = {}
    type_info_class = type_info['H'].__class__
    for struct in structures:
        for atom in struct.atoms:
            if atom.element.number == 0:
                res = atom.residue
                struct.delete_atom(atom)
        idatm_lookup = {}
        if template:
            template_lookup = {}
            get_template = TmplResidue.get_template
            for res in struct.residues:
                if get_template(res.name):
                    continue
                try:
                    exemplar = template_lookup[res.name]
                except KeyError:
                    from chimerax.mmcif import find_template_residue
                    tmpl = find_template_residue(session, res.name)
                    if not tmpl:
                        continue
                    from chimerax.atomic import AtomicStructure
                    s = AtomicStructure(session)
                    r = exemplar = template_lookup[res.name] = s.new_residue(res.name, 'A', 1)
                    atom_map = {}
                    for ta in tmpl.atoms:
                        if ta.element.number > 1:
                            a = s.new_atom(ta.name, ta.element)
                            a.coord = ta.coord
                            r.add_atom(a)
                            atom_map[ta] = a
                            for tnb in ta.neighbors:
                                if tnb in atom_map:
                                    s.new_bond(a, atom_map[tnb])
                for a in res.atoms:
                    ea = exemplar.find_atom(a.name)
                    if ea:
                        # allow type to change if structure later modified
                        a.set_implicit_idatm_type(ea.idatm_type)
            for r in template_lookup.values():
                r.structure.delete()
            template_lookup.clear()

        for atom in struct.atoms:
            atom_type = atom.idatm_type
            idatm_type[atom] = atom_type
            if atom_type in type_info:
                # don't want to ask for idatm_type in middle
                # of hydrogen-adding loop (since that will
                # force a recomp), so remember here
                type_info_for_atom[atom] = type_info[atom_type]
                # if atom is in standard residue but has missing bonds to
                # heavy atoms, skip it instead of incorrectly protonating
                # (or possibly throwing an error if e.g. it's planar)
                # also
                # UNK/N/DN residues will be missing some or all of their side-chain atoms, so
                # skip atoms that would otherwise be incorrectly protonated due to their
                # missing neighbors
                truncated = \
                        atom.is_missing_heavy_template_neighbors(no_template_okay=True) \
                    or \
                        (atom.residue.name in ["UNK", "N", "DN"] \
                        and atom.residue.polymer_type != Residue.PT_NONE \
                        and unk_atom_truncated(atom)) \
                    or \
                        (atom.residue.polymer_type == Residue.PT_NUCLEIC and atom.name == "P"
                        and atom.num_explicit_bonds < 4)

                if truncated:
                    session.logger.warning("Not adding hydrogens to %s because it is missing heavy-atom"
                        " bond partners" % atom)
                    type_info_for_atom[atom] = type_info_class(4, atom.num_bonds, atom.name)
                else:
                    atoms.append(atom)
                # sulfonamide nitrogens coordinating a metal
                # get an additional hydrogen stripped
                if coordinations.get(atom, []) and atom.element.name == "N":
                    if "Son" in [nb.idatm_type for nb in atom.neighbors]:
                        orig_ti = type_info[atom_type]
                        type_info_for_atom[atom] = orig_ti.__class__(orig_ti.geometry,
                            orig_ti.substituents-1, orig_ti.description)
                continue
            if atom in unknowns_info:
                type_info_for_atom[atom] = unknowns_info[atom]
                atoms.append(atom)
                continue
            remaining_unknowns.setdefault(atom.residue.name, set()).add(atom.name)
            # leave remaining unknown atoms alone
            type_info_for_atom[atom] = type_info_class(4, atom.num_bonds, atom.name)

        for rname, atom_names in remaining_unknowns.items():
            names_text = ", ".join([nm for nm in atom_names])
            atom_text, obj_text = ("atoms", "them") if len(atom_names) > 1 else ("atom", "it")
            logger.warning("Unknown hybridization for %s (%s) of residue type %s;"
                " not adding hydrogens to %s" % (atom_text, names_text, rname, obj_text))
        naming_schemas.update(determine_naming_schemas(struct, type_info_for_atom))

    if need_all:
        from chimerax.atomic import AtomicStructure
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
    # HIS and CYS treated as 'unspecified'; use built-in typing
    for scheme_type, res_names, res_check, typed_atoms in [
            ('his', ["HID", "HIE", "HIP"], None, []),
            ('asp', asp_res_names, _asp_check, asp_prot_names),
            ('glu', glu_res_names, _glu_check, glu_prot_names),
            ('lys', ["LYS", "LYN"], _lys_check, ["NZ"]),
            ('cys', ["CYM"], _cys_check, ["SG"]) ]:
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
            hydrogen_totals, his_Ns, coordinations, fake_N, fake_C, fake_5p, fake_3p

def find_nearest(pos, atom, exclude, check_dist, avoid_metal_info=None):
    nearby = search_tree.search(pos, check_dist)
    near_pos = n = near_atom = None
    exclude_pos = set([tuple(ex._addh_coord) for ex in exclude])
    exclude_pos.add(tuple(atom._addh_coord))
    from chimerax.geometry import distance
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

from chimerax.atomic.bond_geom import cos705 as cos70_5
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
    from chimerax.geometry import Plane
    plane = Plane(center, normal=v)
    radius = sin70_5 * bond_len
    check_dist += radius

    nearby = search_tree.search(center, check_dist)
    near_pos = n = near_atom = None
    for nb in nearby:
        if nb._addh_coord in [at_pos, n_pos]:
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

        from tinyarray import zeros
        all_zero = zeros(3)
        for candidate, a_rad in candidates:
            c_pos = candidate._addh_coord
            # project into plane...
            proj = plane.nearest(c_pos)

            # find nearest approach of circle...
            cv = proj - center
            if cv == all_zero:
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
    if parent_atom.element.valence < 5 and parent_type_info.geometry != linear:
        # non-sp1 carbons, et al, can't coordinate metals
        return False
    from chimerax.geometry import distance, angle
    if distance(metal_pos, pos) > _metal_dist - 1.25:
        # default "_metal_dist" is 2.7 + putative S-H bond length of 1.25;
        # see nitrogen stripping in CYS 77 and 120 in 3r24
        return False
    # 135.0 is not strict enough (see :1004.a in 1nyr)
    if angle(parent_pos, pos, metal_pos) > 120.0:
        return True
    return False

def unk_atom_truncated(atom):
    if atom.is_side_chain:
        num_heavy_nbs = len([nb for nb in atom.neighbors if nb.element.number > 1])
        if atom.is_side_connector:
            # CA or ribose ring
            if atom.is_backbone(atom.BBE_MIN) or atom.name == "C1'":
                # atoms that connect the side chain to the backbone
                return num_heavy_nbs < 3
            elif atom.name == "O2'":
                return num_heavy_nbs < 1
        return num_heavy_nbs < 2 and atom.name not in ["N2", "N4", "N6", "O2", "O4", "O6"]
    return False

def vdw_radius(atom):
    # to avoid repeated IDATM computation in the middle of hydrogen addition
    return _radii.get(atom, h_rad)

def add_altloc_hyds(atom, altloc_hpos_info, invert, bonding_info, total_hydrogens, naming_schema):
    added_hs = []
    create_alt_loc = atom.num_alt_locs < 2 and len(altloc_hpos_info) > 1
    for alt_loc, occupancy, positions in altloc_hpos_info:
        if create_alt_loc:
            parent_alt_locs.setdefault(atom, []).append((alt_loc, occupancy))
        if added_hs:
            for h, pos in zip(added_hs, positions):
                if h is None:
                    continue
                h.set_alt_loc(alt_loc, True)
                if invert is None:
                    h.coord = pos
                else:
                    h.coord = invert * pos
                h.occupancy = occupancy
        else:
            for i, pos in enumerate(positions):
                if invert is not None:
                    pos = invert * pos
                h = new_hydrogen(atom, i+1, total_hydrogens, naming_schema,
                                    pos, bonding_info, alt_loc)
                added_hs.append(h)
                if h is not None:
                    h.occupancy = occupancy
    # creating alt locs doesn't change any other atom's altloc settings, so...
    for alt_loc, occupany, positions in altloc_hpos_info:
        for added in added_hs:
            if added is None:
                continue
            added.alt_loc = alt_loc
            added.bfactor = atom.bfactor
    return added_hs

def new_hydrogen(parent_atom, h_num, total_hydrogens, naming_schema, pos, parent_type_info, alt_loc):
    global _serial, _metals
    nearby_metals = _metals.search(pos, _metal_dist) if _metal_dist > 0.0 else []
    use_scene = Atom._addh_coord == Atom.scene_coord
    for metal in nearby_metals:
        if metal.structure != parent_atom.structure:
            continue
        if metal.has_alt_loc(alt_loc):
            if use_scene:
                metal_pos = metal.get_alt_loc_scene_coord(alt_loc)
            else:
                metal_pos = metal.get_alt_loc_coord(alt_loc)
        else:
            metal_pos = metal._addh_coord
        if parent_atom.has_alt_loc(alt_loc):
            if use_scene:
                parent_pos = parent_atom.get_alt_loc_scene_coord(alt_loc)
            else:
                parent_pos = parent_atom.get_alt_loc_coord(alt_loc)
        else:
            parent_pos = parent_atom._addh_coord
        if metal_clash(metal_pos, pos, parent_pos, parent_atom, parent_type_info):
            return
    # determine added H color before actually adding it...
    h_color = determine_h_color(parent_atom)
    new_h = add_atom(_h_name(parent_atom, h_num, total_hydrogens, naming_schema), "H",
        parent_atom.residue, pos, serial_number=_serial, bonded_to=parent_atom, alt_loc=alt_loc)
    _serial = new_h.serial_number + 1
    new_h.color = h_color
    new_h.hide = parent_atom.hide
    return new_h

def determine_h_color(parent_atom):
    global _h_coloring, _solvent_atoms
    res = parent_atom.residue
    struct = parent_atom.structure
    if struct not in _solvent_atoms:
        from weakref import WeakSet
        solvent_set = WeakSet()
        struct_atoms = struct.atoms
        solvent_set.update(
            [a for a in struct_atoms.filter(struct_atoms.structure_categories == "solvent")])
        _solvent_atoms[struct] = solvent_set
    else:
        solvent_set = _solvent_atoms[struct]
    if res.name in res.water_res_names or parent_atom in solvent_set:
        return element_color(1)
    if parent_atom.structure in _h_coloring:
        color_scheme = _h_coloring[parent_atom.structure]
    else:
        num_match_elements = 0
        for a in parent_atom.structure.atoms:
            if a.residue.name in res.water_res_names or struct in solvent_set:
                continue
            if a.element.name == "C":
                continue
            if a.color == element_color(a.element.number):
                num_match_elements += 1
                if num_match_elements > 1:
                    color_scheme = "element"
                    break
            else:
                color_scheme = "parent"
                break
        else:
            color_scheme = "element"
        _h_coloring[parent_atom.structure] = color_scheme
    return parent_atom.color if color_scheme == "parent" else element_color(1)

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
    },
    'NH2': {
        "N": ["HN1", "HN2"]
    }
}

def to_h36(base10, max_digits):
    decimal_limit = eval('9' * max_digits)
    if base10 <= decimal_limit:
        return str(base10)
    base36_num =  10 * 36 ** (max_digits-1) + (base10 - decimal_limit - 1)
    base36_str = ""
    while base36_num > 0:
        digit = base36_num % 36
        if digit < 10:
            d_str = str(digit)
        else:
            d_str = chr(ord('A') + digit - 10)
        base36_str = d_str + base36_str
        base36_num = int(base36_num / 36)
    return base36_str

def _h_name(atom, h_num, total_hydrogens, naming_schema):
    res_name = atom.residue.name
    find_atom = atom.residue.find_atom

    res_schema, pdb_version = naming_schema
    if res_schema == "simple":
        i = 1
        while atom.residue.find_atom("H%s" % to_h36(i, 3)):
            i += 1
        return "H%s" % to_h36(i, 3)
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
        # glycosylated asparagines should use the un-glycosylated names
        if total_hydrogens > 1 or find_atom(h_name) or (res_name == "ASN" and atom.name == "ND2"):
            while find_atom("%d%s" % (h_num, h_name)):
                h_num += 1
            h_name = "%d%s" % (h_num, h_name)
    elif h_name[-1] == "'" and len(h_name) + (total_hydrogens-1) <= 4:
        while find_atom(h_name):
            h_name += "'"
            if len(h_name) > 4:
                for digit in range(10):
                    h_name = h_name[:3] + str(digit)
                    if not find_atom(h_name):
                        break
                else:
                    raise ValueError("Too many hydrogens attached to %s" % atom)
    elif total_hydrogens > 1 or find_atom(h_name) or (res_name == "ASN" and atom.name == "ND2"):
        # amino acids number their CH2 hyds as 2/3 rather than 1/2
        if atom.residue.polymer_type == atom.residue.PT_AMINO and total_hydrogens == 2 and len(
                [nb for nb in atom.neighbors if nb.element.number > 1]) == 2:
            h_num += 1
        h_digits = max(4 - len(h_name), 1)
        while find_atom("%s%s" % (h_name, to_h36(h_num, h_digits))):
            h_num += 1
        h_name = "%s%s" % (h_name, to_h36(h_num, h_digits))
    return h_name

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, Or, EmptyArg, FloatArg
    from chimerax.atomic import AtomicStructuresArg
    desc = CmdDesc(
        required=[('structures', Or(AtomicStructuresArg,EmptyArg))],
        keyword = [('hbond', BoolArg), ('in_isolation', BoolArg), ('metal_dist', FloatArg),
            ('template', BoolArg), ('use_his_name', BoolArg), ('use_glu_name', BoolArg),
            ('use_asp_name', BoolArg), ('use_lys_name', BoolArg), ('use_cys_name', BoolArg)],
        synopsis = 'Add hydrogens'
    )
    register('addh', desc, cmd_addh, logger=logger)
