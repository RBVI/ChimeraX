# vim: set expandtab ts=4 sw=4:

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

verbose = False

from .acceptor_geom import acc_syn_anti, acc_phi_psi, acc_theta_tau, acc_generic
from .donor_geom import don_theta_tau, don_upsilon_tau, don_generic, don_water
from .common_geom import ConnectivityError, AtomTypeError
from chimerax.chem_group import find_group
from chimerax.geometry import distance_squared
from .hydpos import hyd_positions
from chimerax.atomic.idatm import type_info, tetrahedral, planar, linear, single
from chimerax.atomic import Element
from chimerax.core.errors import UserError
import copy

from chimerax.chem_group import H, N, C, O, R
from chimerax.chem_group.chem_group import find_ring_planar_NHR2, find_nonring_ether, \
    find_nonring_NR2, find_6ring_planar_NR2, find_5ring_planar_NR2, find_5ring_OR2
_ring5_NH = lambda structs, ret_coll: find_ring_planar_NHR2(structs, ret_coll, 5)
_ring6_aro_NH = lambda structs, ret_coll: find_ring_planar_NHR2(structs, ret_coll, 6,
    aromatic_only=True)
_ring6_sym_N = lambda structs, ret_coll: find_6ring_planar_NR2(structs, ret_coll, symmetric=True)
_ring5_sym_N = lambda structs, ret_coll: find_5ring_planar_NR2(structs, ret_coll, symmetric=True)
_ring6_asym_N = lambda structs, ret_coll: find_6ring_planar_NR2(structs, ret_coll, symmetric=False)
_ring5_asym_N = lambda structs, ret_coll: find_5ring_planar_NR2(structs, ret_coll, symmetric=False)
_ring5_O = lambda structs, ret_coll: find_5ring_OR2(structs, ret_coll)

tet = {'geometry':tetrahedral}
explicit_single_bond = ({'geometry':tetrahedral}, {'geometry':single})
non_O3_minus_sb = ({'geometry':tetrahedral, 'not type': ['O3-']},
                            {'geometry':single})
non_Sac_Pac_H_sb = ({'geometry':tetrahedral, 'not type': ['Sac', 'Pac']},
                {'geometry':single, 'not type': ['H', 'HC', 'D']})

# recommended distance and angle constraint relaxations for biomolecules
rec_dist_slop = 0.4
rec_angle_slop = 20.0

# layout of acceptor_params is:
#     name or spec of acceptor group,
#     tuple indicating index of acceptor or acceptors in group,
#     function to determine if acceptor satisfied,
#     tuple of arguments to give to that function
#
#     in argument tuple, conversions will occur before function gets called.
#     namely:
#         floats are assumed to be distances, and will be squared
#         integers are assumed to be angles in degrees, and will be
#             converted to radians
#         integer/None tuples are assumed to refer to other atoms in
#             the group, and will be replaced by them.  None remains
#             None.  '-1' means the other acceptor of the group.

acceptor_params = [
    # phosphodiester-like (negative oxygens) [3+ O-]
    [[['Pac', ['O3-', 'O3-', 'O3-', explicit_single_bond]], [1,1,1,1,0]],
        (1,), acc_syn_anti, ((2, 0), 3.03, 110, 145, 3.03, 110, 135)],
    # phosphodiester-like (negative oxygens) [2 O-]
    [[['Pac', ['O3-', 'O3-', non_O3_minus_sb, non_O3_minus_sb]], [1,1,1,0,0]],
        (1, 2), acc_syn_anti, ((-1, 0), 3.03, 110, 145, 3.03, 110, 135)],
    # carboxylate
    [[['Cac', ['O2-', 'O2-', explicit_single_bond]], [1,1,1,0]], (1, 2), acc_syn_anti,
        ((-1, 0), 3.17, 90, 145, 3.17, 110, 145)],
    ["thiocarbonyl", (0,), acc_phi_psi, ((1, None), 3.73, 90, 145)],
    # carboxylic acid
    [[[C, [['O2', []], ['O3', [H]], explicit_single_bond]],[1,1,1,0,0]], (1,),
        acc_syn_anti, ((2, 0), 3.17, 110, 130, 3.17, 110, 135)],
    # amide
    [[['C2', ['O2', 'Npl',  explicit_single_bond]], [1,1,1,0]], (1,), acc_syn_anti,
        ((2, 0), 3.30, 110, 135, 3.30, 110, 130)],
    # ketone/aldehyde carbonyl
    [[['C2', ['O2', C, R]], [1,1,0,0]],
        (1,), acc_phi_psi, ((0, None), 3.30, 110, 130)],
    # ester carbonyl
    [[['C2', ['O2', ['O3', [tet]], C]], [1,1,0,0,0]],
        (1,), acc_phi_psi, ((0, None), 3.17, 110, 135)],
    # nitro group
    [[['Ntr', ['O2-', 'O2-', explicit_single_bond]], [1,1,1,0]], (1, 2), acc_syn_anti,
        ((-1, 0), 3.42, 90, 145, 3.53, 110, 130)],
    # sulfone
    [[['Son', ['O2', 'O2', explicit_single_bond, explicit_single_bond]], [1,1,1,0,0]], (1, 2), acc_syn_anti,
        ((-1, 0), 3.30, 110, 130, 3.30, 120, 135)],
    # phosphones
    [[['Pac', ['O3', 'O3-', 'O3-', explicit_single_bond]], [1,1,0,0,0]],
        (1,), acc_theta_tau, ((0,), 3.03, 103, -146, 145)],
    [[['O2', ['Pox']], [1,1]],
        (0,), acc_theta_tau, ((1,), 3.03, 109, -180, 145)],
    # sulfoxide
    [[['O2', ['Sxd']], [1,1]],
        (0,), acc_theta_tau, ((1,), 3.03, 100, -180, 145)],
    # non-protonated aliphatic primary amine
    [[['N3', [tet, H, H]], [1,1,0,0]],
        (0,), acc_theta_tau, ((1,), 3.03, 103, -180, 155)],
    # non-phenol hydroxyl
    [[['O3', [non_Sac_Pac_H_sb, H]], [1,1,1]],
        (0,), acc_theta_tau, ((1,), 3.03, 100, -161, 145)],
    [[['Sac', ['O3-', 'O3-', ['O3', [{'default': True, 'not type': ['Pac']}]], explicit_single_bond]], [1,1,1,1,0,0]],
        (3,), acc_theta_tau, ((0,), 3.17, 103, -180, 150)],
    # non-protonated aliphatic tertiary amine
    [[['N3', [tet, tet, tet]], [1,1,1,1]],
        (0,), acc_theta_tau, (None, 3.17, 153, -180, 145)],
    # nitrile
    [[['N1', ['C1']], [1,1]],
        (0,), acc_theta_tau, ((1,), 3.30, 153, -180, 150)],
    # non-protonated aliphatic secondary amine
    [[['N3', [tet, tet, H]], [1,1,1,0]],
        (0,), acc_theta_tau, (None, 3.30, 153, -180, 150)],
    # phenol
    [[['O3', ['Car', H]], [1,1,0]],
        (0,), acc_theta_tau, ((1,), 3.17, 100, -153, 150)],
    # anilene
    [[[('Npl', 'N3'), ['Car', H, H]], [1,1,1,1]],
        (0,), lambda *args: acc_theta_tau(*args, tau=37.5, tau_sym=4), ((1,), 3.42, 90, -137, 140)],
    # waddah
    [[[O, [H, H]], [1,0,0]], (0,), acc_phi_psi,
                ((None, None), 3.03, 120, 145)],
    # non-ring ether
    [[lambda structs, ret_coll: find_nonring_ether(structs, ret_coll), None],
        (0,), acc_phi_psi, ((1, 2), 3.42, 140, 140)],
    # secondary amine not in a ring system
    [[lambda structs, ret_coll: find_nonring_NR2(structs, ret_coll), None],
        (0,), acc_phi_psi, ((1, 2), 3.42, 140, 140)],

    # check ring systems last, since conflicts of ring systems with
    # non-ring systems are not considered to be a problem (non-ring
    # "wins") and the code that checks the conflict assumes this order

    # nitrogen in symmetric 6-member ring
    [[_ring6_sym_N, None], (0,), acc_phi_psi, ((1, 2), 3.17, 150, 145)],
    # nitrogen in symmetric 5-member ring
    [[_ring5_sym_N, None], (0,), acc_phi_psi, ((1, 2), 3.30, 140, 155)],
    # nitrogen in asymmetric 6-member ring
    [[_ring6_asym_N, None], (0,), acc_phi_psi, ((1, 2), 3.30, 140, 140)],
    # nitrogen in asymmetric 5-member ring
    [[_ring5_asym_N, None], (0,), acc_phi_psi, ((1, 2), 3.30, 150, 135)],
    # oxygen in 5-member ring
    [[_ring5_O, None], (0,), acc_phi_psi, ((1, 2), 3.42, 150, 135)]
]
processed_acceptor_params = {}

# layout of donor_params somewhat similar to acceptor_params:
#    name or spec of donor group,
#    index of donor in group,
#    type of donor geometry
#    degree of tau symmetry
#    argument tuple used when geometry-check function is called
#
#    in argument tuple, conversions will occur before function gets called.
#    namely:
#        positive floats are assumed to be distances, and will be squared
#        integers are assumed to be angles in degrees, and will be
#            converted to radians

import sys
water = sys.intern("water")
theta_tau = sys.intern('theta_tau')
upsilon_tau = sys.intern('upsilon_tau')
OH_bond_dist = Element.bond_length("O", "H")

donor_params = [
    # neutral carboxylic acid
    [[['O3', ['Cac', H]], [1,1,0]],
        0, upsilon_tau, 2,
        (2.87, 103, -128, 140, -30,
         2.87, 103, -128, 155, -30, 150,
         2.87, 103, -128, 140, -30, 150)],
    # protonated nitrogen double-bonded to carbon
    [[['Npl', [['C2', [explicit_single_bond, explicit_single_bond]], H, H]], [1,1,0,0,0,0]],
        0, upsilon_tau, 4,
        (3.17,  90, -146, 140, -30,
         3.17,  90, -146, 140, -22.5, 150,
         # next line extrapolated
         3.17, 113, -161, 125, None, 140)],
    [[['Ng+', ['C2', H, H]], [1,1,0,0]],
        0, upsilon_tau, 4,
        (3.17,  90, -146, 140, -30,
         3.17,  90, -146, 140, -22.5, 150,
         # next line extrapolated
         3.17, 113, -161, 125, None, 140)],
    # protonated primary amine
    [[['N3+', [tet, H, H, H]], [1,1,0,0,0]],
        0, upsilon_tau, 6,
        (3.17,  90, -136, 140, 20,
         3.17,  90, -132, 150, 25, 130,
         3.17,  90, -124, 155, 15, 150)],
    # hydroxyl
    [[['O3', [tet, H]], [1,1,0]],
        0, upsilon_tau, 6,
        (3.17,  90, -141, 130, None,
         3.03,  90, -141, 135, None, 140,
         3.03, 100, -132, 145, None, 150)],
    # amide (two hydrogens)
    #   though the paper gives the tau symmetry as '2', we use '4' since
    #   we aren't distinguishing between the carbonyl oxygen and the
    #   other bonded substituent
    [[['C2', ['O2', [N, [H, H]], explicit_single_bond]], [1,1,1,0,0,0]],
        2, upsilon_tau, 4,
        (3.30, 103, -153, 135, -45,
         3.30, 109, -141, 140, -30, 140,
         3.30, 113, -161, 125, -30, 145)],
    # aniline-like
    ["aromatic primary amine",
        0, upsilon_tau, 4,
        (3.30, 100, -146, 145, -37.5,
         3.30, 103, -146, 140, -37.5, 130,
         # estimated
         3.42,  90,  161, 125, None, 140)],
    # phenol-like
    [[['O3', ['Car', H]], [1,0,0]],
        0, upsilon_tau, 4,
        (3.03, 103, -136, 145, -37.5,
         3.03, 103, -136, 150, -45.0, 140,
         2.87, 109, -128, 145, -52.5, 150)],
    # neutral primary aliphatic amine
    [[['N3', [tet, H, H]], [1,0,1,1]],
        0, upsilon_tau, 6,
        (3.30, 90, -128, 150, 30,
         3.42, 90, -124, 155, 25, 130,
         3.42, 90, -132, 155, 20, 140)],
    # planar charged nitrogen with one hydrogen
    [[['Npl', [tet, [{'geometry':planar}, [(tet,H), (tet,H)]], H]], [1,0,0,0,0,0]],
        0, theta_tau, 4,
        (1.92, 146, 1.92, 153, 150, 2.48, 141, 145)],
    # sp3 charged nitrogen with 2 hydrogens
    [[['N3+', [tet, tet, H, H]], [1,0,0,1,1]],
        0, theta_tau, 2,
        (2.09, 141, 2.09, 141, 155, 2.48, 141, 145)],
    # waddah
    [[[O, [H, H]], [1,0,0]],
        0, water, 2,
        (2.36, 2.36 + OH_bond_dist, 146,
        2.23, 2.23 + OH_bond_dist, 146, 140,
        2.23, 2.23 + OH_bond_dist, 146, 150)],
    # amide (one hydrogen)
    [[['C2', ['O2', [N, [tet, H]], explicit_single_bond]], [1,1,1,0,0,0]],
        2, theta_tau, 2,
        (2.23, 146, 2.23, 146, 140, 2.36, 141, 145)],
    # aliphatic tertiary amine
    [[['N3+', [tet, tet, tet, H]], [1,0,0,0,1]],
        0, theta_tau, 6,
        (1.92, 146, 2.09, 146, 140, 2.48, 141, 145)],
    # double-bonded nitrogen
    [[['N2', [{'geometry':planar}, H]], [1,1,0]],
        0, theta_tau, 2,
        (2.23, 132, 2.36, 146, 140, 2.48, 141, 145)],
    # secondary amine
    [[['N3', [tet, tet, H]], [1,0,0,1]],
        0, theta_tau, 2,
        (2.36, 141, 2.48, 146, 150, 2.48, 153, 155)],

    # check ring systems last, since conflicts of ring systems with
    # non-ring systems are not considered to be a problem (non-ring
    # "wins") and the code that checks the conflict assumes this order

    # nitrogen in 5-member ring
    [[_ring5_NH, None],
        0, theta_tau, 4,
        (2.09, 146, 2.09, 141, 140, 2.48, 141, 145)],
    # nitrogen in aromatic 6-member ring
    [[_ring6_aro_NH, None],
        0, theta_tau, 2,
        (2.23, 136, 2.23, 141, 150, 2.48, 141, 145)],
    # need below to cause check for generic donors
    [None, 0, 'generic', 0, (3.73, 3.73, 3.73)]
]
processed_donor_params = {}

def flush_cache():
    global _d_cache, _a_cache, _prev_limited
    _prev_limited = _d_cache = _a_cache = None
flush_cache()

_problem = None
_ring_funcs = [_ring5_asym_N, _ring6_asym_N, _ring5_O,
                _ring5_sym_N, _ring6_sym_N, _ring5_NH, _ring6_aro_NH]

def find_coordset_hbonds(session, structure, **kw):
    """Like find_hbonds, but takes a single structure and cycles through its coordsets
       and finds the hydrogen bonds for each.  Returns a list of lists of hydrogen
       bonds, one list per coordset.
    """
    hbonds = []
    cs_ids = structure.coordset_ids
    structure.active_coordset_change_notify = False
    cur_cs_id = structure.active_coordset_id
    try:
        for cs_id in cs_ids:
            structure.active_coordset_id = cs_id
            kw['cache_da'] = cs_id != cs_ids[-1]
            hbonds.append(find_hbonds(session, [structure], **kw))
    finally:
        structure.active_coordset_od = cur_cs_id
        structure.active_coordset_change_notify = True
    return hbonds

def find_hbonds(session, structures, *, inter_model=True, intra_model=True, donors=None, acceptors=None,
        dist_slop=0.0, angle_slop=0.0, inter_submodel=False, cache_da=False, status=True):
    """Hydrogen bond detection based on criteria in "Three-dimensional
        hydrogen-bond geometry and probability information from a
        crystal survey", J. Computer-Aided Molecular Design, 10 (1996),
        607-622

        If donors and/or acceptors are specified (as :py:class:`~chimerax.atomic.Atoms` collections
        or anything an Atoms collection can be constructued from), then H-bond donors/acceptors
        are restricted to being from those atoms.

        Dist/angle slop are the amount that distances/angles are allowed to exceed
        the values given in the above reference and still be considered hydrogen bonds.

        'cache_da' allows donors/acceptors in molecules to be cached if it is anticipated that
        the same structures will be examined for H-bonds repeatedly (e.g. a dynamics trajectory).

        If 'per_coordset' is True and 'structures' contains a single structure with multiple coordinate
        sets, then hydrogen bonds will be computed for each coordset.

        If 'status' is True, progress will be logged to the status line.

        Returns a list of donor/acceptor pairs, unless the conditions for 'per_coordset' are
        satisfied, in which case a list of such lists will be returned, one per coordset.
    """

    # hack to speed up coordinate lookup...
    from chimerax.atomic import Atoms, Atom
    if len(structures) == 1 or not inter_model or (
            len(set([m if m.id is None else (m.id[0] if len(m.id) == 1 else m.id[:-1])
            for m in structures])) == 1 and not inter_submodel):
        Atom._hb_coord = Atom.coord
    else:
        Atom._hb_coord = Atom.scene_coord
    try:
        if donors and not isinstance(donors, Atoms):
            limited_donors = Atoms(donors)
        else:
            limited_donors = donors
        if acceptors and not isinstance(acceptors, Atoms):
            limited_acceptors = Atoms(acceptors)
        else:
            limited_acceptors = acceptors
        global _d_cache, _a_cache, _prev_limited
        if cache_da:
            if limited_donors:
                dIDs = [id(d) for d in limited_donors]
                dIDs.sort()
            else:
                dIDs = None
            if limited_acceptors:
                aIDs = [id(a) for a in limited_acceptors]
                aIDs.sort()
            else:
                aIDs = None
            key = (dIDs, aIDs)
            if _prev_limited and _prev_limited != key:
                flush_cache()
            _prev_limited = key
            from weakref import WeakKeyDictionary
            if _d_cache is None:
                _d_cache = WeakKeyDictionary()
                _a_cache = WeakKeyDictionary()
        else:
            flush_cache()
        global donor_params, acceptor_params
        global processed_donor_params, processed_acceptor_params
        global _compute_cache
        global verbose
        global _problem
        _problem = None
        global _truncated
        _truncated = set()

        bad_connectivities = 0

        # Used (as necessary) to cache expensive calculations (by other functions also)
        _compute_cache = {}

        process_key = (dist_slop, angle_slop)
        if process_key not in processed_acceptor_params:
            # copy.deepcopy() refuses to copy functions (even as
            # references), so do this instead...
            a_params = []
            for p in acceptor_params:
                a_params.append(copy.copy(p))

            for i in range(len(a_params)):
                a_params[i][3] = _process_arg_tuple(a_params[i][3], dist_slop, angle_slop)
            processed_acceptor_params[process_key] = a_params
        else:
            a_params = processed_acceptor_params[process_key]

        # compute some info for generic acceptors/donors
        generic_acc_info = {}
        # oxygens...
        generic_O_acc_args = _process_arg_tuple([3.53, 90], dist_slop, angle_slop)
        generic_acc_info['misc_O'] = (acc_generic, generic_O_acc_args)
        # dictionary based on bonded atom's geometry...
        generic_acc_info['O2-'] = {
            single: (acc_generic, generic_O_acc_args),
            linear: (acc_generic, generic_O_acc_args),
            planar: (acc_phi_psi, _process_arg_tuple([3.53, 90, 130], dist_slop, angle_slop)),
            tetrahedral: (acc_generic, generic_O_acc_args)
        }
        generic_acc_info['O3-'] = generic_acc_info['O2-']
        generic_acc_info['O2'] = {
            single: (acc_generic, generic_O_acc_args),
            linear: (acc_generic, generic_O_acc_args),
            planar: (acc_phi_psi, _process_arg_tuple([3.30, 110, 130], dist_slop, angle_slop)),
            tetrahedral: (acc_theta_tau, _process_arg_tuple(
                [3.03, 100, -180, 145], dist_slop, angle_slop))
        }
        # list based on number of known bonded atoms...
        generic_acc_info['O3'] = [
            (acc_generic, generic_O_acc_args),
            (acc_theta_tau, _process_arg_tuple([3.17, 100, -161, 145], dist_slop, angle_slop)),
            (acc_phi_psi, _process_arg_tuple([3.42, 120, 135], dist_slop, angle_slop))
        ]
        # nitrogens...
        generic_N_acc_args = _process_arg_tuple([3.42, 90], dist_slop, angle_slop)
        generic_acc_info['misc_N'] = (acc_generic, generic_N_acc_args)
        generic_acc_info['N2'] = (acc_phi_psi, _process_arg_tuple([3.42, 140, 135],
                dist_slop, angle_slop))
        # tuple based on number of bonded heavy atoms...
        generic_N3_mult_heavy_acc_args = _process_arg_tuple([3.30, 153, -180, 145],
                dist_slop, angle_slop)
        generic_acc_info['N3'] = (
            (acc_generic, generic_N_acc_args),
            # only one example to draw from; weaken by .1A, 5 degrees
            (acc_theta_tau, _process_arg_tuple([3.13, 98, -180, 150], dist_slop, angle_slop)),
            (acc_theta_tau, generic_N3_mult_heavy_acc_args),
            (acc_theta_tau, generic_N3_mult_heavy_acc_args)
        )
        # one example only; weaken by .1A, 5 degrees
        generic_acc_info['N1'] = (acc_theta_tau, _process_arg_tuple(
                    [3.40, 136, -180, 145], dist_slop, angle_slop))
        # sulfurs...
        # one example only; weaken by .1A, 5 degrees
        generic_acc_info['S2'] = (acc_phi_psi, _process_arg_tuple([3.83, 85, 140],
                dist_slop, angle_slop))
        generic_acc_info['Sar'] = generic_acc_info['S3-'] = (acc_generic,
                _process_arg_tuple([3.83, 85], dist_slop, angle_slop))
        # now the donors...

        # planar nitrogens
        gen_don_Npl_1h_params = (don_theta_tau, _process_arg_tuple([2.23, 136,
            2.23, 141, 140, 2.46, 136, 140], dist_slop, angle_slop))
        gen_don_Npl_2h_params = (don_upsilon_tau, _process_arg_tuple([3.30, 90, -153,
            135, -45, 3.30, 90, -146, 140, -37.5, 130, 3.40, 108, -166, 125, -35, 140],
            dist_slop, angle_slop))
        gen_don_O_dists = [2.41, 2.28, 2.28, 3.27, 3.14, 3.14]
        gen_don_O_params = (don_generic, _process_arg_tuple(gen_don_O_dists, dist_slop, angle_slop))
        gen_don_N_dists = [2.36, 2.48, 2.48, 3.30, 3.42, 3.42]
        gen_don_N_params = (don_generic, _process_arg_tuple(gen_don_N_dists, dist_slop, angle_slop))
        gen_don_S_dists = [2.42, 2.42, 2.42, 3.65, 3.65, 3.65]
        gen_don_S_params = (don_generic, _process_arg_tuple(gen_don_S_dists, dist_slop, angle_slop))
        generic_don_info = {
            'O': gen_don_O_params,
            'N': gen_don_N_params,
            'S': gen_don_S_params
        }

        from chimerax.atom_search import AtomSearchTree
        metal_coord = {}
        acc_trees = {}
        hbonds = []
        has_sulfur = {}
        for structure in structures:
            if status:
                session.logger.status("Finding acceptors in model '%s'" % structure.name, blank_after=0)
            if cache_da and structure in _a_cache and (dist_slop,
                    angle_slop) in _a_cache[structure]:
                acc_atoms = []
                acc_data = []
                for acc_atom, data in _a_cache[structure][(dist_slop, angle_slop)].items():
                    if not acc_atom.deleted:
                        acc_atoms.append(acc_atom)
                        acc_data.append(data)
            else:
                acc_atoms, acc_data = _find_acceptors(structure, a_params,
                        limited_acceptors, generic_acc_info)
                if cache_da:
                    cache = WeakKeyDictionary()
                    for i in range(len(acc_atoms)):
                        cache[acc_atoms[i]] = acc_data[i]
                    if structure not in _a_cache:
                        _a_cache[structure] = {}
                    _a_cache[structure][(dist_slop, angle_slop)] = cache
            #xyz = []
            has_sulfur[structure] = False
            for acc_atom in acc_atoms:
                #c = acc_atom._hb_coord
                #xyz.append([c[0], c[1], c[2]])
                if acc_atom.element == Element.get_element('S'):
                    has_sulfur[structure] = True
            if status:
                session.logger.status("Building search tree of acceptor atoms", blank_after=0)
            acc_tree = acc_trees[structure] = AtomSearchTree(acc_atoms, data=acc_data, sep_val=3.0,
                scene_coords=(Atom._hb_coord == Atom.scene_coord))
            metals = structure.atoms.filter(structure.atoms.elements.is_metal)
            for metal in metals:
                for acc_atom, geom_func, args in acc_tree.search(metal._hb_coord, 4.0):
                    metal_coord.setdefault(acc_atom, []).append(metal)

        if process_key not in processed_donor_params:
            # find max donor distances before they get squared..

            # copy.deepcopy() refuses to copy functions (even as
            # references), so do this instead...
            d_params = []
            for p in donor_params:
                d_params.append(copy.copy(p))

            for di in range(len(d_params)):
                geom_type = d_params[di][2]
                arg_list = d_params[di][4]
                don_rad = Element.bond_radius('N')
                if geom_type == theta_tau:
                    max_dist = max((arg_list[0], arg_list[2], arg_list[5]))
                elif geom_type == upsilon_tau:
                    max_dist = max((arg_list[0], arg_list[5], arg_list[11]))
                elif geom_type == water:
                    max_dist = max((arg_list[1], arg_list[4], arg_list[8]))
                else:
                    max_dist = max(gen_don_O_dists + gen_don_N_dists + gen_don_S_dists)
                    don_rad = Element.bond_radius('S')
                d_params[di].append(max_dist + dist_slop + don_rad + Element.bond_radius('H'))

            for i in range(len(d_params)):
                d_params[i][4] = _process_arg_tuple(d_params[i][4], dist_slop, angle_slop)
            processed_donor_params[process_key] = d_params
        else:
            d_params = processed_donor_params[process_key]

        generic_water_params = _process_arg_tuple([2.36, 2.36 + OH_bond_dist, 146],
                                dist_slop, angle_slop)
        generic_theta_tau_params = _process_arg_tuple([2.48, 132], dist_slop, angle_slop)
        generic_upsilon_tau_params = _process_arg_tuple([3.42, 90, -161, 125], dist_slop, angle_slop)
        generic_generic_params = _process_arg_tuple([2.48, 3.42, 130, 90], dist_slop, angle_slop)
        for dmi in range(len(structures)):
            structure = structures[dmi]
            if status:
                session.logger.status("Finding donors in model '%s'" % structure.name, blank_after=0)
            if cache_da and structure in _d_cache and (dist_slop, angle_slop) in _d_cache[structure]:
                don_atoms = []
                don_data = []
                for don_atom, data in _d_cache[structure][(dist_slop, angle_slop)].items():
                    if not don_atom.deleted:
                        don_atoms.append(don_atom)
                        don_data.append(data)
            else:
                don_atoms, don_data = _find_donors(structure, d_params, limited_donors, generic_don_info)
                if cache_da:
                    cache = WeakKeyDictionary()
                    for i in range(len(don_atoms)):
                        cache[don_atoms[i]] = don_data[i]
                    if structure not in _d_cache:
                        _d_cache[structure] = {}
                    _d_cache[structure][(dist_slop, angle_slop)] = cache

            if status:
                session.logger.status("Matching donors in model '%s' to acceptors" % structure.name,
                    blank_after=0)
            for i in range(len(don_atoms)):
                donor_atom = don_atoms[i]
                geom_type, tau_sym, arg_list, test_dist = don_data[i]
                donor_hyds = hyd_positions(donor_atom)
                coord = donor_atom._hb_coord
                for acc_structure in structures:
                    if acc_structure == structure and not intra_model or acc_structure != structure and not inter_model:
                        continue
                    if not inter_submodel \
                    and acc_structure.id and structure.id \
                    and acc_structure.id[0] == structure.id[0] \
                    and acc_structure.id[:-1] == structure.id[:-1] \
                    and acc_structure.id[1:] != structure.id[1:]:
                        continue
                    if has_sulfur[acc_structure]:
                        from .common_geom import SULFUR_COMP
                        td = test_dist + SULFUR_COMP
                    else:
                        td = test_dist
                    accs = acc_trees[acc_structure].search(coord, td)
                    if verbose:
                        session.logger.info("Found %d possible acceptors for donor %s:"
                            % (len(accs), donor_atom))
                        for acc_data in accs:
                            session.logger.info("\t%s\n" % acc_data[0])
                    for acc_atom, geom_func, args in accs:
                        if acc_atom == donor_atom:
                            # e.g. hydroxyl
                            if verbose:
                                print("skipping: donor == acceptor")
                            continue
                        try:
                            if not geom_func(donor_atom, donor_hyds, *args):
                                continue
                        except ConnectivityError as e:
                            session.logger.info("Skipping possible acceptor with bad geometry: %s\n%s\n"
                                % (acc_atom, e))
                            bad_connectivities += 1
                            continue
                        except Exception:
                            print("donor:", donor_atom, " acceptor:", acc_atom)
                            raise
                        if verbose:
                            session.logger.info("\t%s satisfies acceptor criteria" % acc_atom)
                        if geom_type == upsilon_tau:
                            donor_func = don_upsilon_tau
                            add_args = generic_upsilon_tau_params + [tau_sym]
                        elif geom_type == theta_tau:
                            donor_func = don_theta_tau
                            add_args = generic_theta_tau_params
                        elif geom_type == water:
                            donor_func = don_water
                            add_args = generic_water_params
                        else:
                            if donor_atom.idatm_type in ["Npl", "N2+"]:
                                heavys = 0
                                for bonded in donor_atom.neighbors:
                                    if bonded.element.number > 1:
                                        heavys += 1
                                if heavys > 1:
                                    info = gen_don_Npl_1h_params
                                else:
                                    info = gen_don_Npl_2h_params
                            else:
                                info = generic_don_info[donor_atom.element.name]
                            donor_func, arg_list = info
                            add_args = generic_generic_params
                            if donor_func == don_upsilon_tau:
                                # tack on generic
                                # tau symmetry
                                add_args = generic_upsilon_tau_params + [4]
                            elif donor_func == don_theta_tau:
                                add_args = generic_theta_tau_params
                        try:
                            if not donor_func(donor_atom, donor_hyds, acc_atom,
                                    *tuple(arg_list + add_args)):
                                continue
                        except ConnectivityError as e:
                            session.logger.info("Skipping possible donor with bad geometry: %s\n%s\n"
                                % (donor_atom, e))
                            bad_connectivities += 1
                            continue
                        except AtomTypeError as e:
                            session.logger.warning(str(e))
                            #_problem = ("atom type", donor_atom, str(e), None)
                            continue
                        if verbose:
                            session.logger.info("\t%s satisfies donor criteria" % donor_atom)
                        # ensure hbond isn't precluded by metal-coordination...
                        if acc_atom in metal_coord:
                            from chimerax.geometry import angle
                            conflict = False
                            for metal in metal_coord[acc_atom]:
                                if angle(donor_atom._hb_coord, acc_atom._hb_coord, metal._hb_coord) < 45.0:
                                    if verbose:
                                        session.logger.info("\tH-bond between %s and %s conflicts with"
                                            " metal coordination to %s" % (donor_atom, acc_atom, metal))
                                    conflict = True
                                    break
                            if conflict:
                                continue
                        hbonds.append((donor_atom, acc_atom))
            if status:
                session.logger.status("")
        if bad_connectivities:
            session.logger.warning("Skipped %d atom(s) with bad connectivities; see log for details"
                % bad_connectivities);
        if _problem:
            if session.ui.is_gui:
                # report a bug when atom matches multiple donor/acceptor descriptions
                da, atom, grp1, grp2 = _problem
                res_atoms = atom.residue.atoms
                def res_atom_rep(a):
                    try:
                        i = res_atoms.index(a)
                    except ValueError:
                        return "other %s" % a.element.name
                    return "%2d" % (i+1)
                descript = "geometry class 1: %s\n\ngeometry class 2: %s" % (repr(grp1), repr(grp2))
                session.logger.report_exception(error_description=
    """At least one atom was classified into more than one acceptor or donor
    geometry class.  This indicates a problem in the
    donr/acceptor classification mechanism and we would appreciate it if you
    would use the bug-report button below to send us the information that
    will allow us to improve the classification code.

    residue name: %s

    problem %s atom: %d

    residue atoms:
        %s

    residue bonds:
        %s

    %s
    """ % (atom.residue.name, da, res_atoms.index(atom)+1,
    "\n\t".join(["%2d %-4s %-s (%s)" % (en[0]+1, en[1].name, en[1].idatm_type, str(en[1].coord)) for en in enumerate(res_atoms)]),
    "\n\t".join(["%s <-> %-s" % (res_atom_rep(b.atoms[0]), res_atom_rep(b.atoms[1])) for b in atom.residue.atoms.bonds]),
    descript)
    )
            _problem = None
        if _truncated:
            if len(_truncated) > 20:
                session.logger.warning("%d atoms were skipped as donors/acceptors due to missing"
                    " heavy-atom bond partners" % len(_truncated))
            else:
                session.logger.warning("The following atoms were skipped as donors/acceptors due to missing"
                    " heavy-atom bond partners: %s" % "; ".join([str(a) for a in _truncated]))
            _truncated = None
    finally:
        delattr(Atom, "_hb_coord")
    return hbonds

def _process_arg_tuple(arg_tuple, dist_slop, angle_slop):
    new_args = []
    for arg in arg_tuple:
        if isinstance(arg, (int, float)):
            if arg > 0.0 and arg < 7.5:
                # square distances
                arg = arg + dist_slop
                new_args.append(arg * arg)
            else:
                if arg < 0.0:
                    arg = arg - angle_slop
                    if arg < -180.0:
                        arg = -180.0
                    # can't convert to positive value
                    # here because some constraints vary
                    # between "greater than" and "less than"
                else:
                    arg = arg - angle_slop
                    if arg < 0.0:
                        arg = 0.0
                new_args.append(arg)
        else:
            new_args.append(arg)
    return new_args  # returns list

def _find_acceptors(structure, a_params, limited_acceptors, generic_acc_info):
    global _problem
    global _truncated
    acc_atoms = []
    acc_data = []
    std_acceptors = {}
    for ap in a_params:
        group_key, acc_indices, geom_func, arg_list = ap
        groups = find_group(group_key, [structure])
        for group in groups:
            for i in range(len(acc_indices)):
                acc_atom = group[acc_indices[i]]
                if limited_acceptors and acc_atom not in limited_acceptors:
                    continue
                if acc_atom in std_acceptors:
                    if group_key != std_acceptors[acc_atom] \
                    and not (
                    # conflicts of non-ring groups with
                    # ring groups not considered a problem
                    # (non-ring groups "win")
                    group_key[0] in _ring_funcs and std_acceptors[acc_atom][0] not in _ring_funcs):
                        _problem = ("acceptor", acc_atom, std_acceptors[acc_atom], group_key)
                    continue
                if acc_atom.is_missing_heavy_template_neighbors(no_template_okay=True):
                    _truncated.add(acc_atom)
                    continue
                std_acceptors[acc_atom] = group_key

                args = [acc_atom]
                if isinstance(arg_list[0], tuple):
                    for index in arg_list[0]:
                        if index is None:
                            args.append(None)
                        elif index < 0:
                            args.append(group[acc_indices[1-i]])
                        else:
                            args.append(group[index])
                    args = args + arg_list[1:]
                else:
                    args = args + arg_list
                
                acc_atoms.append(acc_atom)
                acc_data.append([acc_atom, geom_func, tuple(args)])
    # find acceptors not covered by above
    for atom in structure.atoms:
        if atom in std_acceptors:
            continue
        if limited_acceptors and atom not in limited_acceptors:
            continue
        if atom.idatm_type in generic_acc_info:
            if atom.is_missing_heavy_template_neighbors(no_template_okay=True):
                _truncated.add(atom)
                continue
            acc_info = generic_acc_info[atom.idatm_type]
            if isinstance(acc_info, dict):
                try:
                    bonded_geom = type_info[atom.neighbors[0].idatm_type].geometry
                except KeyError:
                    bonded_geom = single
                except IndexError:
                    # no neighbors
                    continue
                acc_func, args = acc_info[bonded_geom]
            elif isinstance(acc_info, list):
                try:
                    acc_func, args = acc_info[atom.num_bonds]
                except IndexError:
                    structure.session.logger.warning("%s has bad number of bonds (%d)" % (atom, atom.num_bonds))
                    #_problem = ("connectivity", atom,
                    #    "bad number of bonds (%d)" % atom.num_bonds, None)
                    continue
            elif isinstance(acc_info, tuple) and isinstance(acc_info[0], tuple):
                acc_func, args = acc_info[sum([nb.element.number > 1 for nb in atom.neighbors])]
            else:
                acc_func, args = acc_info
            if acc_func == acc_phi_psi:
                bonded = atom.neighbors
                args = list(bonded) + [None] * (2-len(bonded)) + args
            elif acc_func == acc_theta_tau:
                bonded = atom.neighbors
                if len(bonded) > 1:
                    args = [None] + args
                else:
                    args = [nb for nb in bonded] + args
        else:
            atomic_number = atom.element.number
            if atomic_number == 8:
                # oxygen
                acc_func, args = generic_acc_info['misc_O']
            elif atomic_number == 7:
                # nitrogen
                if atom.idatm_type not in ['N2', 'N3']:
                    continue
                acc_func, args = generic_acc_info['misc_N']
            else:
                continue
            if atom.is_missing_heavy_template_neighbors(no_template_okay=True):
                _truncated.add(atom)
                continue
            if verbose:
                print("miscellaneous generic acceptor:", atom)
        acc_atoms.append(atom)
        acc_data.append([atom, acc_func, tuple([atom] + args)])
    return acc_atoms, acc_data

def _find_donors(structure, d_params, limited_donors, generic_don_info):
    don_atoms = []
    don_data = []
    std_donors = {}
    for dp in d_params:
        group_key, donorIndex, geom_type, tau_sym, arg_list, test_dist = dp

        if group_key:
            groups = find_group(group_key, [structure])
        else:
            # generic donors
            groups = []
            for atom in structure.atoms:
                if atom in std_donors:
                    continue
                if atom.element.number not in [7,8,16]:
                    continue
                if limited_donors and atom not in limited_donors:
                    continue
                # oxygen, nitrogen, or sulfur
                try:
                    expect_bonds = type_info[atom.idatm_type].substituents
                except KeyError:
                    expect_bonds = 0
                num_bonds = atom.num_bonds
                # screen out the partial terminal N that AddH can leave, since the geometry is
                # problematic and the H direction isn't really determined
                if atom.idatm_type == "Npl" and num_bonds == 2 \
                and 1 in [n.element.number for n in atom.neighbors]:
                    continue
                if num_bonds < expect_bonds:
                    groups.append([atom])
                    continue
                for bonded in atom.neighbors:
                    if bonded.element.number == 1:
                        groups.append([atom])
                        break
            if verbose:
                for g in groups:
                    print("generic donor:", g[0])

        if groups and geom_type == theta_tau:
            # extend probe distance by H-bond length so that all relevant acceptors will be found
            test_dist = test_dist + Element.bond_length(groups[0][donorIndex].element, 'H')
        for group in groups:
            donor_atom = group[donorIndex]
            if limited_donors and donor_atom not in limited_donors:
                continue
            if donor_atom in std_donors:
                if group_key != std_donors[donor_atom] and not (
                # conflicts of non-ring groups with ring
                # groups not considered a problem (non-ring
                # groups "win")
                group_key[0] in _ring_funcs and std_donors[donor_atom][0] not in _ring_funcs):
                    global _problem
                    _problem = ("donor", donor_atom, std_donors[donor_atom], group_key)
                continue
            if donor_atom.is_missing_heavy_template_neighbors(no_template_okay=True):
                global _truncated
                _truncated.add(donor_atom)
                continue
            std_donors[donor_atom] = group_key
            don_atoms.append(donor_atom)
            don_data.append((geom_type, tau_sym, arg_list, test_dist))
    return don_atoms, don_data
