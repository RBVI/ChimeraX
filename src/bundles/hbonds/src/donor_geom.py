# vim: set expandtab ts=4 sw=4:

"""donor geometry testing functions"""

from chimerax.atomic.idatm import type_info, planar, tetrahedral
from chimerax.geometry import angle, distance_squared
from chimerax.atomic.bond_geom import bond_positions
from . import hbond
from .common_geom import AtomTypeError, get_phi_plane_params, \
        test_phi, test_theta, sulphur_compensate, test_tau
from math import sqrt

def don_theta_tau(donor, donor_hyds, acceptor, sp2_O_rp2, sp2_O_theta, sp3_O_rp2, sp3_O_theta,
        sp3_O_phi, sp3_N_rp2, sp3_N_theta, sp3_N_upsilon, gen_rp2, gen_theta, is_water=False):
        # 'is_water' only for hydrogenless water
    if hbond.verbose:
        print("don_theta_tau")
    if len(donor_hyds) == 0 and not is_water:
        if hbond.verbose:
            print("No hydrogens; default failure")
        return False
    ap = acceptor._hb_coord
    dp = donor._hb_coord

    acc_type = acceptor.idatm_type
    if acc_type not in type_info:
        if hbond.verbose:
            print("Unknown acceptor type failure")
        return False

    geom = type_info[acc_type].geometry
    element = acceptor.element.name
    if element == 'O' and geom == planar:
        if hbond.verbose:
            print("planar O")
        for hyd_pos in donor_hyds:
            if distance_squared(hyd_pos, ap) <= sp2_O_rp2:
                break
        else:
            if not is_water:
                if hbond.verbose:
                    print("dist criteria failed (all > %g)"% sqrt(sp2_O_rp2))
                return False
        theta = sp2_O_theta
    elif element == 'O' and geom == tetrahedral or element == 'N' and geom == planar:
        if hbond.verbose:
            print("planar N or tet O")
        for hyd_pos in donor_hyds:
            if distance_squared(hyd_pos, ap) <= sp3_O_rp2:
                break
        else:
            if not is_water:
                if hbond.verbose:
                    print("dist criteria failed (all > %g)" % sqrt(sp3_O_rp2))
                return False
        theta = sp3_O_theta

        # only test phi for acceptors with two bonded atoms
        if acceptor.num_bonds == 2:
            if hbond.verbose:
                print("testing donor phi")
            bonded = acceptor.neighbors
            phi_plane, base_pos = get_phi_plane_params(acceptor, bonded[0], bonded[1])
            if not test_phi(donor._hb_coord, ap, base_pos, phi_plane, sp3_O_phi):
                return False

    elif element == 'N' and geom == tetrahedral:
        if hbond.verbose:
            print("tet N")
        for hyd_pos in donor_hyds:
            if distance_squared(hyd_pos, ap) <= sp3_N_rp2:
                break
        else:
            if not is_water:
                if hbond.verbose:
                    print("dist criteria failed (all > %g)" % sqrt(sp3_N_rp2))
                return False
        theta = sp3_N_theta

        # test upsilon against lone pair directions
        bonded_pos = []
        for bonded in acceptor.neighbors:
            bonded_pos.append(bonded._hb_coord)
        lp_pos = bond_positions(ap, geom, 1.0, bonded_pos)
        if len(lp_pos) > 0:
            # fixed lone pair positions
            for lp in bond_positions(ap, geom, 1.0, bonded_pos):
                # invert position so that we are measuring angles correctly
                ang = angle(dp, ap, ap - (lp - ap))
                if ang > sp3_N_upsilon:
                    if hbond.verbose:
                        print("acceptor upsilon okay (%g > %g)" % (ang, sp3_N_upsilon))
                    break
            else:
                if hbond.verbose:
                    print("all acceptor upsilons failed (< %g)" % sp3_N_upsilon)
                return False
        # else: indefinite lone pair positions; default okay
    else:
        if hbond.verbose:
            print("generic acceptor")
        if acceptor.element.name == "S":
            gen_rp2 = sulphur_compensate(gen_rp2)
        for hyd_pos in donor_hyds:
            if distance_squared(hyd_pos, ap) <= gen_rp2:
                break
        else:
            if hbond.verbose:
                print("dist criteria failed (all > %g)" % sqrt(gen_rp2))
            return False
        theta = gen_theta
    if hbond.verbose:
        print("dist criteria OK")

    return test_theta(dp, donor_hyds, ap, theta)

def don_upsilon_tau(donor, donor_hyds, acceptor,
        sp2_O_r2, sp2_O_upsilon_low, sp2_O_upsilon_high, sp2_O_theta, sp2_O_tau,
        sp3_O_r2, sp3_O_upsilon_low, sp3_O_upsilon_high, sp3_O_theta, sp3_O_tau, sp3_O_phi,
        sp3_N_r2, sp3_N_upsilon_low, sp3_N_upsilon_high, sp3_N_theta, sp3_N_tau, sp3_N_upsilon_N,
        gen_r2, gen_upsilon_low, gen_upsilon_high, gen_theta, tau_sym):

    if hbond.verbose:
        print("don_upsilon_tau")

    acc_type = acceptor.idatm_type
    if acc_type not in type_info:
        return False

    geom = type_info[acc_type].geometry
    element = acceptor.element.name
    if element == 'O' and geom == planar:
        if hbond.verbose:
            print("planar O")
        return test_upsion_tau_acceptor(donor, donor_hyds, acceptor, sp2_O_r2,
                sp2_O_upsilon_low, sp2_O_upsilon_high, sp2_O_theta, sp2_O_tau, tau_sym)
    elif element == 'O' and geom == tetrahedral or element == 'N' and geom == planar:
        if hbond.verbose:
            print("planar N or tet O")
        return test_upsion_tau_acceptor(donor, donor_hyds, acceptor, sp3_O_r2,
                sp3_O_upsilon_low, sp3_O_upsilon_high, sp3_O_theta, sp3_O_tau, tau_sym)
    elif element == 'N' and geom == tetrahedral:
        if hbond.verbose:
            print("tet N")
        # test upsilon at the N
        # see if lone pairs point at the donor
        bonded_pos = []
        for bonded in acceptor.neighbors:
            bonded_pos.append(bonded._hb_coord)
        if len(bonded_pos) > 1:
            ap = acceptor._hb_coord
            dp = donor._hb_coord
            lone_pairs = bond_positions(ap, tetrahedral, 1.0, bonded_pos)
            for lp in lone_pairs:
                up_pos = ap - (lp - ap)
                ang = angle(up_pos, ap, dp)
                if ang >= sp3_N_upsilon_N:
                    if hbond.verbose:
                        print("upsilon(N) okay (%g >= %g)" % (ang, sp3_N_upsilon_N))
                    break
            else:
                if hbond.verbose:
                    print("all upsilon(N) failed (< %g)" % sp3_N_upsilon_N)
                return False
        elif hbond.verbose:
            print("lone pair positions indeterminate at N; upsilon(N) default okay")
        return test_upsion_tau_acceptor(donor, donor_hyds, acceptor, sp3_N_r2,
                sp3_N_upsilon_low, sp3_N_upsilon_high, sp3_N_theta, sp3_N_tau, tau_sym)
    else:
        if hbond.verbose:
            print("generic acceptor")
        if acceptor.element.name == "S":
            gen_r2 = sulphur_compensate(gen_r2)
        return test_upsion_tau_acceptor(donor, donor_hyds, acceptor,
            gen_r2, gen_upsilon_low, gen_upsilon_high, gen_theta, None, None)
    if hbond.verbose:
        print("failed criteria")
    return False

def test_upsion_tau_acceptor(donor, donor_hyds, acceptor, r2, upsilon_low, upsilon_high,
        theta, tau, tau_sym):
    dc = donor._hb_coord
    ac = acceptor._hb_coord

    d2 = distance_squared(dc, ac)
    if d2 > r2:
        if hbond.verbose:
            print("dist criteria failed (%g > %g)" % (sqrt(d2), sqrt(r2)))
        return False

    upsilon_high = 0 - upsilon_high
    heavys = [a for a in donor.neighbors if a.element.number > 1]
    if len(heavys) != 1:
        raise AtomTypeError("upsilon tau donor (%s) not bonded to"
            " exactly one heavy atom" % donor)
    ang = angle(heavys[0]._hb_coord, dc, ac)
    if ang < upsilon_low or ang > upsilon_high:
        if hbond.verbose:
            print("upsilon criteria failed (%g < %g or %g > %g)"
                % (ang, upsilon_low, ang, upsilon_high))
        return False
    if hbond.verbose:
        print("upsilon criteria OK (%g < %g < %g)" % (upsilon_low, ang, upsilon_high))

    dp = dc
    ap = ac

    if not test_theta(dp, donor_hyds, ap, theta):
        return False

    return test_tau(tau, tau_sym, donor, dp, ap)

def don_generic(donor, donor_hyds, acceptor, sp2_O_rp2, sp3_O_rp2, sp3_N_rp2,
    sp2_O_r2, sp3_O_r2, sp3_N_r2, gen_rp2, gen_r2, min_hyd_angle, min_bonded_angle):
    if hbond.verbose:
        print("don_generic")
    dc = donor._hb_coord
    ac = acceptor._hb_coord

    acc_type = acceptor.idatm_type
    if acc_type not in type_info:
        return False

    geom = type_info[acc_type].geometry
    element = acceptor.element.name
    if element == 'O' and geom == planar:
        if hbond.verbose:
            print("planar O")
        r2 = sp2_O_r2
        rp2 = sp2_O_rp2
    elif element == 'O' and geom == tetrahedral or element == 'N' and geom == planar:
        if hbond.verbose:
            print("planar N or tet O")
        r2 = sp3_O_r2
        rp2 = sp3_O_rp2
    elif element == 'N' and geom == tetrahedral:
        if hbond.verbose:
            print("tet N")
        r2 = sp3_N_r2
        rp2 = sp3_N_rp2
    else:
        if hbond.verbose:
            print("generic acceptor")
        if acceptor.element.name == "S":
            r2 = sulphur_compensate(gen_r2)
            min_bonded_angle = min_bonded_angle - 9
        r2 = gen_r2
        rp2 = gen_rp2

    ap = acceptor._hb_coord
    dp = donor._hb_coord
    if len(donor_hyds) == 0:
        d2 = distance_squared(dc, ac)
        if d2 > r2:
            if hbond.verbose:
                print("dist criteria failed (%g > %g)" % (sqrt(d2), sqrt(r2)))
            return False
    else:
        for hyd_pos in donor_hyds:
            if distance_squared(hyd_pos, ap) < rp2:
                break
        else:
            if hbond.verbose:
                print("hyd dist criteria failed (all >= %g)" % (sqrt(rp2)))
            return False

    if hbond.verbose:
        print("dist criteria OK")

    for bonded in donor.neighbors:
        if bonded.element.number <= 1:
            continue
        bp = bonded._hb_coord
        ang = angle(bp, dp, ap)
        if ang < min_bonded_angle:
            if hbond.verbose:
                print("bonded angle too sharp (%g < %g)" % (ang, min_bonded_angle))
            return False

    if len(donor_hyds) == 0:
        if hbond.verbose:
            print("No specific hydrogen positions; default accept")
        return True

    for hyd_pos in donor_hyds:
        ang = angle(dp, hyd_pos, ap)
        if ang >= min_hyd_angle:
            if hbond.verbose:
                print("hydrogen angle okay (%g >= %g)" % (ang, min_hyd_angle))
            return True
    if hbond.verbose:
        print("hydrogen angle(s) too sharp (< %g)" % min_hyd_angle)
    return False

def don_water(donor, donor_hyds, acceptor, sp2_O_rp2, sp2_O_r2, sp2_O_theta,
        sp3_O_rp2, sp3_O_r2, sp3_O_theta, sp3_O_phi, sp3_N_rp2, sp3_N_r2,
        sp3_N_theta, sp3_N_upsilon, gen_rp2, gen_r2, gen_theta):
    if hbond.verbose:
        print("don_water")
    if len(donor_hyds) > 0:
        # hydrogens explicitly present, can immediately call don_theta_tau
        return don_theta_tau(donor, donor_hyds, acceptor, sp2_O_rp2,
            sp2_O_theta, sp3_O_rp2, sp3_O_theta, sp3_O_phi, sp3_N_rp2,
            sp3_N_theta, sp3_N_upsilon, gen_rp2, gen_theta)

    ap = acceptor._hb_coord
    dp = donor._hb_coord

    acc_type = acceptor.idatm_type
    if acc_type not in type_info:
        if hbond.verbose:
            print("Unknown acceptor type failure")
        return False

    geom = type_info[acc_type].geometry
    element = acceptor.element.name
    if element == 'O' and geom == planar:
        if hbond.verbose:
            print("planar O")
        sq = distance_squared(dp, ap)
        if sq > sp2_O_r2:
            if hbond.verbose:
                print("dist criteria failed (%g > %g)" % (sqrt(sq), sqrt(sp2_O_r2)))
            return False
    elif element == 'O' and geom == tetrahedral or element == 'N' and geom == planar:
        if hbond.verbose:
            print("planar N or tet O")
        sq = distance_squared(dp, ap)
        if sq > sp3_O_r2:
            if hbond.verbose:
                print("dist criteria failed (%g > %g)" % (sqrt(sq), sqrt(sp3_O_r2)))
            return False
    elif element == 'N' and geom == tetrahedral:
        if hbond.verbose:
            print("tet N")
        sq = distance_squared(dp, ap)
        if sq > sp3_N_r2:
            if hbond.verbose:
                print("dist criteria failed (%g > %g)" % (sqrt(sq), sqrt(sp3_N_r2)))
            return False
    else:
        if hbond.verbose:
            print("generic acceptor")
        sq = distance_squared(dp, ap)
        if sq > gen_r2:
            if hbond.verbose:
                print("dist criteria failed (%g > %g)" % (sqrt(sq), sqrt(gen_r2)))
            return False
    if hbond.verbose:
        print("dist criteria OK")

    return don_theta_tau(donor, donor_hyds, acceptor, sp2_O_rp2,
        sp2_O_theta, sp3_O_rp2, sp3_O_theta, sp3_O_phi, sp3_N_rp2,
        sp3_N_theta, sp3_N_upsilon, gen_rp2, gen_theta, is_water=True)
