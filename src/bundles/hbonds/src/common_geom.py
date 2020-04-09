# vim: set expandtab ts=4 sw=4:


from numpy import zeros, dot
from chimerax.geometry import angle, cross_product, normalize_vector
from .hydpos import hyd_positions
from . import hbond

SULFUR_COMP = 0.35

class ConnectivityError(ValueError):
    pass

class AtomTypeError(ValueError):
    pass

def test_phi(dp, ap, bp, phi_plane, phi):
    if phi_plane:
        normal = normalize_vector(cross_product(phi_plane[1] - phi_plane[0],
                        phi_plane[2] - phi_plane[1]))
        D = dot(normal, phi_plane[1])
        bproj = project(bp, normal, D)
        aproj = project(ap, normal, D)
        dproj = project(dp, normal, D)

        ang = angle(bproj, aproj, dproj)
        if ang < phi:
            if hbond.verbose:
                print("phi criteria failed (%g < %g)" % (ang, phi))
            return False
        if hbond.verbose:
            print("phi criteria OK (%g >= %g)" % (ang, phi))
    else:
        if hbond.verbose:
            print("phi criteria irrelevant")
    return True

def get_phi_plane_params(acceptor, bonded1, bonded2):
    ap = acceptor._hb_coord
    if bonded2:
        # two principal bonds
        phi_plane = [ap]
        mid_point = zeros(3)
        for bonded in [bonded1, bonded2]:
            pt = bonded._hb_coord
            phi_plane.append(pt)
            mid_point = mid_point + pt
        phi_base_pos = mid_point / 2.0
    elif bonded1:
        # one principal bond
        phi_base_pos = bonded1._hb_coord
        grand_bonded = list(bonded1.neighbors)
        if acceptor in grand_bonded:
            grand_bonded.remove(acceptor)
        else:
            raise ValueError("Acceptor %s not found in bond list of %s" % (acceptor, bonded1))
        if len(grand_bonded) == 1:
            phi_plane = [ap, bonded1._hb_coord, grand_bonded[0]._hb_coord]
        elif len(grand_bonded) == 2:
            phi_plane = [ap]
            for gb in grand_bonded:
                phi_plane.append(gb._hb_coord)
        elif len(grand_bonded) == 0:
            # e.g. O2
            phi_plane = None
        else:
            raise ConnectivityError("Wrong number of grandchild"
                    " atoms for phi/psi acceptor %s" % acceptor)
    else:
        return None, None
    return phi_plane, phi_base_pos

def project(point, normal, D):
    # project point into plane defined by normal and D.
    return point - normal * (dot(normal, point) - D)

def test_tau(tau, tau_sym, don_acc, dap, op):
    if tau is None:
        if hbond.verbose:
            print("tau test irrelevant")
        return True

    # sulfonamides and phosphonamides can have bonded NH2 groups that are planar enough
    # to be declared Npl, so use the hydrogen positions to determine planarity if possible
    if tau_sym == 4:
        bonded_pos = hyd_positions(don_acc)
    else:
        # since we expect tetrahedral hydrogens to be oppositely aligned from the attached
        # tetrahedral center, we can't use their positions for tau testing
        bonded_pos = []
    heavys = [a for a in don_acc.neighbors if a.element.number > 1]
    if 2 * len(bonded_pos) != tau_sym:
        bonded_pos = hyd_positions(heavys[0], include_lone_pairs=True)
        for b in heavys[0].neighbors:
            if b == don_acc or b.element.number < 2:
                continue
            bonded_pos.append(b._hb_coord)
        if not bonded_pos:
            if hbond.verbose:
                print("tau indeterminate; default okay")
            return True

    if 2 * len(bonded_pos) != tau_sym:
        raise AtomTypeError("Unexpected tau symmetry (%d, should be %d) for donor/acceptor %s" % (
                2 * len(bonded_pos), tau_sym, don_acc))

    normal = normalize_vector(heavys[0]._hb_coord - dap)

    if tau < 0.0:
        test = lambda ang, t=tau: ang <= 0.0 - t
    else:
        test = lambda ang, t=tau: ang >= t

    proj_other_pos = project(op, normal, 0.0)
    proj_da_pos = project(dap, normal, 0.0)
    for bpos in bonded_pos:
        proj_bpos = project(bpos, normal, 0.0)
        ang = angle(proj_other_pos, proj_da_pos, proj_bpos)
        if test(ang):
            if tau < 0.0:
                if hbond.verbose:
                    print("tau okay (%g < %g)" % (ang, -tau))
                return True
        else:
            if tau > 0.0:
                if hbond.verbose:
                    print("tau too small (%g < %g)" % (ang, tau))
                return False
    if tau < 0.0:
        if hbond.verbose:
            print("all taus too big (> %g)" % -tau)
        return False

    if hbond.verbose:
        print("all taus acceptable (> %g)" % tau)
    return True

def test_theta(dp, donor_hyds, ap, theta):
    if len(donor_hyds) == 0:
        if hbond.verbose:
            print("no hydrogens for theta test; default accept")
        return True
    for hyd_pos in donor_hyds:
        ang =  angle(ap, hyd_pos, dp)
        if ang >= theta:
            if hbond.verbose:
                print("theta okay (%g >= %g)" % (ang, theta))
            return True
        if hbond.verbose:
            print("theta failure (%g < %g)" % (ang, theta))

    return False

def sulphur_compensate(base_r2):
    from .hbond import _compute_cache
    try:
        return _compute_cache[base_r2]
    except KeyError:
        pass
    import math
    r = math.sqrt(base_r2)
    r = r + SULFUR_COMP
    new_r2 = r * r
    _compute_cache[base_r2] = new_r2
    return new_r2
