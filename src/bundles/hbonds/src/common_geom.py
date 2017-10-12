# vim: set expandtab ts=4 sw=4:


from numpy import cross, linalg, zeros, dot
from chimerax.core.geometry import angle
from . import hbond

SULFUR_COMP = 0.35

class ConnectivityError(ValueError):
    pass

class AtomTypeError(ValueError):
    pass

def test_phi(dp, ap, bp, phi_plane, phi):
    if phi_plane:
        normal = cross(phi_plane[1] - phi_plane[0],
                        phi_plane[2] - phi_plane[1])
        normal = normal / linalg.norm(normal)
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
