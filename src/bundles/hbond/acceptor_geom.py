# vim: set expandtab ts=4 sw=4:

"""acceptor geometry testing functions"""

from chimerax.core.geometry import look_at
import numpy
from numpy import linalg
"""
from chimera import angle, Vector, cross
from chimera.bondGeom import bondPositions
from chimera.idatm import tetrahedral
"""
from . import hbond
from .common_geom import test_phi, test_theta, sulphur_compensate
"""
from commonGeom import getPhiPlaneParams
from hydpos import hydPositions
from math import sqrt
"""

def acc_syn_anti(donor, donor_hyds, acceptor, syn_atom, plane_atom, syn_r2, syn_phi,
        syn_theta, anti_r2, anti_phi, anti_theta):
    """'plane_atom' (in conjunction with acceptor, and with 'syn_atom'
        defining the 'up' direction) defines a plane.  If donor is on
        the same side as 'syn_atom', use syn params, otherwise anti params.
    """

    if hbond.verbose:
        print("acc_syn_anti")
    dc = donor.scene_coord
    dp = dc
    ac = acceptor.scene_coord
    ap = ac
    pp = plane_atom.scene_coord
    sp = syn_atom.scene_coord

    syn_xform = lookAt(ap, pp, sp - pp)
    xdp = syn_xform * dp

    phi_base_pos = pp
    phi_plane = [ap, pp, sp]

    if xdp[1] > 0.0:
        if hbond.verbose:
            print("Syn")
        if donor.element.name == "S":
            # increase distance cutoff to allow for larger vdw radius of sulphur
            # (which isn't considered as a donor in original paper)
            syn_r2 = sulphur_compensate(syn_r2)
        return test_phi_psi(dp, donor_hyds, ap, phi_base_pos, phi_plane,
                        syn_r2, syn_phi, syn_theta)
    if hbond.verbose:
        print("Anti")
    if donor.element.name == "S":
        # increase distance to compensate for sulphur radius
        anti_r2 = sulphur_compensate(anti_r2)
    return test_phi_psi(dp, donor_hyds, ap, phi_base_pos, phi_plane,
                        anti_r2, anti_phi, anti_theta)

"""
def accPhiPsi(donor, donor_hyds, acceptor, bonded1, bonded2, r2, phi, theta):
    if hbond.verbose:
        print "accPhiPsi"

    # when acceptor bonded to one heavy atom
    if not bonded1:
        # water
        bonded = acceptor.primaryNeighbors()
        if len(bonded) > 0:
            bonded1 = bonded[0]
            if len(bonded) > 1:
                bonded2 = bonded[1]
        
    phi_plane, phi_base_pos = getPhiPlaneParams(acceptor, bonded1, bonded2)
    if donor.element.name == "S":
        r2 = sulphur_compensate(r2)
    return test_phi_psi(donor.scene_coord, donor_hyds,
        acceptor.scene_coord, phi_base_pos, phi_plane, r2, phi, theta)
"""

def test_phi_psi(dp, donor_hyds, ap, bp, phi_plane, r2, phi, theta):
    if hbond.verbose:
        print("distance: %g, cut off: %g" % (linalg.norm(dp-ap), sqrt(r2)))
    if numpy.sum((dp-ap)**2) > r2:
        if hbond.verbose:
            print "dist criteria failed"
        return False
    if hbond.verbose:
        print("dist criteria OK")

    if not test_phi(dp, ap, bp, phi_plane, phi):
        return False

    return test_theta(dp, donor_hyds, ap, theta)

"""
def accThetaTau(donor, donor_hyds, acceptor, upsilonPartner, r2,
                        upsilonLow, upsilonHigh, theta):
    if hbond.verbose:
        print "accThetaTau"
    dp = donor.scene_coord
    ap = acceptor.scene_coord

    if donor.element.name == "S":
        r2 = sulphur_compensate(r2)
    if hbond.verbose:
        print "distance: %g, cut off: %g" % (linalg.norm(dp-ap), sqrt(r2))
    if numpy.sum((dp-ap)**2) > r2:
        if hbond.verbose:
            print "dist criteria failed"
        return 0
    if hbond.verbose:
        print "dist criteria okay"
    
    if upsilonPartner:
        upPos = upsilonPartner.scene_coord
    else:
        # upsilon measured from "lone pair" (bisector of attached
        # atoms)
        bondedPos = []
        for bonded in acceptor.primaryNeighbors():
            bondedPos.append(bonded.scene_coord)
        lonePairs = bondPositions(ap, tetrahedral, 1.0, bondedPos)
        bisectors = []
        for lp in lonePairs:
            bisectors.append(ap - (lp - ap))
        upPos = bisectors[0]
        for bs in bisectors[1:]:
            if hbond.verbose:
                print "Testing 'extra' lone pair"
            if test_thetaTau(dp, donor_hyds, ap, bs,
                        upsilonLow, upsilonHigh, theta):
                return 1
    return test_thetaTau(dp, donor_hyds, ap, upPos,
                        upsilonLow, upsilonHigh, theta)

def test_thetaTau(dp, donor_hyds, ap, pp, upsilonLow, upsilonHigh, theta):
    if pp:
        upsilonHigh = 0 - upsilonHigh
        upsilon = angle(pp, ap, dp)
        if upsilon < upsilonLow or upsilon > upsilonHigh:
            if hbond.verbose:
                print "upsilon (%g) failed (%g-%g)" % (
                    upsilon, upsilonLow, upsilonHigh)
            return 0
    else:
        if hbond.verbose:
            print "can't determine upsilon; default okay"
    if hbond.verbose:
        print "upsilon okay"
    return test_theta(dp, donor_hyds, ap, theta)

def accGeneric(donor, donor_hyds, acceptor, r2, minAngle):
    if hbond.verbose:
        print "generic acceptor"
    dp = donor.scene_coord
    ap = acceptor.scene_coord
    if donor.element.name == "S":
        r2 = sulphur_compensate(r2)
    if hbond.verbose:
        print "distance: %g, cut off: %g" % (linalg.norm(dp-ap), sqrt(r2))
    if numpy.sum((dp-ap)**2) > r2:
        if hbond.verbose:
            print "dist criteria failed"
        return 0
    if hbond.verbose:
        print "dist criteria okay"
     
    ap = acceptor.scene_coord
    dp = donor.scene_coord
    for bonded in acceptor.primaryNeighbors():
        bp = bonded.scene_coord
        if hbond.verbose:
            print "angle: %g" % angle(bp, ap, dp)
        ang = angle(bp, ap, dp)
        if ang < minAngle:
            if hbond.verbose:
                print "angle too sharp (%g < %g)" % (ang,
                                minAngle)
            return 0
    if hbond.verbose:
        print "angle(s) okay (all > %g)" % minAngle
    return 1
"""
