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

def align_points(xyz, ref_xyz):
    '''
    Supported API.
    Computes rotation and translation to align one set of points with another.
    The sum of the squares of the distances between corresponding positions is
    minimized.

    Parameters
    ----------
    xyz : numpy float n by 3 array
    ref_xyz : numpy float n by 3 array

    Returns
    -------
    place : :py:class:`.Place`
      transformation taking xyz onto ref_xyz.
    rms : float
      root mean square (RMS) value calculated from distances between aligned points.
    '''

    # TODO: Testing if float64 has less roundoff error.
    from numpy import float64
    xyz = xyz.astype(float64)
    ref_xyz = ref_xyz.astype(float64)

    center = xyz.mean(axis = 0)
    ref_center = ref_xyz.mean(axis = 0)
    if len(xyz) == 1:
        # No rotation if aligning one point.
        from numpy import array, float64
        tf = array(((1,0,0,0),(0,1,0,0),(0,0,1,0)), float64)
        tf[:,3] = ref_center - center
        rms = 0
    else:
        Si = xyz - center
        Sj = ref_xyz - ref_center
        from numpy import dot, transpose, trace, zeros, empty, float64, identity
        Sij = dot(transpose(Si), Sj)
        M = zeros((4,4), float64)
        M[:3,:3] = Sij
        MT = transpose(M)
        trM = trace(M)*identity(4, float64)
        P = M + MT - 2 * trM
        P[3, 0] = P[0, 3] = M[1, 2] - M[2, 1]
        P[3, 1] = P[1, 3] = M[2, 0] - M[0, 2]
        P[3, 2] = P[2, 3] = M[0, 1] - M[1, 0]
        P[3, 3] = 0.0

        # Find the eigenvalues and eigenvectors
        from numpy import linalg
        evals, evecs = linalg.eig(P)    # eigenvectors are columns
        q = evecs[:,evals.argmax()]
        R = quaternion_rotation_matrix(q)
        tf = empty((3,4), float64)
        tf[:,:3] = R
        tf[:,3] = ref_center - dot(R,center)

        # Compute RMS
        # TODO: This RMS calculation has rather larger round-off errors
        #  probably from subtracting two large numbers.
        rms2 = (Si*Si).sum() + (Sj*Sj).sum() - 2 * (transpose(R)*Sij).sum()
        from math import sqrt
        rms = sqrt(rms2/len(Si)) if rms2 >= 0 else 0
#        df = dot(Sj,R) - Si
#        arms = sqrt((df*df).sum()/len(Si))
#        print (rms, arms)

    from . import Place
    p = Place(tf)

    return p, rms

def quaternion_rotation_matrix(q):
    l,m,n,s = q
    l2 = l*l
    m2 = m*m
    n2 = n*n
    s2 = s*s
    lm = l*m
    ln = l*n
    ls = l*s
    ns = n*s
    mn = m*n
    ms = m*s
    m = ((l2 - m2 - n2 + s2, 2 * (lm - ns), 2 * (ln + ms)),
         (2 * (lm + ns), - l2 + m2 - n2 + s2, 2 * (mn - ls)),
         (2 * (ln - ms), 2 * (mn + ls), - l2 - m2 + n2 + s2))
    return m
