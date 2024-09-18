# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def local_distance_difference_test(ref_xyz, xyz, paired_mask,
                                   r0 = 15, thresholds = (0.5, 1.0, 2.0, 4.0),
                                   score_missing = False):
    '''
    Compute LDDT (local distance difference test) score for each position for each sequence
    of coordinates in the xyz array (N x 3 size).  The ref_xyz array is typically C-alpha coordinates
    of a reference structure while xyz can be one or more sequences to score for conformation
    similarity (M x N x 3 size).  Only non-zero the positions in the paired_mask array have valid xyz position
    data.  If a ref_xyz value has no paired coordinate in the xyz array then it can be ignored
    (score_missing = False) or it can decrease the LDDT score (score_missing = True).
    '''
    n = len(ref_xyz)
    m = len(xyz)
    nt = len(thresholds)
    from numpy import zeros, sqrt, abs, int32
    total = zeros((m,n), int32)
    count = zeros((nt,m,n), int32)
    for i in range(n):
        # Find close positions to reference position i.
        dref = ref_xyz - ref_xyz[i]
        dist = sqrt((dref*dref).sum(axis=1))
        close = (dist <= r0)			# Size n
        close[i] = False
        rd = dist[close]			# Size c
        d = xyz[:,close,:] - xyz[:,i,:].reshape((m,1,3))   # Size m,c,3
        td = sqrt((d*d).sum(axis=2))		# m,c
        derror = abs(td - rd)			# m,c
        pmask = paired_mask[:,close]		# m,c
        for k in range(m):
            if paired_mask[k,i]:
                kpmask = pmask[k]
                dkerror = derror[k,kpmask]
                for j,t in enumerate(thresholds):
                    count[j,k,i] += (dkerror <= t).sum()
        total[:,i] += close.sum() if score_missing else pmask.sum(axis=1)
    total[total == 0] = 1      # Avoid divide by zero
    scores = count.sum(axis=0)/total
    scores /= nt
    return scores
