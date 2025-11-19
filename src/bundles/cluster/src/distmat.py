# vim: set expandtab shiftwidth=4 softtabstop=4:

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

"A Distance Matrix class with support for selecting best representative."

# See
#    Sutcliffe, M.J., "Representing an ensemble of NMR-derived protein
#    structures by a single structure." Protein Science, 1993 2:936-944.
# for (brief) description of algorithm for selecting best representative.
# In the description on p.943, the author states that the "origin
# corresponds to the average of the ensemble."  I believe the correct
# word is "centroid" rather than "origin", given that there are an
# infinite number of possible embeddings and they do not necessarily
# share the same origin.

class DistanceMatrix:

    def __init__(self, size):
        import numpy
        self.size = size
        self.dm = numpy.zeros(shape=(size, size), dtype='d')
        self.dm.fill(-1.0)
        for i in range(size):
            self.set(i, i, 0.0)

    def __repr__(self):
        return repr(self.dm)

    def __str__(self):
        return str(self.dm)

    def get(self, i, j):
        return self.dm[i, j]

    def set(self, i, j, value):
        self.dm[i, j] = value
        self.dm[j, i] = value

    def representative(self):
        from chimerax.core.errors import LimitationError
        import numpy
        try:
            return self._representative()
        except LimitationError:
            return numpy.argmin(numpy.sum(self.dm, axis=1))

    def _representative(self):
        # Make sure distance matrix is completely filled
        minval = min(self.dm.flat)
        if minval < 0:
            raise ValueError("incomplete distance matrix")

        # Convert distance matrix into metric matrix
        import numpy
        from numpy import linalg
        import math
        rho0i = numpy.zeros_like(self.dm)
        for i in range(self.size):
            rho0i[i,:] = self.dm[0, i]
        rho0j = numpy.zeros_like(self.dm)
        for j in range(self.size):
            rho0j[:,j] = self.dm[0, j]
        m = (rho0i + rho0j - self.dm) / 2.0
        mm = m[1:,1:]

        # Embed into lower dimension
        evals, evecs = linalg.eig(mm)
        L = numpy.zeros_like(mm)
        from chimerax.core.errors import LimitationError
        try:
            for i in range(self.size - 1):
                L[i,i] = math.sqrt(evals[i])
        except ValueError:
            raise LimitationError("unexpected negative eigenvalues")
        X = numpy.dot(L, numpy.transpose(evecs))
        coord = [ numpy.zeros(shape=(self.size - 1,), dtype='d') ]
        for i in range(self.size - 1):
            coord.append(X[:,i])

        # Verify distance matrix is properly reproduced
        delta = self.dm - make_distance_matrix(coord).dm
        if min(delta.flat) > 1e-5:
            raise LimitationError("cannot embed distance matrix")

        # Find point nearest centroid as representative
        coord = numpy.array(coord)
        centroid = coord.mean(axis=0)
        delta = coord - centroid
        dsq = numpy.sum(delta * delta, axis=1)
        return numpy.argmin(dsq)

def make_distance_matrix(v):
    import numpy
    dm = DistanceMatrix(len(v))
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            d = v[i] - v[j]
            dm.set(i, j, numpy.dot(d, d))
    return dm
