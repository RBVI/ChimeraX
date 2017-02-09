def fitModels(m0, m1, fraction=0.5):
        """sieveFitModels takes two conformers and identifies the
        fraction of the residues that produce the minimum RMSD."""

        rList0 = m0.residues
        rList1 = m1.residues

        return fitResidues(rList0, rList1, fraction)

def fitResidues(rList0, rList1, fraction=0.5, maxrmsd=0.1):
        """sieveFitModels takes two lists of corresponding residues
        and identifies the fraction of the residues that produce
        the minimum RMSD."""
        # make sure the residues correspond
        if len(rList0) != len(rList1):
                raise ValueError("matching different number of residues")
        #for r0, r1 in zip(rList0, rList1):
        #        if r0.type != r1.type:
        #                raise ValueError("matching residues of different types")
        from .util import getAtomList
        aList0 = getAtomList(rList0)
        aList1 = getAtomList(rList1)
        keep = int(len(rList0) * fraction)
        while len(aList0) > keep:
                if not sieve(aList0, aList1, maxrmsd):
                        break
        return ([ a.residue for a in aList0 ], [ a.residue for a in aList1 ])

svt = 0
def sieve(aList0, aList1, maxrmsd):
        from time import time
        t0 = time()
        from numpy import array, subtract, inner, add, argmax, transpose, multiply
        position0 = array([a.coord for a in aList0])        # fixed
        position1 = array([a.coord for a in aList1])        # movable
        from chimerax.core.geometry import align_points
        p, rms = align_points(position1, position0)
        if rms < maxrmsd:
                return False
        Si = subtract(position0, position0.mean(axis=0))
        Sj = subtract(position1, position1.mean(axis=0))
        rot = p.matrix[:,:3]        # Rotation part of alignment
        result = inner(Si, rot)
        d = subtract(result, Sj)
        dsq = add.reduce(transpose(multiply(d, d)))
        worst = argmax(dsq)
        del aList0[worst]
        del aList1[worst]
        t1 = time()
        global svt
        svt += t1-t0
        return True
