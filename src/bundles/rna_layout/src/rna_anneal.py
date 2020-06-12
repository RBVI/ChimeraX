# -----------------------------------------------------------------------------
# Layout single strand RNA with specified base pairing.
# Tries to specify relative positions of bases and then numerically optimize
# by updating one base at a time.
#
# Proved way too slow to use for 10,000 nt HIV RNA.
#
stack_spacing = 4       # Angstroms
stack_rotation = 30     # degrees
pair_shift = 4
pair_separation = 2
pair_rotation = 180

# -----------------------------------------------------------------------------
#
class RNA_Model:

    def __init__(self, sequence):

        from _surface import SurfaceModel
        surf = SurfaceModel()
        from chimera import openModels
        openModels.add([surf])
        ntlist = [Nucleotide(sym, i+1, surf) for i, sym in enumerate(sequence)]
        self.stack_bases(ntlist, self.stacking_transform())
        self.nucleotides = ntlist

    def add_contact(self, nt1, nt2, tf):

        from Matrix import invert_matrix
        nt1.contacts.append((tf, nt2))
        nt2.contacts.append((invert_matrix(tf), nt1))

    def optimize_contacts(self):

        ntlist = self.nucleotides
        for nt in ntlist:
            nt.optimize_contacts()
        from math import sqrt
        rms = sqrt(sum(nt.contact_rms2() for nt in ntlist)/len(ntlist))
        print rms

    def stack_bases(self, ntlist, stf):

        import Matrix as M
        for i, nt in enumerate(ntlist[:-1]):
            cf = nt.coordinate_frame()
            tf = M.multiply_matrices(cf, stf)
            ntnext = ntlist[i+1]
#            ntnext.place(tf)
            self.add_contact(nt, ntnext, stf)

    def stacking_transform(self):

        axis = (0,0,1)
        angle = stack_rotation
        shift = (0,0,stack_spacing)
        import Matrix as M
        tf = M.multiply_matrices(M.translation_matrix(shift),
                                 M.rotation_transform(axis, angle))
        return tf

    def pair_bases(self, index_pairs, ptf):

        ntlist = self.nucleotides
        import Matrix as M
        for i1,i2 in index_pairs:
            nt1 = ntlist[i1]
            nt2 = ntlist[i2]
            cf = nt1.coordinate_frame()
            tf = M.multiply_matrices(cf, ptf)
            nt2.place(tf)
            self.add_contact(nt1, nt2, ptf)

    def pairing_transform(self):

        axis = (0,0,1)
        angle = pair_rotation
        shift = (-pair_separation, pair_shift,0)
        import Matrix as M
        tf = M.multiply_matrices(M.translation_matrix(shift),
                                 M.rotation_transform(axis, angle))
        return tf
        
# -----------------------------------------------------------------------------
#
colors = {
    'A': (1,1,.5,1),     # yellow
    'C': (1,.5,.5,1),
    'G': (1,.5,1,1),     # green
    'T': (.5,1,1,1),     # cyan
    'U': (.5,1,1,1),     # cyan
}

# -----------------------------------------------------------------------------
#
class Nucleotide:

    thickness = 2       # half stacking distance
    length = 9
    width = 4
    def __init__(self, symbol, seqnum, surface):

        self.symbol = symbol
        self.seqnum = seqnum
        self.contacts = []

        varray, tarray = self.shape()
        varray[:,2] += stack_spacing*seqnum
        color = colors[symbol]
        self.surface_piece = p = surface.addPiece(varray, tarray, color)
        p.oslName = '%d' % seqnum

    def shape(self):

        z,y,x = self.thickness, self.width, self.length
        from numpy import array, float32, int32
        varray = array(((0,0,0),(x,0,0),(x,y,0),(0,y,0),
                        (0,0,z),(x,0,z),(x,y,z),(0,y,z)), float32)
        tarray = array(((0,2,1),(0,3,2), # -z
                        (4,5,6),(6,7,4), # +z
                        (0,4,7),(0,7,3), # -x
                        (1,6,5),(1,2,6), # +x
                        (0,1,5),(0,5,4), # -y
                        (3,6,2),(3,7,6), # +y
                        ), int32)
        return varray, tarray

    def reference_points(self):

        z,y,x = 0.5*self.thickness, self.width, self.length
        p = ((0,0,z), (x,0,z), (0,y,z), (x,y,z))
        return p

    def place(self, transform):

        varray, tarray = self.shape()
        import _contour
        _contour.affine_transform_vertices(varray, transform)
        self.surface_piece.geometry = varray, tarray

    def coordinate_frame(self):

        varray = self.surface_piece.geometry[0]
        o = varray[0]
        x,y,z = varray[1] - o, varray[3] - o, varray[4] - o
        from Matrix import normalize_vector
        x,y,z = [normalize_vector(v) for v in (x,y,z)]
        tf = ((x[0],y[0],z[0],o[0]),
              (x[1],y[1],z[1],o[1]),
              (x[2],y[2],z[2],o[2]))
        return tf

    def optimize_contacts(self):

        xyz_ideal, xyz_actual = self.contact_pairs()
        from chimera import match
        xf, rms = match.matchPositions(xyz_actual, xyz_ideal)
        import Matrix as M
        tf = M.xform_matrix(xf)
        ptf = M.multiply_matrices(tf, self.coordinate_frame())
        self.place(ptf)

    def contact_pairs(self):
        
        xyz_ideal = []
        xyz_actual = []
        rp = self.reference_points()
        import Matrix as M
        for tf0, nt in self.contacts:
            cf = self.coordinate_frame()
            ntcf = nt.coordinate_frame()
            tf = M.multiply_matrices(cf, tf0)
            ntrp = nt.reference_points()
            rpt0 = M.apply_matrix(tf, ntrp)
            xyz_ideal.extend(rpt0)
            rpt = M.apply_matrix(ntcf, ntrp)
            xyz_actual.extend(rpt)
            # Add symmetric pairs.
            rpt0 = M.apply_matrix(cf, rp)
            xyz_ideal.extend(rpt0)
            tf = M.multiply_matrices(ntcf, M.invert_matrix(tf0))
            rpt = M.apply_matrix(tf, rp)
            xyz_actual.extend(rpt)
            
        return xyz_ideal, xyz_actual

    def contact_rms2(self):

        xyz_ideal, xyz_actual = self.contact_pairs()
        import numpy
        dxyz = numpy.subtract(xyz_actual, xyz_ideal)
        dxyz *= dxyz
        rms2 = dxyz.sum() / len(xyz_ideal)
        return rms2
        
# -----------------------------------------------------------------------------
#
def open_rna(fasta_path, pairings_path):

    n = 363
    f = open(fasta_path, 'r')
    f.readline()        # header
    seq = f.read()
    f.close()
    seq = seq.replace('\n', '')
    seq = seq[:n]
    m = RNA_Model(seq)
    pairs = read_base_pairs(pairings_path)
    pairs = [(i1,i2) for i1,i2 in pairs if i1 < n and i2 < n]
    m.pair_bases(pairs, m.pairing_transform())
    return m
        
# -----------------------------------------------------------------------------
#
def read_base_pairs(pairings_path):

    f = open(pairings_path, 'r')
    lines = f.readlines()
    f.close()
    pairs = []
    for line in lines:
        s, e, l = [int(f) for f in line.split()]
        pairs.extend([(s-1+i, e-1-i) for i in range(l)])
    return pairs

# -----------------------------------------------------------------------------
#
rna = open_rna('hiv-rna/pairings.fasta', 'hiv-rna/pairings.txt')

from Accelerators import add_accelerator
add_accelerator('0', 'Minimize shape',
                lambda rna=rna: rna.optimize_contacts())

def opt(rna = rna):
    import chimera
    from chimera import triggers
    if hasattr(chimera, 'minrnatrigger'):
        triggers.deleteHandler('new frame', chimera.minrnatrigger)
        delattr(chimera, 'minrnatrigger')
    else:
        mcb = lambda t,d,e,rna=rna: rna.optimize_contacts()
        chimera.minrnatrigger = triggers.addHandler('new frame', mcb, None)

add_accelerator('-', 'Start/stop shape optimization', opt)
