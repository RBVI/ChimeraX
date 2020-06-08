# -----------------------------------------------------------------------------
# Create a new path of specified maximum curvature that follows a given path
# to the extent possible and uses equispaced markers.  Marker colors and radii
# are copied and markers are linked if the input path markers are linked.
#
# This is for converting a single stranded RNA path into a duplex DNA path.
#
def smooth_path(path_markers, min_radius, spacing,
                kink_interval = None, kink_radius = None, name = 'smooth path'):

    from VolumePath import Marker_Set, Link
    ms = Marker_Set(name)

    mlist = []
    xyzp = prev_step = None
    mcp = mp = None
    for i,m in enumerate(path_markers):
        r = kink_radius if kink_interval and i > 0 and i % kink_interval == 0 else min_radius
        xyz = next_point(m.xyz(), xyzp, prev_step, r, spacing)
        mc = ms.place_marker(tuple(xyz), m.rgba(), m.radius())
        mlist.append(mc)
        if mcp and mp:
            l = m.linked_to(mp)
            if l:
                Link(mc, mcp, l.rgba(), l.radius())
        mcp, mp = mc, m
        if not xyzp is None:
            prev_step = xyz - xyzp
        xyzp = xyz

    return ms

# -----------------------------------------------------------------------------
#
def next_point(xyz, xyzp, prev_step, min_radius, spacing):

    from numpy import array
    import Matrix as M

    xyz = array(xyz)
    if xyzp is None:
        xyzn = xyz
    elif prev_step is None:
        d = M.norm(xyz - xyzp)
        xyzn = xyzp + (spacing/d) * (xyz - xyzp)
    else:
        d = M.norm(xyz - xyzp)
        ip = M.inner_product(prev_step, xyz - xyzp)
        from math import asin, sin, cos
        a = 2 *asin(0.5*spacing/min_radius)
        min_sin, min_cos = sin(a), cos(a)
        if ip < d*spacing*min_cos:
            c = M.cross_product(prev_step, xyz - xyzp)
            step_perp = array(M.normalize_vector(M.cross_product(c, prev_step)))
            step = min_cos*prev_step + (min_sin*spacing)*step_perp
            xyzn = xyzp + step
        else:
            xyzn = xyzp + (spacing/d) * (xyz - xyzp)
    return xyzn


# -----------------------------------------------------------------------------
#
def duplex_dna_atomic_model(path, seq, dna_templates_file, twist = 36):

    from chimera import Molecule, openModels
    mol = Molecule()
    mol.name = 'DNA'

    # Calculate twist along path.
    tflist = twist_path_transforms(path, twist)

    # Find template residues for sequence
    rmap, tmol = basepair_templates(dna_templates_file)
    n = len(path)
    rplist = [rmap[seq[i]] for i in range(n)]
    
    # Place base pairs at each position
    rtflist = zip(rplist, tflist)
    from rna_layout import copy_residue
    r1list = [copy_residue(r1, i+1, tf, mol)
              for i, ((r1,r2), tf) in enumerate(rtflist)]
    r2list = [copy_residue(r2, i+1, tf, mol)
              for i, ((r1,r2), tf) in enumerate(reversed(rtflist))]

    # Join consecutive residues
    connect_backbone(r1list)
    connect_backbone(r2list)

    openModels.add([mol], noprefs = True)
    return mol

# -----------------------------------------------------------------------------
#
def twist_path_transforms(path, twist):
    
    tflist = []
    n = len(path)

    from math import cos, sin, pi
    a = twist * pi/180
    ct, st = cos(a), sin(a)

    import Matrix as M
    from numpy import array
    for i in range(n):
        if i+1 < n:
            z = path[i+1] - path[i]
        y = None if i == 0 else ct*array(f[1]) - st*array(f[0])
        f = M.orthonormal_frame(z, y)
        ((x1,y1,z1), (x2,y2,z2), (x3,y3,z3)) = f
        x0,y0,z0 = path[i]
        tf = ((x1, x2, x3, x0), (y1, y2, y3, y0), (z1, z2, z3, z0))
        tflist.append(tf)

    return tflist

# -----------------------------------------------------------------------------
#
def connect_backbone(rlist):

    rp = rlist[0]
    mol = rp.molecule
    for r in rlist[1:]:
        ap = rp.findAtom("O3'")
        a = r.findAtom('P')
        if a and ap:
            mol.newBond(ap,a)
        rp = r

# -----------------------------------------------------------------------------
#
def basepair_templates(dna_templates_file, chain = 'A'):

    import rna_layout as RL
    t = RL.template_molecule(dna_templates_file)

    rpairs = []
    pmap = dict((r.id.position, r) for r in t.residues if r.id.chainId == chain)
    for r in t.residues:
        p = r.id.position
        if r.id.chainId != chain and p in pmap:
            rpairs.append((pmap[p], r))

    rmap = dict((r1.type, (r1,r2)) for r1, r2 in rpairs)

    synonyms = (('A','DA'), ('C','DC'), ('G','DG'), ('U','T','DT'))
    for slist in synonyms:
        sc = [s for s in slist if s in rmap]
        if len(sc) == 1:
            r1r2 = rmap[sc[0]]
            for s in slist:
                rmap[s] = r1r2

    return rmap, t

# -----------------------------------------------------------------------------
#
def nick_backbone(mol, spacing, cid):

    rlist = [r for r in mol.residues if r.id.chainId == cid]
    n = len(rlist)
    for i in range(spacing-1, n-1, spacing):
        rp, r = rlist[i], rlist[i+1]
        ap = rp.findAtom("O3'")
        a = r.findAtom('P')
        if a and ap:
            b = ap.findBond(a)
            if b:
                mol.deleteBond(b)

# -----------------------------------------------------------------------------
#
def color_intervals(mol, interval, cid, rgba_even, rgba_odd):

    from chimera import MaterialColor
    ce, co = MaterialColor(*rgba_even), MaterialColor(*rgba_odd)
    rlist = [r for r in mol.residues if r.id.chainId == cid]
    for i,r in enumerate(rlist):
        c = ce if (i/interval) % 2 else co
        r.ribbonColor = c
        for a in r.atoms:
            a.color = c
    
# -----------------------------------------------------------------------------
# Path is an N by 3 array of x,y,z positions.
#
def make_dna_following_path(path, seq, polymer_type = 'DNA'):

    template_file = ('rna-dna-templates.pdb' if polymer_type == 'RNADNA'
                     else 'dna-templates.pdb')
    mol = duplex_dna_atomic_model(path, seq, template_file)
    return mol

# -----------------------------------------------------------------------------
#
def make_hiv_dna_for_selected_path():

    import VolumePath as VP
    mlist = list(VP.selected_markers())
    mlist.sort(lambda m1,m2: cmp(m1.id, m2.id))
    if len(mlist) == 0:
        return

    dna_mset = smooth_path(mlist, 50, 3.33)
#                              kink_interval = 10, kink_radius = 5)
    dna_markers = dna_mset.markers()

    seq_path = '/usr/local/src/staff/goddard/presentations/hiv-rna-may2011/hiv-pNL4-3.fasta'
    seq_start = 455
    import rna_layout
    seq = rna_layout.read_fasta(seq_path)[seq_start-1:]

    from numpy import array
    path = array([m.xyz() for m in dna_markers])
    mol = make_dna_following_path(path, seq, 'DNA')

#    nick_backbone(mol, 10, 'B')
#    color_intervals(mol, 10, 'B', (1,1,0,1), (1,0.5,0,1))
