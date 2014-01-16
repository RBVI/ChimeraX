# -----------------------------------------------------------------------------
# Read in a Neuron trace in Stockley-Wheal-Cannon (SWC) format.
#
# From NeuroMorpho.org FAQ
# What is SWC format?
#
# The three dimensional structure of a neuron can be represented in a
# SWC format (Cannon et al., 1998). SWC is a simple Standardized
# format. Each line has 7 fields encoding data for a single neuronal
# compartment:
#
#  an integer number as compartment identifier
#  type of neuronal compartment 
#     0 - undefined
#     1 - soma
#     2 - axon
#     3 - basal dendrite
#     4 - apical dendrite
#  x coordinate of the compartment
#  y coordinate of the compartment
#  z coordinate of the compartment
#  radius of the compartment
#  parent compartment
#
# Every compartment has only one parent and the parent compartment for
# the first point in each file is always -1 (if the file does not
# include the soma information then the originating point of the tree
# will be connected to a parent of -1). The index for parent
# compartments are always less than child compartments. Loops and
# unconnected branches are excluded. All trees should originate from
# the soma and have parent type 1 if the file includes soma
# information. Soma can be a single point or more than one point. When
# the soma is encoded as one line in the SWC, it is interpreted as a
# "sphere". When it is encoded by more than 1 line, it could be a set
# of tapering cylinders (as in some pyramidal cells) or even a 2D
# projected contour ("circumference").
#
# "Soma format representation in NeuroMorpho.Org as of version 5.3" says
# that soma will be represented by exactly 3 points, the second two being
# displayed +/- radius along y axis.  Probably best to display just the
# first soma point.
# 
def read_swc(path):
    '''
    Read a Stockley-Wheal-Cannon (SWC) format traced neuron file
    and create a fake molecule model using atoms and bonds to represent
    the neuron.
    '''
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    points = parse_swc_points(lines)
    i2a = {}

    tcolors = {
        1:(255,255,255,255),     # soma, white
        2:(128,128,128,255),  # axon gray
        3:(0,255,0,255),     # basal dendrite, green
        4:(255,0,255,255),     # apical dendrite, magenta
    }
    other_color = (1,1,0,1)   # yellow

    from numpy import empty, float32, array, int32, ones, uint8, zeros
    np = len(points)
    xyz = empty((np,3), float32)
    radii = empty((np,), float32)
    colors = empty((np,4), uint8)
    rnums = empty((np,), int32)
    bonds = []
    brad = []
    for a,(n,t,x,y,z,r,pid) in enumerate(points):
        xyz[a,:] = (x,y,z)
        radii[a] = r
        colors[a,:] = tcolors.get(t, other_color)
        rnums[a] = n
        i2a[n] = a
        if pid in i2a:
            ap = i2a[pid]
            bonds.append((a, ap))
            brad.append(radii[a])
    element_nums = ones((np,), uint8)
    chain_ids = zeros((np,), 'S4')
    res_names = zeros((np,), 'S4')
    atom_names = zeros((np,), 'S4')
    from ..molecule import Molecule
    m = Molecule(path, xyz, element_nums, chain_ids, rnums, res_names, atom_names)
    scale = 10000.0    # Convert microns to Angstroms
    m.xyz *= scale
    m.atom_colors = colors
    m.radii = radii
    m.radii *= scale
    m.bonds = array(bonds, int32)
    m.bond_radii = array(brad, float32)
    m.bond_radii *= scale
    m.ball_scale = 1
    m.set_atom_style('ballstick')
    from ..ui import show_info
    show_info('Read neuron trace %s, %d nodes' % (path, np))
    return m
    
def parse_swc_points(lines):

    points = []
    for i, line in enumerate(lines):
        sline = line.strip()
        if sline.startswith('#'):
            continue    # Comment line
        fields = sline.split()
        if len(fields) != 7:
            msg = 'Line %d does not have 7 fields: "%s"' % (i, line)
            from ..ui import show_info
            show_info(msg)
            continue
        try:
            n = int(fields[0])      # id
            t = int(fields[1])      # type
            x,y,z = (float(f) for f in fields[2:5])
            r = float(fields[5])    # radius
            pid = int(fields[6])    # parent id, or -1 if no parent
        except ValueError:
            msg = 'Error parsing line %d: "%s"' % (i, line)
            from ..ui import show_info
            show_info(msg)
            continue
        if t == 1 and pid != -1:
            continue    # Drop all but first soma point
        points.append((n,t,x,y,z,r,pid))

    return points
