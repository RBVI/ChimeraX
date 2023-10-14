# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

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
def read_swc(session, path):

    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    points = parse_swc_points(lines)
    i2m = {}
    from chimerax.markers import MarkerSet, create_link
    from os.path import basename
    name = basename(path)
    mset = MarkerSet(session, name)
    tcolors = {
        1:(255,255,255,255),     # soma, white
        2:(128,128,128,255),  # axon gray
        3:(0,255,0,255),     # basal dendrite, green
        4:(255,0,255,255),     # apical dendrite, magenta
    }
    other_color = (255,255,0,255)   # yellow
    for n,t,x,y,z,r,pid in points:
        if r < 0:
            r = 0.5
        color = tcolors.get(t, other_color)
        i2m[n] = m = mset.create_marker((x,y,z), color, r, id = n)
        if pid in i2m:
            m2 = i2m[pid]
            rlink = min(r, m2.radius)
            create_link(m, m2, color, rlink)
    msg = 'Opened neuron traces %s' % name
    return [mset], msg
    
def parse_swc_points(lines):

    points = []
    for i, line in enumerate(lines):
        sline = line.strip()
        if sline.startswith('#'):
            continue    # Comment line
        fields = sline.split()
        if len(fields) != 7:
            msg = 'Line %d does not have 7 fields: "%s"' % (i, line)
            from replyobj import info
            info(msg)
            continue
        try:
            n = int(fields[0])      # id
            t = int(fields[1])      # type
            x,y,z = (float(f) for f in fields[2:5])
            r = float(fields[5])    # radius
            pid = int(fields[6])    # parent id, or -1 if no parent
        except ValueError:
            msg = 'Error parsing line %d: "%s"' % (i, line)
            from replyobj import info
            info(msg)
            continue
        if r == 0:
            continue    # Drop radius 0 points
        if t == 1 and pid != -1:
            continue    # Drop all but first soma point
        points.append((n,t,x,y,z,r,pid))

    return points
