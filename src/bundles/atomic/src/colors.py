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

"""
colors: atomic coloring schemes
===============================
"""

# -----------------------------------------------------------------------------
#
element_rgba_256 = None

def element_colors(element_numbers):
    ect = element_color_table()
    colors = ect[element_numbers]
    return colors

def element_color(element_number):
    ect = element_color_table()
    # returning a tuple instead of a numpy array makes comparing colors for
    # equality less stupid
    return tuple(ect[element_number])

def element_color_table():
    global element_rgba_256
    if element_rgba_256 is not None:
        return element_rgba_256

    from numpy import empty, uint8
    element_rgba_256 = ec = empty((256, 4), uint8)
    ec[:, :3] = 180
    ec[:, 3] = 255
    # jmol element colors
    colors = (
        (1, (255, 255, 255)),   # H
        (2, (217, 255, 255)),   # He
        (3, (204, 128, 255)),   # Li
        (4, (194, 255, 0)),     # Be
        (5, (255, 181, 181)),   # B
        (6, (144, 144, 144)),   # C
        (7, (48, 80, 248)),     # N
        (8, (255, 13, 13)),     # O
        (9, (144, 224, 80)),    # F
        (10, (179, 227, 245)),  # Ne
        (11, (171, 92, 242)),   # Na
        (12, (138, 255, 0)),    # Mg
        (13, (191, 166, 166)),  # Al
        (14, (240, 200, 160)),  # Si
        (15, (255, 128, 0)),    # P
        (16, (255, 255, 48)),   # S
        (17, (31, 240, 31)),    # Cl
        (18, (128, 209, 227)),  # Ar
        (19, (143, 64, 212)),   # K
        (20, (61, 255, 0)),     # Ca
        (21, (230, 230, 230)),  # Sc
        (22, (191, 194, 199)),  # Ti
        (23, (166, 166, 171)),  # V
        (24, (138, 153, 199)),  # Cr
        (25, (156, 122, 199)),  # Mn
        (26, (224, 102, 51)),   # Fe
        (27, (240, 144, 160)),  # Co
        (28, (80, 208, 80)),    # Ni
        (29, (200, 128, 51)),   # Cu
        (30, (125, 128, 176)),  # Zn
        (31, (194, 143, 143)),  # Ga
        (32, (102, 143, 143)),  # Ge
        (33, (189, 128, 227)),  # As
        (34, (255, 161, 0)),    # Se
        (35, (166, 41, 41)),    # Br
        (36, (92, 184, 209)),   # Kr
        (37, (112, 46, 176)),   # Rb
        (38, (0, 255, 0)),      # Sr
        (39, (148, 255, 255)),  # Y
        (40, (148, 224, 224)),  # Zr
        (41, (115, 194, 201)),  # Nb
        (42, (84, 181, 181)),   # Mo
        (43, (59, 158, 158)),   # Tc
        (44, (36, 143, 143)),   # Ru
        (45, (10, 125, 140)),   # Rh
        (46, (0, 105, 133)),    # Pd
        (47, (192, 192, 192)),  # Ag
        (48, (255, 217, 143)),  # Cd
        (49, (166, 117, 115)),  # In
        (50, (102, 128, 128)),  # Sn
        (51, (158, 99, 181)),   # Sb
        (52, (212, 122, 0)),    # Te
        (53, (148, 0, 148)),    # I
        (54, (66, 158, 176)),   # Xe
        (55, (87, 23, 143)),    # Cs
        (56, (0, 201, 0)),      # Ba
        (57, (112, 212, 255)),  # La
        (58, (255, 255, 199)),  # Ce
        (59, (217, 255, 199)),  # Pr
        (60, (199, 255, 199)),  # Nd
        (61, (163, 255, 199)),  # Pm
        (62, (143, 255, 199)),  # Sm
        (63, (97, 255, 199)),   # Eu
        (64, (69, 255, 199)),   # Gd
        (65, (48, 255, 199)),   # Tb
        (66, (31, 255, 199)),   # Dy
        (67, (0, 255, 156)),    # Ho
        (68, (0, 230, 117)),    # Er
        (69, (0, 212, 82)),     # Tm
        (70, (0, 191, 56)),     # Yb
        (71, (0, 171, 36)),     # Lu
        (72, (77, 194, 255)),   # Hf
        (73, (77, 166, 255)),   # Ta
        (74, (33, 148, 214)),   # W
        (75, (38, 125, 171)),   # Re
        (76, (38, 102, 150)),   # Os
        (77, (23, 84, 135)),    # Ir
        (78, (208, 208, 224)),  # Pt
        (79, (255, 209, 35)),   # Au
        (80, (184, 184, 208)),  # Hg
        (81, (166, 84, 77)),    # Tl
        (82, (87, 89, 97)),     # Pb
        (83, (158, 79, 181)),   # Bi
        (84, (171, 92, 0)),     # Po
        (85, (117, 79, 69)),    # At
        (86, (66, 130, 150)),   # Rn
        (87, (66, 0, 102)),     # Fr
        (88, (0, 125, 0)),      # Ra
        (89, (112, 171, 250)),  # Ac
        (90, (0, 186, 255)),    # Th
        (91, (0, 161, 255)),    # Pa
        (92, (0, 143, 255)),    # U
        (93, (0, 128, 255)),    # Np
        (94, (0, 107, 255)),    # Pu
        (95, (84, 92, 242)),    # Am
        (96, (120, 92, 227)),   # Cm
        (97, (138, 79, 227)),   # Bk
        (98, (161, 54, 212)),   # Cf
        (99, (179, 31, 212)),   # Es
        (100, (179, 31, 186)),  # Fm
        (101, (179, 13, 166)),  # Md
        (102, (189, 13, 135)),  # No
        (103, (199, 0, 102)),   # Lr
        (104, (204, 0, 89)),    # Rf
        (105, (209, 0, 79)),    # Db
        (106, (217, 0, 69)),    # Sg
        (107, (224, 0, 56)),    # Bh
        (108, (230, 0, 46)),    # Hs
        (109, (235, 0, 38)),    # Mt
    )
    for e, rgb in colors:
        ec[e, :3] = rgb

    return element_rgba_256

# -----------------------------------------------------------------------------
#
chain_rgba_256 = None


def chain_colors(cids, palette=None):
    if palette is not None:
        return _palette_chain_colors(cids, palette)
    global chain_rgba_256
    if chain_rgba_256 is None:
        chain_rgba_256 = {
            'a': (123, 104, 238, 255),
            'b': (240, 128, 128, 255),
            'c': (143, 188, 143, 255),
            'd': (222, 184, 135, 255),
            'e': (255, 127, 80, 255),
            'f': (128, 128, 128, 255),
            'g': (107, 142, 35, 255),
            'h': (100, 100, 100, 255),
            'i': (52, 231, 123, 255),
            'j': (55, 19, 112, 255),
            'k': (255, 255, 150, 255),
            'l': (202, 62, 94, 255),
            'm': (205, 145, 63, 255),
            'n': (183, 118, 231, 255),
            'o': (110, 251, 201, 255),
            'p': (175, 155, 50, 255),
            'q': (105, 205, 48, 255),
            'r': (37, 70, 25, 255),
            's': (121, 33, 135, 255),
            't': (83, 140, 208, 255),
            'u': (0, 154, 37, 255),
            'v': (178, 220, 205, 255),
            'w': (255, 152, 213, 255),
            'x': (200, 90, 174, 255),
            'y': (175, 200, 74, 255),
            'z': (63, 25, 12, 255),
            '1': (87, 87, 87, 255),
            '2': (129, 215, 234, 255),
            '3': (102, 242, 126, 255),
            '4': (29, 105, 20, 255),
            '5': (129, 74, 25, 255),
            '6': (129, 38, 192, 255),
            '7': (160, 160, 160, 255),
            '8': (129, 197, 122, 255),
            '9': (157, 175, 255, 255),
            '0': (41, 208, 208, 255),
        }

    for cid in set(cids):
        c = str(cid).lower()
        if c not in chain_rgba_256:
            from random import randint, seed
            seed(c)
            chain_rgba_256[c] = (randint(128, 255), randint(128, 255), randint(128, 255), 255)

    from numpy import array, uint8, empty
    if len(cids) == 0:
        c = empty((0, 4), uint8)
    else:
        c = array(tuple(chain_rgba_256[cid.lower()] for cid in cids), uint8)
    return c

# -----------------------------------------------------------------------------
#
def _palette_chain_colors(cids, palette):
    c = palette.interpolated_rgba8(_palette_values(cids))
    return c

def _palette_values(cids):
    from numpy import empty, float32
    values = empty([len(cids)], float32)
    vmap = {}
    from random import seed, random
    for i,cid in enumerate(cids):
        if cid in vmap:
            values[i] = vmap[cid]
        else:
            seed(cid)
            vmap[cid] = values[i] = random()
    return values

# -----------------------------------------------------------------------------
#
def chain_rgba(cid):
    return tuple(float(c / 255.0) for c in chain_colors([cid])[0])


# -----------------------------------------------------------------------------
#
def chain_rgba8(cid):
    return chain_colors([cid])[0]


# -----------------------------------------------------------------------------
# Unique color for each unique polymer sequence.
#
def polymer_colors(residues):
    seqs, seq_ids = residues.unique_sequences	# id = 0 for non-chain residues (e.g. solvent).
    nc = len(seqs)
    from random import seed, randint
    colors = []
    for i in range(nc):
        seed(seqs[i])
        colors.append((randint(128,255), randint(128,255), randint(128,255), 255))
    from numpy import array, uint8
    sc = array(colors, uint8)
    c = sc[seq_ids,:]
    mask = (seq_ids > 0)	# mask for residues that are part of a chain
    return c, mask


# -----------------------------------------------------------------------------
# Default nucleotide colors based on the ones in the NDB from 3xdna

#_ndb_colors = {
#    # original NDB colors -- too saturated
#    'A': (255, 0, 0, 255),      # red
#    'T': (0, 0, 255, 255),      # blue
#    'G': (0, 255, 0, 255),      # green
#    'C': (255, 255, 0, 255),    # yellow
#    'I': (0, 100, 0, 255),      # dark green
#    'P': (211, 211, 211, 255),  # light gray
#    'U': (0, 255, 255, 255),    # cyan
#}
_ndb_colors = {
    # desaturated original NDB colors -- whiteness 25%
    'A': (255, 64, 64, 255),    # red
    'T': (64, 64, 255, 255),    # blue
    'G': (64, 255, 64, 255),    # green
    'C': (255, 255, 64, 255),   # yellow
    'I': (0, 100, 0, 255),      # dark green
    'P': (211, 211, 211, 255),  # light gray
    'U': (64, 255, 255, 255),   # cyan
}
#_ndb_colors = {
#    # from ColorBrewer 9-class Paired
#    'A': (227, 26, 28, 255),    # red
#    'T': (31, 120, 180, 255),   # blue
#    'G': (178, 223, 138, 255),  # green
#    'C': (253, 191, 111, 255),  # yellow
#    'I': (51, 160, 44, 255),    # dark green
#    'P': (202, 178, 214, 255),  # violet
#    'U': (166, 206, 227, 255),  # cyan
#}


def nucleotide_colors(residues):
    from numpy import empty, uint8
    from . import Sequence, Residue
    nucleic3to1 = Sequence.nucleic3to1
    mask = residues.polymer_types == Residue.PT_NUCLEIC
    nucleotides = residues.filter(mask)
    colors = empty([len(residues), 4], dtype=uint8)
    cache = _ndb_colors.copy()
    cache['PSU'] = cache['P']
    for ((i, name), is_nuc) in zip(enumerate(residues.names), mask):
        if not is_nuc:
            continue
        color = cache.get(name, None)
        if color is None:
            try:
                n = nucleic3to1(name)
                color = cache[n]
            except KeyError:
                color = (128, 128, 128, 255)
            cache[name] = color
        colors[i] = color
    return colors, mask


# -----------------------------------------------------------------------------
#
atomic_color_names = ["tan", "sky blue", "plum", "light green",
                      "salmon", "light gray", "deep pink", "gold", "dodger blue", "purple"]

def structure_color(id, bg_color):
    from chimerax.core.colors import BuiltinColors, distinguish_from, Color
    try:
        cname = atomic_color_names[id[0]-1]
        model_color = BuiltinColors[cname]
        if (model_color.rgba[:3] == bg_color[:3]).all():
            # force use of another color...
            raise IndexError("Same as background color")
    except IndexError:
        # pick a color that distinguishes from the standard list
        # as well as white and black and green (highlight), and hope...
        avoid = [BuiltinColors[cn].rgba[:3] for cn in atomic_color_names]
        avoid.extend([(0,0,0), (0,1,0), (1,1,1), bg_color[:3]])
        model_color = Color(distinguish_from(avoid, num_candidates=7, seed=id[0]))
    return model_color

def _displayed_colors(atoms):
    unhidden = atoms.filter(atoms.hides == 0)
    displayed_normally = unhidden.filter(unhidden.displays)
    hidden = atoms.filter(atoms.hides != 0)
    displayed_cartoon = hidden.filter(hidden.residues.ribbon_displays)
    from .molarray import concatenate
    return concatenate([displayed_normally, displayed_cartoon]).colors

def predominant_color(atoms, *, none_fraction=0.3):
    '''Returns the single predominant color among the (displayed) atoms (which could be the cartoon color).
    If the predominant color is in less than 'none_fraction' of the displayed atoms, then None will be
    returned.  If no atoms are displayed in any way and 'none_fraction' is 0, then gray is returned.
    '''
    colors = _displayed_colors(atoms)
    if len(colors) == 0:
        if none_fraction > 0:
            return None
        return element_color(6)
    import numpy
    unique, indices, counts = numpy.unique(colors, return_inverse=True, return_counts=True)
    # courtesy of Stack Overflow...
    color = unique[numpy.argmax(numpy.apply_along_axis(numpy.bincount, 0, indices.reshape(colors.shape),
        None, numpy.max(indices)+1), axis=0)]
    if numpy.count_nonzero((colors == color).all(axis=1)) < none_fraction * len(colors):
        return None
    return color

def average_color(atoms):
    '''Returns the average of the displayed atoms' (possibly cartoon) color.  If none of the atoms is
    displayed in any way, return the average color of all the atoms.
    '''
    colors = _displayed_colors(atoms)
    if len(colors) == 0:
        colors = atoms.colors
    return colors.mean(axis=0)

