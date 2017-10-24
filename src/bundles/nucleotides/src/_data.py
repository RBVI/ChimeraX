# vim: set expandtab sw=4:
# --- UCSF Chimera Copyright ---
# Copyright (c) 2004 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.
# --- UCSF Chimera Copyright ---
#
# Base slabs -- approximate purine and pyrimidine bases
#
# Written by Greg Couch, UCSF Computer Graphics Lab, April 2004
# with help from Nikolai Ulyanov.

import math
import re
import weakref
from . import default
import numpy
from chimerax.core.geometry import Place, translation, scale, distance, z_align, Plane, normalize_vector, normalize_vectors
from chimerax.core.surface import box_geometry, sphere_geometry2, cylinder_geometry
from chimerax.core.atomic import Residues, Sequence
nucleic3to1 = Sequence.nucleic3to1

_SQRT2 = math.sqrt(2)
_SQRT3 = math.sqrt(3)

_mol_handler = None
_rebuild_handler = None
_need_rebuild = weakref.WeakSet()
_rebuilding = False

SideOptions = ['orient', 'fill/slab', 'slab', 'tube/slab', 'ladder']

BackboneRE = re.compile("^(C[345]'|O[35]'|P|OP[12])$", re.I)
BackboneSugarRE = re.compile("^(C[12345]'|O[2345]'|P|OP[12])$", re.I)
BaseAtomsRE = re.compile("^(C[245678]|C5M|N[1234679]|O[246])$", re.I)
BaseExceptRE = re.compile("^C1'$", re.I)
SugarAtomsRE = re.compile("^(C[1234]|O[24])'$", re.I)
SugarExceptRE = re.compile("^(C5|N[19]|C5'|O3')$", re.I)
SugarAtomsNoRibRE = re.compile("^(C[12]|O[24])'$", re.I)
SugarExceptNoRibRE = re.compile("^(C5|N[19]|C[34]')$", re.I)


class Always_RE:
    def match(self, text):
        return True


class NeverRE:
    def match(self, text):
        return False


Always_RE = Always_RE()
NeverRE = NeverRE()

# # clockwise rings
# purine = ("N9", "C8", "N7", "C5", "C4")        # + pyrimidine
# pyrimidine = ("N1", "C6", "C5", "C4", "N3", "C2")
# counterclockwise rings
_purine = ("N9", "C4", "C5", "N7", "C8")        # without reversed pyrimidine
_full_purine = ("N9", "C4", "N3", "C2", "N1", "C6", "C5", "N7", "C8")
_full_purine_1 = (0, 1, 6, 7, 8)
_full_purine_2 = (1, 2, 3, 4, 5, 6)
_pyrimidine = ("N1", "C2", "N3", "C4", "C5", "C6")
_pyrimidine_1 = (0, 1, 2, 3, 4, 5)
_sugar = ("C1'", "C2'", "C3'", "C4'", "O4'")

ANCHOR = 'anchor'
BASE = 'base'
SUGAR = 'sugar'
PURINE = 'purine'
PYRIMIDINE = 'pyrimidine'
PSEUDO_PYRIMIDINE = 'pseudo-pyrimidine'

# Standard base geometries with C1'
#
# From "A Standard Reference Frame for the Description of Nucleic Acid
# Base-pair Geometry", Olsen et. al., J. Mol. Biol. (2001) V. 313, pp.
# 229-237.  A preliminary version is available for free at
# <http://ndbserver.rutgers.edu/archives/report/tsukuba/tsukuba.pdf>.
standard_bases = {
    'A': {
        "base": PURINE,
        "ring atom names": _full_purine,
        "NDB color": "red",
        "atoms": {
            "C1'": numpy.array((-2.479, 5.346, 0.000)),
            "N9": numpy.array((-1.291, 4.498, 0.000)),
            "C8": numpy.array((0.024, 4.897, 0.000)),
            "N7": numpy.array((0.877, 3.902, 0.000)),
            "C5": numpy.array((0.071, 2.771, 0.000)),
            "C6": numpy.array((0.369, 1.398, 0.000)),
            "N6": numpy.array((1.611, 0.909, 0.000)),
            "N1": numpy.array((-0.668, 0.532, 0.000)),
            "C2": numpy.array((-1.912, 1.023, 0.000)),
            "N3": numpy.array((-2.320, 2.290, 0.000)),
            "C4": numpy.array((-1.267, 3.124, 0.000)),
        },
        "other bonds": (("C1'", "N9"), ("C4", "C5"), ("C6", "N6"))
    },
    'C': {
        "base": PYRIMIDINE,
        "ring atom names": _pyrimidine,
        "NDB color": "yellow",
        "atoms": {
            "C1'": numpy.array((-2.477, 5.402, 0.000)),
            "N1": numpy.array((-1.285, 4.542, 0.000)),
            "C2": numpy.array((-1.472, 3.158, 0.000)),
            "O2": numpy.array((-2.628, 2.709, 0.001)),
            "N3": numpy.array((-0.391, 2.344, 0.000)),
            "C4": numpy.array((0.837, 2.868, 0.000)),
            "N4": numpy.array((1.875, 2.027, 0.001)),
            "C5": numpy.array((1.056, 4.275, 0.000)),
            "C6": numpy.array((-0.023, 5.068, 0.000)),
        },
        "other bonds": (("C1'", "N1"), ("O2", "C2"), ("C4", "N4"))
    },
    'G': {
        "base": PURINE,
        "ring atom names": _full_purine,
        "NDB color": "green",
        "atoms": {
            "C1'": numpy.array((-2.477, 5.399, 0.000)),
            "N9": numpy.array((-1.289, 4.551, 0.000)),
            "C8": numpy.array((0.023, 4.962, 0.000)),
            "N7": numpy.array((0.870, 3.969, 0.000)),
            "C5": numpy.array((0.071, 2.833, 0.000)),
            "C6": numpy.array((0.424, 1.460, 0.000)),
            "O6": numpy.array((1.554, 0.955, 0.000)),
            "N1": numpy.array((-0.700, 0.641, 0.000)),
            "C2": numpy.array((-1.999, 1.087, 0.000)),
            "N2": numpy.array((-2.949, 0.139, -0.001)),
            "N3": numpy.array((-2.342, 2.364, 0.001)),
            "C4": numpy.array((-1.265, 3.177, 0.000)),
        },
        "other bonds": (("C1'", "N9"), ("C4", "C5"), ("C6", "O6"), ("N2", "C2"))
    },
    'I': {
        # inosine (NOS) -- like G, but without the N2
        # taken from RNAView, ndbserver.rutgers.edu
        "base": PURINE,
        "ring atom names": _full_purine,
        "NDB color": "dark green",
        "atoms": {
            "C1'": numpy.array((-2.477, 5.399, 0.000)),
            "N9": numpy.array((-1.289, 4.551, 0.000)),
            "C8": numpy.array((0.023, 4.962, 0.000)),
            "N7": numpy.array((0.870, 3.969, 0.000)),
            "C5": numpy.array((0.071, 2.833, 0.000)),
            "C6": numpy.array((0.424, 1.460, 0.000)),
            "O6": numpy.array((1.554, 0.955, 0.000)),
            "N1": numpy.array((-0.700, 0.641, 0.000)),
            "C2": numpy.array((-1.999, 1.087, 0.000)),
            "N3": numpy.array((-2.342, 2.364, 0.001)),
            "C4": numpy.array((-1.265, 3.177, 0.000)),
        },
        "other bonds": (("C1'", "N9"), ("C4", "C5"), ("C6", "O6"))
    },
    'P': {
        # pseudouridine (PSU) -- like U with the base ring flipped
        # (C1'->C5 not N1)
        # taken from RNAView, ndbserver.rutgers.edu
        "base": PSEUDO_PYRIMIDINE,
        "ring atom names": _pyrimidine,
        "NDB color": "light gray",
        "atoms": {
            "C1'": numpy.array((-2.506, 5.371, 0.000)),
            "N1": numpy.array((1.087, 4.295, 0.000)),
            "C2": numpy.array((1.037, 2.915, 0.000)),
            "O2": numpy.array((2.036, 2.217, 0.000)),
            "N3": numpy.array((-0.229, 2.383, 0.000)),
            "C4": numpy.array((-1.422, 3.076, 0.000)),
            "O4": numpy.array((-2.485, 2.453, 0.000)),
            "C5": numpy.array((-1.284, 4.500, 0.000)),
            "C6": numpy.array((-0.064, 5.048, 0.000)),
        },
        "other bonds": (("C1'", "C5"), ("O2", "C2"), ("C4", "O4"))
    },
    'T': {
        "base": PYRIMIDINE,
        "ring atom names": _pyrimidine,
        "NDB color": "blue",
        "atoms": {
            "C1'": numpy.array((-2.481, 5.354, 0.000)),
            "N1": numpy.array((-1.284, 4.500, 0.000)),
            "C2": numpy.array((-1.462, 3.135, 0.000)),
            "O2": numpy.array((-2.562, 2.608, 0.000)),
            "N3": numpy.array((-0.298, 2.407, 0.000)),
            "C4": numpy.array((0.994, 2.897, 0.000)),
            "O4": numpy.array((1.944, 2.119, 0.000)),
            "C5": numpy.array((1.106, 4.338, 0.000)),
            "C5M": (2.466, 4.961, 0.001),  # PDB V2
            "C7": (2.466, 4.961, 0.001),  # PDB V3
            "C6": numpy.array((-0.024, 5.057, 0.000)),
        },
        "other bonds": (("C1'", "N1"), ("O2", "C2"), ("C4", "O4"), ("C5", "C5M"), ("C5", "C7"))
    },
    'U': {
        "base": PYRIMIDINE,
        "ring atom names": _pyrimidine,
        "NDB color": "cyan",
        "atoms": {
            "C1'": numpy.array((-2.481, 5.354, 0.000)),
            "N1": numpy.array((-1.284, 4.500, 0.000)),
            "C2": numpy.array((-1.462, 3.131, 0.000)),
            "O2": numpy.array((-2.563, 2.608, 0.000)),
            "N3": numpy.array((-0.302, 2.397, 0.000)),
            "C4": numpy.array((0.989, 2.884, 0.000)),
            "O4": numpy.array((1.935, 2.094, -0.001)),
            "C5": numpy.array((1.089, 4.311, 0.000)),
            "C6": numpy.array((-0.024, 5.053, 0.000)),
        },
        "other bonds": (("C1'", "N1"), ("O2", "C2"), ("C4", "O4"))
    },
}

# map pyrminidine anchors to pseudopryimidine ones
pseudopyrimidine_anchor_map = {
    "C1'": "C1'",
    "N1": "C5",
    "C2": "C4",
    "O2": "C4",
    "N3": "N3",
    "C4": "C2",
    "O4": "C2",
    "C5": "N1",
    "C6": "C6",
}

# precompute bounding boxes
purine_min = purine_max = None
pyrimidine_min = pyrimidine_max = None
for b in standard_bases.values():
    min = max = None
    for coord in b["atoms"].values():
        if min is None:
            min = list(coord)
            max = list(coord)
            continue
        if min[0] > coord[0]:
            min[0] = coord[0]
        elif max[0] < coord[0]:
            max[0] = coord[0]
        if min[1] > coord[1]:
            min[1] = coord[1]
        elif max[1] < coord[1]:
            max[1] = coord[1]
        if min[2] > coord[2]:
            min[2] = coord[2]
        elif max[2] < coord[2]:
            max[2] = coord[2]
    b['min coord'] = min
    b['max coord'] = max
    if b['base'] == PURINE:
        if purine_min is None:
            purine_min = list(min)
            purine_max = list(max)
        else:
            if purine_min[0] > min[0]:
                purine_min[0] = min[0]
            if purine_min[1] > min[1]:
                purine_min[1] = min[1]
            if purine_min[2] > min[2]:
                purine_min[2] = min[2]
            if purine_max[0] < max[0]:
                purine_max[0] = max[0]
            if purine_max[1] < max[1]:
                purine_max[1] = max[1]
            if purine_max[2] < max[2]:
                purine_max[2] = max[2]
    elif b['base'] == PYRIMIDINE:
        if pyrimidine_min is None:
            pyrimidine_min = min[:]
            pyrimidine_max = max[:]
        else:
            if pyrimidine_min[0] > min[0]:
                pyrimidine_min[0] = min[0]
            if pyrimidine_min[1] > min[1]:
                pyrimidine_min[1] = min[1]
            if pyrimidine_min[2] > min[2]:
                pyrimidine_min[2] = min[2]
            if pyrimidine_max[0] < max[0]:
                pyrimidine_max[0] = max[0]
            if pyrimidine_max[1] < max[1]:
                pyrimidine_max[1] = max[1]
            if pyrimidine_max[2] < max[2]:
                pyrimidine_max[2] = max[2]
pu = (purine_max[1] - purine_min[1])
py = (pyrimidine_max[1] - pyrimidine_min[1])
purine_pyrimidine_ratio = pu / (pu + py)
del b, coord, min, max, pu, py

# precompute z-plane rotation correction factor
z_axis = numpy.array((0, 0, 1.))
for b in standard_bases.values():
    pts = [b["atoms"][n] for n in b["ring atom names"][0:2]]
    y_axis = pts[0] - pts[1]
    # insure that y_axis is perpendicular to z_axis
    # (should be zero already)
    y_axis[2] = 0.0
    normalize_vector(y_axis)
    x_axis = numpy.cross(y_axis, z_axis)
    xf = Place(matrix=(
        (x_axis[0], y_axis[0], z_axis[0], 0.0),
        (x_axis[1], y_axis[1], z_axis[1], 0.0),
        (x_axis[2], y_axis[2], z_axis[2], 0.0))
        # TODO: orthogonalize=True
    )
    # axis, angle = xf.getRotation()
    # print("axis = %s, angle = %s" % (axis, angle))
    b["correction factor"] = xf.inverse()
del b, pts, x_axis, y_axis, z_axis, xf

system_styles = {
    # predefined styles in local coordinate frame
    # note: (0, 0) corresponds to position of C1'
    'skinny': {
        ANCHOR: BASE,
        PURINE: ((0.0, -4.0), (2.1, 0.0)),
        PYRIMIDINE: ((0.0, -2.1), (2.1, 0.0)),
        PSEUDO_PYRIMIDINE: ((0.0, -2.1), (2.1, 0.0)),
    },
    'long': {
        ANCHOR: BASE,
        PURINE: ((0.0, -5.0), (2.1, 0.0)),
        PYRIMIDINE: ((0.0, -3.5), (2.1, 0.0)),
        PSEUDO_PYRIMIDINE: ((0.0, -3.5), (2.1, 0.0)),
    },
    'fat': {
        ANCHOR: SUGAR,
        PURINE: ((0.0, -4.87), (3.3, 0.0)),
        PYRIMIDINE: ((0.0, -2.97), (3.3, 0.0)),
        PSEUDO_PYRIMIDINE: ((0.0, -2.97), (3.3, 0.0)),
    },
    'big': {
        ANCHOR: SUGAR,
        PURINE: ((0.0, -5.47), (4.4, 0.0)),
        PYRIMIDINE: ((0.0, -3.97), (4.4, 0.0)),
        PSEUDO_PYRIMIDINE: ((0.0, -3.97), (4.4, 0.0)),
    },
}

_BaseAnchors = {
    PURINE: 'N9',
    PYRIMIDINE: 'N1',
    PSEUDO_PYRIMIDINE: 'C5'
}


def anchor(sugar_or_base, base):
    if sugar_or_base == SUGAR:
        return "C1'"
    return _BaseAnchors[base]


user_styles = {}
pref_styles = {}

pref = None
PREF_CATEGORY = "Nucleotides"
PREF_SLAB_STYLES = "slab styles"
TRIGGER_SLAB_STYLES = "SlabStyleChanged"


def find_style(name):
    try:
        return user_styles[name]
    except KeyError:
        return system_styles.get(name, None)


def add_style(name, info, session=None):
    from chimerax.core.errors import LimitationError
    raise LimitationError("Custom styles are not supported at this time")
    # TODO: rest of this
    """
    exists = name in user_styles
    if exists and user_styles[name] == info:
        return
    user_styles[name] = info
    if name:
        pref_styles[name] = info
        from chimera import preferences
        preferences.save()
        chimera.triggers.activateTrigger(TRIGGER_SLAB_STYLES, name)
    if session is None:
        return
    # if anything is displayed in this style, rebuild it
    for mol in session.models:
        nuc_info = getattr(mol, '_nucleotide_info', None)
        if nuc_info is None:
            continue
        if mol in _need_rebuild:
            continue
        for rd in nuc_info.values():
            slab_params = rd.get('slab params', None)
            if slab_params and slab_params['style'] == name:
                _need_rebuild.add(mol)
                break
    """


def remove_style(name):
    from chimerax.core.errors import LimitationError
    raise LimitationError("Custom styles are not supported at this time")
    # TODO: rest of this
    """
    del user_styles[name]
    del pref_styles[name]
    from chimera import preferences
    preferences.save()
    chimera.triggers.activateTrigger(TRIGGER_SLAB_STYLES, name)
    """


def list_styles(custom_only=False):
    if custom_only:
        return list(user_styles.keys())
    return list(user_styles.keys()) + list(system_styles.keys())


def initialize():
    return
    # TODO: rest of this
    """
    global pref, user_styles, pref_styles
    from chimera import preferences
    pref = preferences.addCategory(PREF_CATEGORY,
                                   preferences.HiddenCategory)
    pref_styles = pref.setdefault(PREF_SLAB_STYLES, {})
    import copy
    user_styles = copy.deepcopy(pref_styles)
    chimera.triggers.addTrigger(TRIGGER_SLAB_STYLES)
    """


def ndb_color(residues):
    # color residues by their NDB color
    from chimerax.core.colors import BuiltinColors
    color_names = set(std['NDB color'] for std in standard_bases.values())
    convert = {}
    for n in color_names:
        convert[n] = BuiltinColors[n].uint8x4()
    other_color = BuiltinColors['tan'].uint8x4()
    colors = []
    for r in residues:
        try:
            info = standard_bases[nucleic3to1(r.name)]
        except KeyError:
            color = other_color
        else:
            color = convert[info['NDB color']]
        colors.append(color)
    for r, c in zip(residues, colors):
        r.atoms.colors = c
        r.ribbon_color = c


def _nuc_drawing(mol, create=True, recreate=False):
    # creates mol._nucleotide_info for per-residue information
    # creates mol._nucleotides_drawing for the drawing
    # from chimerax.bild.shapemodel import ShapeDrawing
    from chimerax.bild.drawing import ShapeDrawing
    global _mol_handler, _rebuild_handler
    try:
        # expect this to succeed most of the time
        info = mol._nucleotide_info
        if recreate:
            mol.remove_drawing(mol._nucleotides_drawing)
            mol._nucleotides_drawing = ShapeDrawing('nucleotides')
            mol.add_drawing(mol._nucleotides_drawing)
        return info, mol._nucleotides_drawing
    except AttributeError:
        if not create:
            return None, None
        nd = mol._nucleotides_drawing = ShapeDrawing('nucleotides')
        mol.add_drawing(nd)
        mol._nucleotide_info = weakref.WeakKeyDictionary()
        # if _mol_handler is None:
        #     _mol_handler = chimera.triggers.add_handler('Model',
        #                                     _trackMolecules, None)
        if _rebuild_handler is None:
            from chimerax.core.atomic import get_triggers
            _rebuild_handler = get_triggers(mol.session).add_handler(
                'changes', _rebuild)
        return mol._nucleotide_info, nd

# def _trackMolecules(trigger_name, closure, changes):
#     """Model trigger handler"""
#     # Track when Molecules and VRML models are deleted to see if
#     # they are ones that we're interested in.
#     if _rebuilding:
#         # don't care about structure changes while we're rebuilding
#         # the VRML models
#         return
#     if not changes:
#         return
#     if 'major' in changes.reasons:
#         for mol in changes.modified:
#             try:
#                 md = _data[mol]
#             except KeyError:
#                 continue
#             _need_rebuild.add(mol)
#     deleted = changes.deleted
#     if not deleted:
#         return
#     # First remove all Molecules.  With weak dictionaries
#     # most of this cleanup should be unnecessary.
#     for mol in list(deleted):
#         try:
#             md = _data[mol]
#         except KeyError:
#             continue
#         try:
#             vrml = md[VRML]
#             # our VRML models are always associated with mol,
#             # so they should always be on the deleted list.
#             deleted.remove(vrml)
#         except KeyError:
#             pass
#         deleted.remove(mol)
#         del _data[mol]
#     # Now look for VRML models whose structure still exists
#     # so we can remove the nucleotides data and unhide any hidden atoms
#     for v in deleted:
#         if not isinstance(v, chimera.VRMLModel):
#             continue
#         # find the parent structure
#         for mol in _data:
#             try:
#                 if _data[mol][VRML] == v:
#                     break
#             except KeyError:
#                 continue
#         else:
#             continue
#         residues = _data[mol].residue_info.keys()
#         del _data[mol]
#         # unhide any atoms we would have hidden
#         set_hide_atoms(False, Always_RE, BackboneRE, residues)
#         chimera.viewer.invalidateCache(mol)
#     if _data:
#         return
#     global _mol_handler, _rebuild_handler
#     chimera.triggers.deleteHandler('Model', _mol_handler)
#     _mol_handler = None
#     chimera.triggers.deleteHandler('monitor changes', _rebuild_handler)
#     _rebuild_handler = None
#     _need_rebuild.clear()


def _rebuild(trigger_name, changes):
    """'monitor changes' trigger handler"""
    global _rebuilding
    # TODO: check changes for things we're interested in
    # ie., add/delete/moving atoms
    if not _need_rebuild or _rebuilding:
        return
    _rebuilding = True
    for mol in _need_rebuild:
        nuc_info, nd = _nuc_drawing(mol, recreate=True)
        if nuc_info is None:
            continue
        # figure out which residues are of which type because
        # ladder needs knowledge about the whole structure
        sides = {}
        for k in SideOptions:
            sides[k] = []
        for r in tuple(nuc_info):
            if r.deleted:
                # Sometimes the residues are gone,
                # but there's a still reference to them.
                del nuc_info[r]
                continue
            sides[nuc_info[r]['side']].append(r)
        if not nuc_info:
            # no residues to track in structure
            mol.remove_drawing(nd)
            del mol._nucleotide_info
            continue
        all_residues = set(nuc_info.keys())
        # create shapes
        hide_sugars = set()
        hide_bases = set()
        residues = sides['ladder']
        if not residues:
            mol._ladder_params = {}
        else:
            residues = Residues(residues=residues)
            # redo all ladder nodes
            hide_sugars.update(residues)
            hide_bases.update(residues)
            # TODO: hide hydrogens between matched bases
            make_ladder(nd, residues, **mol._ladder_params)
            set_hide_atoms(True, Always_RE, BackboneRE, residues)
        residues = sides['fill/slab'] + sides['slab']
        if residues:
            hide_bases.update(make_slab(nd, residues, nuc_info))
        residues = sides['tube/slab']
        if residues:
            hide_sugars.update(residues)
            make_tube(nd, residues, nuc_info)
            hide_bases.update(make_slab(nd, residues, nuc_info))
        residues = sides['orient']
        if residues:
            for r in residues:
                draw_orientation(nd, r)
        # make sure sugar/base atoms are hidden/shown
        show_sugars = all_residues - hide_sugars
        show_bases = all_residues - hide_bases
        showresidue_info = show_sugars - hide_bases
        show_sugars.difference_update(showresidue_info)
        show_bases.difference_update(showresidue_info)
        set_hide_atoms(False, Always_RE, NeverRE, showresidue_info)
        set_hide_atoms(False, BackboneSugarRE, NeverRE, show_sugars)
        non_ribbon_sugars = [r for r in hide_sugars if not r.ribbon_display]
        set_hide_atoms(False, BackboneRE, NeverRE, non_ribbon_sugars)
        set_hide_atoms(False, BaseAtomsRE, BaseExceptRE, show_bases)
    _need_rebuild.clear()
    _rebuilding = False


def set_hide_atoms(hide, AtomsRE, exceptRE, residues):
    # Hide that atoms match AtomsRE and associated hydrogens.  If
    # a hidden atom is bonded to a displayed atom, then bring it back
    # except for the ones in exceptRE.  If a hidden atom is pseudobonded
    # to another atom, then hide the pseudobond.
    from chimerax.core.atomic import Element
    H = Element.get_element(1)
    for r in residues:
        atoms = []
        for a in r.atoms:
            if AtomsRE.match(a.name):
                atoms.append(a)
                continue
            if a.element.number != H:
                continue
            b = a.neighbors
            if not b:
                continue
            if AtomsRE.match(b[0].name):
                atoms.append(a)
        if not atoms:
            continue

        if hide:
            for ra in atoms:
                ra.hide |= ra.HIDE_NUCLEOTIDE
        else:
            for ra in atoms:
                ra.hide &= ~ra.HIDE_NUCLEOTIDE

        # set hide bit for atoms that bond to non-hidden atoms
        for ra in atoms:
            for b in ra.bonds:
                a = b.other_atom(ra)
                if a in atoms:
                    continue
                if exceptRE.match(a.name):
                    continue
                d = b.display
                if not d:
                    continue
                if a.display:
                    # bring back atom
                    ra.hide &= ra.HIDE_NUCLEOTIDE


def get_cylinder(radius, p0, p1, bottom=True, top=True):
    h = distance(p0, p1)
    # TODO: chose number of triangles
    # TODO: separate cap into bottom and top
    vertices, normals, triangles = cylinder_geometry(radius, height=h, caps=bottom or top)
    # rotate so z-axis matches p0->p1
    xf = z_align(p0, p1)
    inverse = xf.inverse()
    vertices = inverse * (vertices + [0, 0, h / 2])
    normals = inverse.apply_without_translation(normals)
    return vertices, normals, triangles


def get_sphere(radius, pt):
    # TODO: chose number of triangles
    vertices, normals, triangles = sphere_geometry2(30)
    vertices = vertices * radius + pt
    return vertices, normals, triangles


def skip_nonstandard_residue(r):
    if r.polymer_type != r.PT_NUCLEIC:
        return True
    # confirm that residue is displayed
    c5p = r.find_atom("C5'")
    return not c5p or not c5p.display


def get_ring(r, base_ring):
    """Return atoms in nucleotide residue named by base_ring.

    All of the atoms must be present and displayed."""
    atoms = []
    for name in base_ring:
        a = r.find_atom(name)
        if a and a.display:
            atoms.append(a)
        else:
            return []
    # confirm they are in a ring
    # Use minimum rings because that will reuse cached ring information
    # from asking for atom radii.
    for a in atoms:
        if len(a.rings()) == 0:
            return []           # not in a ring
    return atoms


def draw_slab(nd, residue, style, thickness, orient, shape, show_gly):
    try:
        t = residue.name
        if t in ('PSU', 'P'):
            n = 'P'
        elif t in ('NOS', 'I'):
            n = 'I'
        else:
            n = nucleic3to1(t)
    except KeyError:
        return False
    standard = standard_bases[n]
    ring_atom_names = standard["ring atom names"]
    atoms = get_ring(residue, ring_atom_names)
    if not atoms:
        return False
    plane = Plane([a.coord for a in atoms])
    info = find_style(style)
    base = standard['base']
    slab_corners = info[base]
    origin = residue.find_atom(anchor(info[ANCHOR], base)).coord
    origin = plane.nearest(origin)

    pts = [plane.nearest(a.coord) for a in atoms[0:2]]
    y_axis = pts[0] - pts[1]
    normalize_vector(y_axis)
    x_axis = numpy.cross(y_axis, plane.normal)
    xf = Place(matrix=(
        (x_axis[0], y_axis[0], plane.normal[0], origin[0]),
        (x_axis[1], y_axis[1], plane.normal[1], origin[1]),
        (x_axis[2], y_axis[2], plane.normal[2], origin[2]))
    )
    xf = xf * standard["correction factor"]

    color = atoms[0].color
    half_thickness = thickness / 2.0

    llx, lly = slab_corners[0]
    llz = -half_thickness
    urx, ury = slab_corners[1]
    urz = half_thickness
    center = (llx + urx) / 2.0, (lly + ury) / 2.0, 0
    if shape == 'box':
        llb = (llx, lly, llz)
        urf = (urx, ury, urz)
        va, na, ta = box_geometry(llb, urf)
        renormalize = False
    elif shape == 'tube':
        radius = (urx - llx) / 2 * _SQRT2
        xf = xf * translation(center)
        xf = xf * scale(1, 1, half_thickness * _SQRT2 / radius)
        height = ury - lly
        va, na, ta = get_cylinder(radius, (0, -height, 0), (0, height, 0))
        renormalize = True
    elif shape == 'ellipsoid':
        # need to reach anchor atom
        xf = xf * translation(center)
        sr = (ury - lly) / 2 * _SQRT3
        xf = xf * scale((urx - llx) / 2 * _SQRT3 / sr, 1,
                        half_thickness * _SQRT3 / sr)
        va, na, ta = get_sphere(sr, (0, 0, 0))
        renormalize = True
    else:
        raise RuntimeError('unknown base shape')
    va = xf * va
    na = xf.apply_without_translation(na)
    if renormalize:
        normalize_vectors(na)
    nd.add_shape(va, na, ta, color, atoms)

    if show_gly:
        c1p = residue.find_atom("C1'")
        ba = residue.find_atom(anchor(info[ANCHOR], base))
        if c1p and ba:
            c1p.hide = False
            ba.hide = False

    if not orient:
        return True

    # show slab orientation by putting "bumps" on surface
    if standard['base'] == PYRIMIDINE:
        center = (llx + urx) / 2.0, (lly + ury) / 2, half_thickness
        va, na, ta = get_sphere(half_thickness, center)
        nd.add_shape(va, na, ta, color, atoms)
    else:
        # purine
        center = (llx + urx) / 2.0, lly + (ury - lly) / 3, half_thickness
        va, na, ta = get_sphere(half_thickness, center)
        nd.add_shape(va, na, ta, color, atoms)
        center = (llx + urx) / 2.0, lly + (ury - lly) * 2 / 3, half_thickness
        va, na, ta = get_sphere(half_thickness, center)
        nd.add_shape(va, na, ta, color, atoms)
    return True


def slab_nodes(nd, residue, style=default.STYLE, thickness=default.THICKNESS,
               hide=default.HIDE, orient=default.ORIENT, shape=default.SHAPE,
               show_gly=default.GLYCOSIDIC):
    return draw_slab(nd, residue, style, thickness, orient, shape, show_gly)


def bonds_between(atoms):
    bonds = []
    for i in range(len(atoms) - 1):
        a = atoms[i]
        otherAtoms = atoms[i + 1:]
        for b in a.bonds:
            oa = b.other_atom(a)
            if oa in otherAtoms:
                bonds.append(b)
    return bonds


def orient_planar_ring(nd, atoms, ring_indices=[]):
    r = atoms[0].residue
    # TODO:
    # if not r.fill_display or r.fill_mode != r.Thick:
    #     # can't show orientation of thin nor aromatic ring
    #     return []
    pts = [a.coord for a in atoms]
    bonds = bonds_between(atoms)
    # if chimera.Bond.Wire in [b.draw_mode for b in bonds]:
    #     radius = 0
    # else:
    if 1:
        radius = min([b.radius for b in bonds])
    if radius == 0:
        # can't show orientation of thin ring
        return []

    color = atoms[0].color
    # non-zero radius
    planeEq = Plane(pts)
    offset = planeEq.normal * radius
    for r in ring_indices:
        center = numpy.average([pts[i] for i in r], axis=0) + offset
        va, na, ta = get_sphere(radius, center)
        nd.add_shape(va, na, ta, color, atoms)


def draw_orientation(nd, residue):
    ring = get_ring(residue, _full_purine)
    if ring:
        indices = [_full_purine_1, _full_purine_2]
        orient_planar_ring(nd, ring, indices)
    ring = get_ring(residue, _pyrimidine)
    if ring:
        indices = [_pyrimidine_1]
        orient_planar_ring(nd, ring, indices)


def sugar_tube(nd, residue, anchor=SUGAR, show_gly=False):
    if anchor == SUGAR:
        show_gly = False
    if anchor == SUGAR or show_gly:
        aname = "C1'"
    else:
        try:
            t = residue.name
            if t in ('PSU', 'P'):
                n = 'P'
            elif t in ('NOS', 'I'):
                n = 'I'
            else:
                n = nucleic3to1(t)
            base = standard_bases[n]['base']
        except KeyError:
            return []
        aname = _BaseAnchors[base]
        if not aname:
            return []
    a = residue.find_atom(aname)
    if not a or not a.display:
        return []
    ep0 = a.coord
    radius = a.structure.bond_radius
    color = a.color

    # calculate position between C3' and C4' on ribbon
    # TODO
    # hasRibbon = residue.ribbon_display and residue.hasRibbon()
    # if hasRibbon:
    #     rrc = residue.ribbonResidueClass
    #     found, o3pPos = rrc.position("O3'")
    #     if not found:
    #         return []
    #     found, c5pPos = rrc.position("C5'")
    #     if not found:
    #         return []
    #     s = chimera.Spline(chimera.Spline.BSpline,
    #                        residue.ribbonCenters())
    #     ep1 = s.coordinate((o3pPos + c5pPos) / 2)
    # else:
    if 1:
        c3p = residue.find_atom("C3'")
        if not c3p:
            return []
        c4p = residue.find_atom("C4'")
        if not c4p:
            return []
        ep1 = (c3p.coord + c4p.coord) / 2

    va, na, ta = get_cylinder(radius, ep0, ep1, bottom=False)
    nd.add_shape(va, na, ta, color, atoms=None)
    va, na, ta = get_sphere(radius, ep0)
    nd.add_shape(va, na, ta, color, atoms=None)

    set_hide_atoms(True, SugarAtomsRE, SugarExceptRE, [residue])


def _c3pos(residue):
    c3p = residue.find_atom("C3'")
    if not c3p:
        return None
    # TODO:
    # if residue.ribbon_display and residue.hasRibbon():
    #     rrc = residue.ribbonResidueClass
    #     found, o3pPos = rrc.position("O3'")
    #     if found:
    #         found, c5pPos = rrc.position("C5'")
    #         if found:
    #             s = chimera.Spline(chimera.Spline.BSpline,
    #                                residue.ribbonCenters())
    #             return c3p, s.coordinate((o3pPos + c5pPos) / 2)
    return c3p, c3p.coord


def set_normal(molecules, residues):
    rds = {}
    for m in molecules:
        nuc_info, nd = _nuc_drawing(m)
        rds[m] = nuc_info
    changed = set()
    for r in residues:
        if rds[r.structure].pop(r, None) is not None:
            changed.add(r)
            _need_rebuild.add(r.structure)
    set_hide_atoms(False, Always_RE, BackboneRE, changed)


def set_orient(molecules, residues):
    rds = {}
    for m in molecules:
        nuc_info, nd = _nuc_drawing(m)
        rds[m] = nuc_info
    for r in residues:
        rd = rds[r.structure].setdefault(r, {})
        cur_side = rd.get('side', None)
        if cur_side == 'orient':
            continue
        _need_rebuild.add(r.structure)
        rd.pop('slab params', None)
        rd.pop('tube params', None)
        rd['side'] = 'orient'


def set_slab(side, molecules, residues, style=default.STYLE, **slab_params):
    if not side.startswith('tube'):
        tube_params = None
    else:
        info = find_style(style)
        tube_params = {
            'show_gly': slab_params.get('show_gly', default.GLYCOSIDIC),
            ANCHOR: info[ANCHOR],
        }
    slab_params['style'] = style
    rds = {}
    for m in molecules:
        nuc_info, nd = _nuc_drawing(m)
        rds[m] = nuc_info
    for r in residues:
        rd = rds[r.structure].setdefault(r, {})
        cur_side = rd.get('side', None)
        if cur_side == side:
            cur_params = rd.get('slab params', None)
            if (cur_params == slab_params and
                    tube_params == rd.get('tube params', None)):
                continue
        _need_rebuild.add(r.structure)
        rd['slab params'] = slab_params
        if not tube_params:
            rd.pop('tube params', None)
        else:
            rd['tube params'] = tube_params
        rd['side'] = side


def make_slab(nd, residues, rds):
    hide_bases = set()
    for r in residues:
        params = rds[r]['slab params']
        if params.get('hide', default.HIDE):
            hide_bases.add(r)
            set_hide_atoms(True, BaseAtomsRE, BaseExceptRE, [r])
        if not slab_nodes(nd, r, **params):
            hide_bases.discard(r)
    return hide_bases


def make_tube(nd, residues, rds):
    # should be called before make_slab
    for r in residues:
        sugar_tube(nd, r, **rds[r]['tube params'])


def set_ladder(molecules, residues, **ladder_params):
    _need_rebuild.update(molecules)
    rds = {}
    for mol in molecules:
        nuc_info, nd = _nuc_drawing(mol)
        rds[mol] = nuc_info
        if hasattr(mol, '_ladder_params'):
            if mol._ladder_params == ladder_params:
                continue
        mol._ladder_params = ladder_params
    for r in residues:
        rd = rds[r.structure].setdefault(r, {})
        cur_side = rd.get('side', None)
        if cur_side == 'ladder':
            continue
        rd.pop('slab params', None)
        rd.pop('tube params', None)
        rd['side'] = 'ladder'


def make_ladder(nd, residues, rung_radius=0, show_stubs=True, skip_nonbase_Hbonds=False):
    """generate links between residues that are hydrogen bonded together"""
    # create list of atoms from residues for donors and acceptors
    mol = residues.unique_structures[0]

    pbg = mol.pseudobond_group(mol.PBG_HYDROGEN_BONDS, create_type=None)
    if not pbg:
        bonds = ()
    else:
        bonds = (p.atoms for p in pbg.pseudobonds)

    # only make one rung between residues even if there is more than one
    # h-bond
    depict_bonds = {}
    for a0, a1 in bonds:
        non_base = (BackboneSugarRE.match(a0.name),
                    BackboneSugarRE.match(a1.name))
        if skip_nonbase_Hbonds and any(non_base):
            continue
        r0 = a0.residue
        r1 = a1.residue
        if r0.connects_to(r1):
            # skip covalently bonded residues
            continue
        if r1 < r0:
            r0, r1 = r1, r0
            non_base = (non_base[1], non_base[0])
        c3p0 = _c3pos(r0)
        if not c3p0:
            continue
        c3p1 = _c3pos(r1)
        if not c3p1:
            continue
        if rung_radius and not any(non_base):
            radius = rung_radius
        elif r0.ribbon_display and r1.ribbon_display:
            style = r0.ribbonFindStyle()
            radius = min(style.width(.5), style.thickness(.5))
        else:
            # TODO: radius = a0.structure.stickScale \
            #     * chimera.Molecule.DefaultBondRadius
            radius = a0.structure.bond_radius
        key = (r0, r1)
        if key in depict_bonds:
            prev_radius = depict_bonds[key][2]
            if prev_radius >= radius:
                continue
        depict_bonds[key] = (c3p0, c3p1, radius, non_base)

    matched_residues = set()
    for (r0, r1), (c3p0, c3p1, radius, non_base) in depict_bonds.items():
        r0color = r0.ribbon_color
        r1color = r1.ribbon_color
        # choose mid-point to make purine larger
        try:
            isPurine0 = standard_bases[nucleic3to1(r0.name)]['base'] == PURINE
            isPurine1 = standard_bases[nucleic3to1(r1.name)]['base'] == PURINE
        except KeyError:
            isPurine0 = False
            isPurine1 = False
        if any(non_base) or isPurine0 == isPurine1:
            mid = 0.5
        elif isPurine0:
            mid = purine_pyrimidine_ratio
        else:
            mid = 1.0 - purine_pyrimidine_ratio
        midpt = c3p0[1] + mid * (c3p1[1] - c3p0[1])
        va, na, ta = get_cylinder(radius, c3p0[1], midpt, top=False)
        nd.add_shape(va, na, ta, r0color, r0.atoms)
        va, na, ta = get_cylinder(radius, c3p1[1], midpt, top=False)
        nd.add_shape(va, na, ta, r1color, r1.atoms)
        if not non_base[0]:
            matched_residues.add(r0)
        if not non_base[1]:
            matched_residues.add(r1)

    if not show_stubs:
        return
    # draw stubs for unmatched nucleotide residues
    for r in residues:
        if r in matched_residues:
            continue
        c3p = _c3pos(r)
        if not c3p:
            continue
        ep0 = c3p[1]
        color = r.ribbon_color
        # find farthest atom from C3'
        dist_atom = (0, None)
        for a in r.atoms:
            dist = ep0.sqdistance(a.coord)
            if dist > dist_atom[0]:
                dist_atom = (dist, a)
        ep1 = dist_atom[1].coord
        va, na, ta = get_cylinder(rung_radius, ep0, ep1)
        nd.add_shape(va, na, ta, color, r.atoms)


# def save_session(trigger, closure, file):
#     """convert data to session data"""
#     if not _data:
#         return
#     # molecular data
#     mdata = {}
#     for m in _data:
#         if m.__destroyed__:
#             continue
#         nd = _data[m]
#         mid = SimpleSession.sessionID(m)
#         smd = mdata[mid] = {}
#         for k in nd:
#             if k.endswith('params'):
#                     smd[k] = nd[k]
#         rds = nd.residue_info
#         srds = smd.residue_info = {}
#         for r in rds:
#             rid = SimpleSession.sessionID(r)
#             rd = rds[r]
#             srd = srds[rid] = {}
#             for k in rd:
#                 if k.endswith('params'):
#                     srd[k] = rd[k]
#             srd['side'] = rd['side']
#     # save restoring code in session
#     restoring_code = (
# """
# def restoreNucleotides():
#     import NucleicAcids as NA
#     NA.restoreState(%s, %s)
# try:
#     restoreNucleotides()
# except:
#     reportRestoreError('Error restoring Nucleotides')
# """)
#     file.write(restoring_code % (
#         SimpleSession.sesRepr(mdata),
#         SimpleSession.sesRepr(user_styles)
#     ))
#
#
# def restoreState(mdata, sdata={}):
#     for name, info in sdata.items():
#         add_style(name, info, session=session)
#     for mid in mdata:
#         m = SimpleSession.idLookup(mid)
#         nd = _nuc_drawing(m)
#         smd = mdata[mid]
#         for k in smd:
#             if k.endswith('params'):
#                 nd[k] = smd[k]
#         rds = nd.residue_info
#         srds = smd.residue_info
#         for rid in srds:
#             r = SimpleSession.idLookup(rid)
#             rd = rds[r] = srds[rid]
#         _need_rebuild.add(m)
#
#
# def _save_scene(trigger, closure, scene):
#     """convert data to scene data"""
#     # basically the same as save_session, except that we always
#     # save something, and we use scene ids instead of session ids.
#     from Animate.Scenes import scenes
#     # molecular data
#     mdata = {}
#     for m in _data:
#         if m.__destroyed__:
#             continue
#         nd = _data[m]
#         mid = scenes.get_id_by_obj(m)
#         smd = mdata[mid] = {}
#         for k in nd:
#             if k.endswith('params'):
#                 smd[k] = nd[k]
#         rds = nd.residue_info
#         srds = smd.residue_info = {}
#         for r in rds:
#             rid = scenes.get_id_by_obj(r)
#             rd = rds[r]
#             srd = srds[rid] = {}
#             for k in rd:
#                 if k.endswith('params'):
#                     srd[k] = rd[k]
#             srd['side'] = rd['side']
#     scene.tool_settings[PREF_CATEGORY] = mdata
#
#
# def _restore_scene(trigger, closure, scene):
#     """convert data to scene data"""
#     # basically the same as restore_session, except that the
#     # absence of data means that it should be cleared
#     try:
#         mdata = scene.tool_settings[PREF_CATEGORY]
#     except KeyError:
#         mdata = {}
#     from Animate.Scenes import scenes
#     mols = set()
#     for mid in mdata:
#         m = scenes.get_obj_by_id(mid)
#         mols.add(m)
#         nd = _nuc_drawing(m)
#         smd = mdata[mid]
#         for k in smd:
#             if k.endswith('params'):
#                 nd[k] = smd[k]
#         rds = nd.residue_info
#         srds = smd.residue_info
#         for rid in srds:
#             r = scenes.get_obj_by_id(rid)
#             rd = rds[r] = srds[rid]
#         _need_rebuild.add(m)
#     for m in list(_data):
#         if m in mols or m.__destroyed__:
#             continue
#         nd = _data[m]
#         residues = _data[m].residue_info.keys()
#         # unhide any atoms we would have hidden
#         set_hide_atoms(False, AlwaysRE, BackboneRE, residues)
#         nd.residue_info.clear()
#         _need_rebuild.add(m)
#
# TODO: chimera.triggers.add_handler(SimpleSession.SAVE_SESSION, save_session, None)
# TODO: chimera.triggers.add_handler(chimera.SCENE_TOOL_SAVE, _save_scene, None)
# TODO: chimera.triggers.add_handler(chimera.SCENE_TOOL_RESTORE, _restore_scene, None)
