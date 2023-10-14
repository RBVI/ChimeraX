# vim: set expandtab ts=4 sw=4:

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

# Base slabs -- approximate purine and pyrimidine bases
#
# Originally written by Greg Couch, UCSF Computer Graphics Lab, April 2004
# with help from Nikolai Ulyanov.

import math
import re
import weakref
import numpy
from chimerax.geometry import Place, translation, scale, distance, distance_squared, z_align, Plane, normalize_vector
from chimerax.surface import box_geometry, sphere_geometry2, cylinder_geometry
from chimerax.core.state import State, StateManager, RestoreError
from chimerax.atomic import Residues, Atoms, Sequence, Pseudobonds, AtomicShapeDrawing, AtomicShapeInfo, Atom
nucleic3to1 = Sequence.nucleic3to1

_SQRT2 = math.sqrt(2)
_SQRT3 = math.sqrt(3)

HIDE_NUCLEOTIDE = Atoms.HIDE_NUCLEOTIDE
FROM_CMD = 'default from command'
STATE_VERSION = 1  # version of session state information

SideOptions = ['orient', 'fill/slab', 'slab', 'tube/slab', 'ladder']

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
_ribose = ("C1'", "C2'", "C3'", "C4'", "O4'")

ANCHOR = 'anchor'
BASE = 'base'
RIBOSE = 'ribose'
PURINE = 'purine'
PYRIMIDINE = 'pyrimidine'
PSEUDO_PYRIMIDINE = 'pseudo-pyrimidine'

# Standard base geometries with C1'
#
# From "A Standard Reference Frame for the Description of Nucleic Acid
# Base-pair Geometry", Olsen et. al., J. Mol. Biol. (2001) V. 313, pp.
# 229-237.  A preliminary version is available for free at
# <http://ndbserver.rutgers.edu/ndbmodule/archives/reports/tsukuba/tsukuba.pdf>.
# DOI: 10.1006/jmbi.2001.4987
_purine_C2_index = _full_purine.index("C2")
_pyrimidine_C2_index = _pyrimidine.index("C2")
standard_bases = {
    'A': {
        "tag": PURINE,
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
        "tag": PYRIMIDINE,
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
        "tag": PURINE,
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
        "tag": PURINE,
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
        "tag": PSEUDO_PYRIMIDINE,
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
        "tag": PYRIMIDINE,
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
        "tag": PYRIMIDINE,
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
    if b['tag'] == PURINE:
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
    elif b['tag'] == PYRIMIDINE:
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
    y_axis = normalize_vector(y_axis)
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

system_dimensions = {
    # predefined dimensions in local coordinate frame
    # note: (0, 0) corresponds to position of C1'
    'small': {
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
    'big': {
        ANCHOR: RIBOSE,
        PURINE: ((0.0, -4.87), (3.3, 0.0)),
        PYRIMIDINE: ((0.0, -2.97), (3.3, 0.0)),
        PSEUDO_PYRIMIDINE: ((0.0, -2.97), (3.3, 0.0)),
    },
    'fat': {
        ANCHOR: RIBOSE,
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


def anchor(ribose_or_base, tag):
    if ribose_or_base == RIBOSE:
        return "C1'"
    return _BaseAnchors[tag]


user_dimensions = {}
pref_dimensions = {}

pref = None
PREF_CATEGORY = "Nucleotides"
PREF_SLAB_DIMENSIONS = "slab dimensions"
TRIGGER_SLAB_DIMENSIONS = "SlabDimensionsChanged"


def find_dimensions(name):
    try:
        return user_dimensions[name]
    except KeyError:
        return system_dimensions.get(name, None)


def add_dimensions(name, info, session=None):
    from chimerax.core.errors import LimitationError
    raise LimitationError("Custom dimensions are not supported at this time")
    # TODO: rest of this
    """
    exists = name in user_dimensions
    if exists and user_dimensions[name] == info:
        return
    user_dimensions[name] = info
    if name:
        pref_dimensions[name] = info
        from chimera import preferences
        preferences.save()
        chimera.triggers.activateTrigger(TRIGGER_SLAB_DIMENSIONS, name)
    if session is None:
        return
    # if anything is displayed in this dimensions, rebuild it
    for mol in session.models:
        nuc_info = getattr(mol, '_nucleotide_info', None)
        if nuc_info is None:
            continue
        if mol in _need_rebuild:
            continue
        for rd in nuc_info.values():
            slab_params = rd.get('slab params', None)
            if slab_params and slab_params['dimensions'] == name:
                _need_rebuild.add(mol)
                break
    """


def remove_dimensions(name):
    from chimerax.core.errors import LimitationError
    raise LimitationError("Custom dimensions are not supported at this time")
    # TODO: rest of this
    """
    del user_dimensions[name]
    del pref_dimensions[name]
    from chimera import preferences
    preferences.save()
    chimera.triggers.activateTrigger(TRIGGER_SLAB_DIMENSIONS, name)
    """


def list_dimensions(custom_only=False):
    if custom_only:
        return list(user_dimensions.keys())
    return list(user_dimensions.keys()) + list(system_dimensions.keys())


def initialize():
    return
    # TODO: rest of this
    """
    global pref, user_dimensions, pref_dimensions
    from chimera import preferences
    pref = preferences.addCategory(PREF_CATEGORY,
                                   preferences.HiddenCategory)
    pref_dimensions = pref.setdefault(PREF_SLAB_DIMENSIONS, {})
    import copy
    user_dimensions = copy.deepcopy(pref_dimensions)
    chimera.triggers.addTrigger(TRIGGER_SLAB_DIMENSIONS)
    """


class Params(State):

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __eq__(self, other):
        return vars(self) == vars(other)

    def update(self, other):
        for k, v in vars(other).items():
            setattr(self, k, v)


class NucleotideState(StateManager):

    def __init__(self, session):
        self._session = weakref.ref(session)
        self.structures = weakref.WeakSet()
        self.need_rebuild = weakref.WeakSet()
        self.rebuild_handler = session.triggers.add_handler('new frame', self.rebuild)

    def take_snapshot(self, session, flags):
        save_scene = (flags & self.SCENE) != 0
        if not self.structures and not save_scene:
            # no structures with nucleotides, so don't save in session
            return None
        infos = {}
        for mol in self.structures:
            # convert _nucleotide_info from WeakKeyDictionary to dict
            info = {}
            if mol.was_deleted:
                # insurance, in case 'new frame' trigger doesn't happen first
                continue
            info.update(mol._nucleotide_info)
            infos[mol] = (info, mol._ladder_params)
        if save_scene:
            from chimerax.atomic import AtomicStructure
            for model in session.models:
                if model in infos or not isinstance(model, AtomicStructure):
                    continue
                infos[model] = (None, None)
        data = {
            'version': STATE_VERSION,
            'infos': infos
        }
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        if data['version'] != 1:
            raise RestoreError("unknown nucleotide session state version")
        nuc = _nucleotides(session)
        infos = data['infos']
        for mol, (info, params) in infos.items():
            if info is None:
                if not hasattr(mol, '_nucleotide_info'):
                    continue
                residues = Residues(mol._nucleotide_info.keys())
                residues.atoms.clear_hide_bits(HIDE_NUCLEOTIDE)
                _remove_nuc_drawing(nuc, mol)
                continue
            if not hasattr(mol, '_nucleotide_info'):
                prev_residues = None
            else:
                prev_residues = Residues(mol._nucleotide_info.keys())
            nuc.structures.add(mol)
            nuc.need_rebuild.add(mol)
            _make_nuc_drawing(nuc, mol)
            if prev_residues is not None:
                mol._nucleotide_info.clear()
            mol._nucleotide_info.update(info)
            if prev_residues is not None:
                new_residues = Residues(info.keys())
                removed_residues = prev_residues - new_residues
                removed_residues.atoms.clear_hide_bits(HIDE_NUCLEOTIDE)
            mol._ladder_params.update(params)
        return nuc

    def reset_state(self, session):
        pass

    def rebuild(self, trigger_name, update_loop):
        """'monitor changes' trigger handler"""
        deleted = set(s for s in self.structures if s.was_deleted)
        self.structures -= deleted
        if not self.structures:
            session = self._session()
            try:
                del session.nucleotides
            except AttributeError:
                pass
            h = self.rebuild_handler
            self.rebuild_handler = None
            session.triggers.remove_handler(h)
            return
        if not self.need_rebuild:
            return
        for mol in list(self.need_rebuild):
            if not mol.deleted:
                _rebuild_molecule('internal', mol)
        # assert len(_need_rebuild) == 0
        self.need_rebuild.clear()


def _nucleotides(session):
    if not hasattr(session, 'nucleotides'):
        session.nucleotides = NucleotideState(session)
    return session.nucleotides


def hide_hydrogen_bonds(residues, bases_only=False):
    # hide hydrogen bonds to non-ribbon backbone atoms of nucleotide residues
    mol = residues[0].structure
    pbg = mol.pseudobond_group(mol.PBG_HYDROGEN_BONDS, create_type=None)
    if not pbg:
        return

    BBE_RIBBON = Atoms.BBE_RIBBON
    residue_set = set(residues)     # make a set for quick inclusion test
    hbonds = []
    for hb in pbg.pseudobonds:
        a0, a1 = hb.atoms
        r0 = a0.residue
        if r0 in residue_set and not a0.is_backbone(BBE_RIBBON):
            hbonds.append(hb)
            continue
        r1 = a1.residue
        if r1 in residues and not a1.is_backbone(BBE_RIBBON):
            hbonds.append(hb)
    Pseudobonds(hbonds).shown_when_atoms_hiddens = False


def _make_nuc_drawing(nuc, mol, create=True, recreate=False):
    # Side effects:
    #   creates mol._nucleotide_info for per-residue information
    #   creates mol._ladder_params for ladder parameters
    #   creates mol._nucleotides_drawing for the drawing
    try:
        # expect this to succeed most of the time
        if recreate:
            mol.remove_drawing(mol._nucleotides_drawing)
            mol._nucleotides_drawing = AtomicShapeDrawing('nucleotides')
            mol.add_drawing(mol._nucleotides_drawing)
        return mol._nucleotides_drawing
    except AttributeError:
        if not create:
            return None
        nuc.structures.add(mol)
        nd = mol._nucleotides_drawing = AtomicShapeDrawing('nucleotides')
        mol.add_drawing(nd)
        mol._nucleotide_info = weakref.WeakKeyDictionary()
        mol._ladder_params = Params()
        handler = getattr(mol, '_nucleotide_changes', None)
        if handler is None:
            handler = mol.triggers.add_handler('changes', _rebuild_molecule)
            mol._nucleotide_changes = handler
        return nd


def _remove_nuc_drawing(nuc, mol):
    nuc.need_rebuild.discard(mol)
    nuc.structures.discard(mol)
    nd = mol._nucleotides_drawing
    del mol._nucleotides_drawing
    mol.remove_drawing(nd)
    del mol._nucleotide_info
    del mol._ladder_params
    h = mol._nucleotide_changes
    del mol._nucleotide_changes
    mol.triggers.remove_handler(h)


_AtomReasons = frozenset(['coord changed', 'display changed'])
_ResidueReasons = frozenset(['ring_color changed', 'ribbon_display changed'])


def _rebuild_molecule(trigger_name, mol):
    if trigger_name == 'changes':
        mol, changes = mol
        # check changes for reasons we're interested in
        # ie., add/delete/moving atoms
        if changes.num_deleted_atoms():
            pass  # rebuild
        elif not set(changes.residue_reasons()).isdisjoint(_ResidueReasons):
            pass  # rebuild
        elif 'active_coordset changed' in changes.structure_reasons():
            pass  # rebuild
        else:
            reasons = set(changes.atom_reasons())
            if reasons.isdisjoint(_AtomReasons):
                # no reason to rebuild
                return
    mol.update_graphics_if_needed()  # need to recompute ribbon first
    nuc = _nucleotides(mol.session)
    nd = _make_nuc_drawing(nuc, mol, recreate=True)
    if nd is None:
        nuc.need_rebuild.discard(mol)
        return
    nuc_info = mol._nucleotide_info
    all_shapes = []
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
        _remove_nuc_drawing(nuc, mol)
        return
    all_residues = Residues(nuc_info.keys())
    # create shapes
    hide_riboses = []
    hide_bases = []
    show_glys = []
    residues = sides['ladder']
    if residues:
        residues = Residues(residues)
        # redo all ladder nodes
        # TODO: hide hydrogen bonds between matched bases
        shapes, hide_residues = make_ladder(nd, residues, mol._ladder_params)
        all_shapes.extend(shapes)
        hide_riboses.extend(hide_residues)
        hide_bases.extend(hide_residues)
    residues = sides['fill/slab'] + sides['slab']
    if residues:
        shapes, hide_residues = make_slab(nd, residues, nuc_info)
        all_shapes.extend(shapes)
        hide_bases.extend(hide_residues)
        show_glys.extend(hide_residues)
    residues = sides['tube/slab']
    if residues:
        shapes, hide_residues = make_slab(nd, residues, nuc_info)
        all_shapes.extend(shapes)
        hide_bases.extend(hide_residues)
        shapes, hide_residues, need_glys = make_tube(nd, hide_residues, nuc_info)
        all_shapes.extend(shapes)
        hide_riboses.extend(hide_residues)
        show_glys.extend(need_glys)
    residues = sides['orient']
    if residues:
        for r in residues:
            shapes = draw_orientation(nd, r)
            all_shapes.extend(shapes)
    hide_riboses = Residues(hide_riboses)
    hide_bases = Residues(hide_bases)
    if all_shapes:
        nd.add_shapes(all_shapes)

    if hide_bases:
        # Until we have equivalent of ribbon_coord for atoms
        # hidden by nucleotide representations, we hide the
        # hydrogen bonds to atoms hidden by nucleotides.
        hide_hydrogen_bonds(hide_bases)

        # TODO: hide other pseudobonds for the same reason.

    # make sure ribose/base atoms are hidden/shown
    affected_residues = hide_riboses | hide_bases
    hide_all = hide_riboses & hide_bases
    hide_riboses = hide_riboses - hide_all
    hide_bases = hide_bases - hide_all

    atoms = affected_residues.atoms
    backbone_atoms = atoms.filter(atoms.is_backbones(Atom.BBE_RIBBON))
    ribose_atoms = atoms.filter(atoms.is_riboses)

    all_residues.clear_hide_bits(HIDE_NUCLEOTIDE, True)
    # hide all non-backbone atoms
    (hide_all.atoms - backbone_atoms).set_hide_bits(HIDE_NUCLEOTIDE)
    # hide all bases (non-ribose) atoms
    (hide_bases.atoms - ribose_atoms - backbone_atoms).set_hide_bits(HIDE_NUCLEOTIDE)
    # hide all ribose atoms
    (hide_riboses.atoms - backbone_atoms).set_hide_bits(HIDE_NUCLEOTIDE)

    for residue in show_glys:
        rd = nuc_info[residue]
        tag = standard_bases[rd['name']]['tag']
        ba = residue.find_atom(anchor(BASE, tag))
        c1p = residue.find_atom("C1'")
        if c1p and ba:
            c1p.clear_hide_bits(HIDE_NUCLEOTIDE)
            ba.clear_hide_bits(HIDE_NUCLEOTIDE)

    nuc.need_rebuild.discard(mol)


def get_cylinder(radius, p0, p1, bottom=True, top=True):
    h = distance(p0, p1)
    # TODO: chose number of triangles
    # TODO: separate cap into bottom and top
    vertices, normals, triangles = cylinder_geometry(radius, height=h, caps=bottom or top, nc=30)
    # rotate so z-axis matches p0->p1
    xf = z_align(p0, p1)
    inverse = xf.inverse()
    vertices = inverse * (vertices + [0, 0, h / 2])
    inverse.transform_normals(normals, in_place=True, is_rotation=True)
    return vertices, normals, triangles


def get_sphere(radius, pt):
    # TODO: chose number of triangles
    vertices, normals, triangles = sphere_geometry2(300)
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


def draw_slab(nd, residue, name, params):
    standard = standard_bases[name]
    ring_atom_names = standard["ring atom names"]
    atoms = get_ring(residue, ring_atom_names)
    if not atoms:
        return []
    plane = Plane([a.coord for a in atoms])
    info = find_dimensions(params.dimensions)
    tag = standard['tag']
    slab_corners = info[tag]
    origin = residue.find_atom(anchor(info[ANCHOR], tag)).coord
    origin = plane.nearest(origin)

    pts = [plane.nearest(a.coord) for a in atoms[0:2]]
    y_axis = pts[0] - pts[1]
    y_axis = normalize_vector(y_axis)
    x_axis = numpy.cross(y_axis, plane.normal)
    xf = Place(matrix=(
        (x_axis[0], y_axis[0], plane.normal[0], origin[0]),
        (x_axis[1], y_axis[1], plane.normal[1], origin[1]),
        (x_axis[2], y_axis[2], plane.normal[2], origin[2]))
    )
    det = xf.determinant()
    if not numpy.isfinite(det) or abs(det - 1) > 1e-4:
        return []
    xf = xf * standard["correction factor"]

    color = residue.ring_color
    half_thickness = params.thickness / 2

    llx, lly = slab_corners[0]
    llz = -half_thickness
    urx, ury = slab_corners[1]
    urz = half_thickness
    center = (llx + urx) / 2, (lly + ury) / 2, 0
    if params.shape == 'box':
        llb = (llx, lly, llz)
        urf = (urx, ury, urz)
        xf2 = xf
        va, na, ta = box_geometry(llb, urf)
        pure_rotation = True
    elif params.shape == 'muffler':
        radius = (urx - llx) / 2 * _SQRT2
        xf2 = xf * translation(center)
        xf2 = xf2 * scale((1, 1, half_thickness * _SQRT2 / radius))
        height = ury - lly
        va, na, ta = get_cylinder(radius, numpy.array((0, -height / 2, 0)),
                                  numpy.array((0, height / 2, 0)))
        pure_rotation = False
    elif params.shape == 'ellipsoid':
        # need to reach anchor atom
        xf2 = xf * translation(center)
        sr = (ury - lly) / 2 * _SQRT3
        xf2 = xf2 * scale(((urx - llx) / 2 * _SQRT3 / sr, 1,
                           half_thickness * _SQRT3 / sr))
        va, na, ta = get_sphere(sr, (0, 0, 0))
        pure_rotation = False
    else:
        raise RuntimeError('unknown base shape')

    description = '%s %s' % (residue, tag)
    xf2.transform_points(va, in_place=True)
    xf2.transform_normals(na, in_place=True, is_rotation=pure_rotation)
    shapes = [AtomicShapeInfo(va, na, ta, color, atoms, description)]

    if not params.orient:
        return shapes

    # show slab orientation by putting "bumps" on surface
    if tag == PYRIMIDINE:
        center = (llx + urx) / 2.0, (lly + ury) / 2, half_thickness
        va, na, ta = get_sphere(half_thickness, center)
        xf.transform_points(va, in_place=True)
        xf.transform_normals(na, in_place=True, is_rotation=True)
        shapes.append(AtomicShapeInfo(va, na, ta, color, atoms, description))
    else:
        # purine
        center = (llx + urx) / 2.0, lly + (ury - lly) / 3, half_thickness
        va, na, ta = get_sphere(half_thickness, center)
        xf.transform_points(va, in_place=True)
        xf.transform_normals(na, in_place=True, is_rotation=True)
        shapes.append(AtomicShapeInfo(va, na, ta, color, atoms, description))
        center = (llx + urx) / 2.0, lly + (ury - lly) * 2 / 3, half_thickness
        va, na, ta = get_sphere(half_thickness, center)
        xf.transform_points(va, in_place=True)
        xf.transform_normals(na, in_place=True, is_rotation=True)
        shapes.append(AtomicShapeInfo(va, na, ta, color, atoms, description))
    return shapes


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


def orient_planar_ring(nd, atoms, ring_indices):
    shapes = []
    r = atoms[0].residue
    # TODO:
    # if not r.fill_display or r.fill_mode != r.Thick:
    #     # can't show orientation of thin nor aromatic ring
    #     return shapes
    pts = [a.coord for a in atoms]
    bonds = bonds_between(atoms)
    # if chimera.Bond.Wire in [b.draw_mode for b in bonds]:
    #     radius = 0
    # else:
    if 1:
        radius = min([b.radius for b in bonds])
    if radius == 0:
        # can't show orientation of thin ring
        return shapes

    color = r.ring_color
    # non-zero radius
    planeEq = Plane(pts)
    offset = planeEq.normal * radius
    for r in ring_indices:
        center = numpy.average([pts[i] for i in r], axis=0) + offset
        va, na, ta = get_sphere(radius, center)
        shapes.append(AtomicShapeInfo(va, na, ta, color, str(atoms)))
    return shapes


def draw_orientation(nd, residue):
    shapes = []
    ring = get_ring(residue, _full_purine)
    if ring:
        indices = [_full_purine_1, _full_purine_2]
        shapes.extend(orient_planar_ring(nd, ring, indices))
    ring = get_ring(residue, _pyrimidine)
    if ring:
        indices = [_pyrimidine_1]
        shapes.extend(orient_planar_ring(nd, ring, indices))
    return shapes


def draw_tube(nd, residue, name, params):
    shapes = []
    if params.anchor == RIBOSE:
        show_gly = False
    else:
        show_gly = params.show_gly
    if params.anchor == RIBOSE or show_gly:
        aname = "C1'"
    else:
        tag = standard_bases[name]['tag']
        aname = _BaseAnchors[tag]
        if not aname:
            return False
    a = residue.find_atom(aname)
    if not a or not a.display:
        return shapes
    ep0 = a.coord
    if params.radius is None:
        radius = a.structure.bond_radius
    else:
        radius = params.radius

    color = residue.ring_color

    # calculate position between C3' and C4' on ribbon
    c3p = residue.find_atom("C3'")
    if not c3p:
        return shapes
    c4p = residue.find_atom("C4'")
    if not c4p:
        return shapes
    c3p_coord = c3p.effective_coord
    c4p_coord = c4p.effective_coord
    ep1 = (c3p_coord + c4p_coord) / 2

    description = '%s ribose' % residue

    atoms = residue.atoms
    atoms = atoms.filter(atoms.is_riboses)

    va, na, ta = get_cylinder(radius, ep0, ep1, bottom=False)
    shapes.append(AtomicShapeInfo(va, na, ta, color, atoms, description))
    va, na, ta = get_sphere(radius, ep0)
    shapes.append(AtomicShapeInfo(va, na, ta, color, atoms, description))
    return shapes


def _c3pos(residue):
    c3p = residue.find_atom("C3'")
    if not c3p or not c3p.display:
        return None
    return c3p, c3p.effective_coord


def set_normal(residues):
    molecules = residues.unique_structures
    nuc = _nucleotides(molecules[0].session)
    rds = {}
    for mol in molecules:
        _make_nuc_drawing(nuc, mol)
        rds[mol] = mol._nucleotide_info
    changed = {}
    for r in residues:
        if rds[r.structure].pop(r, None) is not None:
            changed.setdefault(r.structure, []).append(r)
    nuc.need_rebuild.update(changed.keys())
    import itertools
    Residues(itertools.chain(*changed.values())).atoms.clear_hide_bits(HIDE_NUCLEOTIDE)


def set_orient(residues):
    molecules = residues.unique_structures
    nuc = _nucleotides(molecules[0].session)
    rds = {}
    for mol in molecules:
        _make_nuc_drawing(nuc, mol)
        rds[mol] = mol._nucleotide_info
    for r in residues:
        rd = rds[r.structure].setdefault(r, {})
        cur_side = rd.get('side', None)
        if cur_side == 'orient':
            continue
        nuc.need_rebuild.add(r.structure)
        rd.pop('slab params', None)
        rd.pop('tube params', None)
        rd['side'] = 'orient'


def set_slab(side, residues, *, dimensions=FROM_CMD,
             thickness=FROM_CMD, hide=FROM_CMD,
             orient=FROM_CMD, shape=FROM_CMD,
             tube_radius=FROM_CMD, show_gly=FROM_CMD):
    molecules = residues.unique_structures
    nuc = _nucleotides(molecules[0].session)
    if not side.startswith('tube'):
        tube_params = None
    else:
        info = find_dimensions(dimensions)
        tube_params = Params(radius=tube_radius, show_gly=show_gly,
                             anchor=info[ANCHOR])
    slab_params = Params(dimensions=dimensions, thickness=thickness,
                         hide=hide, orient=orient, shape=shape)
    rds = {}
    for mol in molecules:
        _make_nuc_drawing(nuc, mol)
        rds[mol] = mol._nucleotide_info
    for r in residues:
        t = r.name
        if t in ('PSU', 'P'):
            n = 'P'
        elif t in ('NOS', 'I'):
            n = 'I'
        else:
            n = nucleic3to1(t)
            if n not in standard_bases:
                continue

        rd = rds[r.structure].setdefault(r, {})
        rd['name'] = n

        cur_side = rd.get('side', None)
        if cur_side == side:
            cur_params = rd.get('slab params', None)
            if (cur_params == slab_params and
                    tube_params == rd.get('tube params', None)):
                continue
        nuc.need_rebuild.add(r.structure)
        rd['slab params'] = slab_params
        if not tube_params:
            rd.pop('tube params', None)
        else:
            rd['tube params'] = tube_params
        rd['side'] = side


def make_slab(nd, residues, rds):
    # returns collection of residues whose bases are drawn as slabs and
    # and have their atoms hidden
    all_shapes = []
    hidden = []
    for r in residues:
        params = rds[r]['slab params']
        hide_base = params.hide
        shapes = draw_slab(nd, r, rds[r]['name'], params)
        if shapes:
            all_shapes.extend(shapes)
        else:
            hide_base = False
        if hide_base:
            hidden.append(r)
    return all_shapes, hidden


def make_tube(nd, residues, rds):
    # should be called before make_slab
    all_shapes = []
    hidden_ribose = []
    shown_gly = []
    for r in residues:
        hide_ribose = True
        rd = rds[r]
        params = rd['tube params']
        show_gly = params.show_gly
        shapes = draw_tube(nd, r, rd['name'], params)
        if shapes:
            all_shapes.extend(shapes)
        else:
            hide_ribose = False
            show_gly = False
        if hide_ribose:
            hidden_ribose.append(r)
        if show_gly:
            shown_gly.append(r)
    return all_shapes, hidden_ribose, shown_gly


def set_ladder(residues, *, rung_radius=FROM_CMD, show_stubs=FROM_CMD, skip_nonbase_Hbonds=FROM_CMD, hide=FROM_CMD, stubs_only=FROM_CMD):
    molecules = residues.unique_structures
    nuc = _nucleotides(molecules[0].session)
    nuc.need_rebuild.update(molecules)
    ladder_params = Params(
        rung_radius=rung_radius, show_stubs=show_stubs,
        skip_nonbase_Hbonds=skip_nonbase_Hbonds, hide=hide,
        stubs_only=stubs_only,
    )
    rds = {}
    for mol in molecules:
        _make_nuc_drawing(nuc, mol)
        rds[mol] = mol._nucleotide_info
        mol._ladder_params = ladder_params
    for r in residues:
        rd = rds[r.structure].setdefault(r, {})
        cur_side = rd.get('side', None)
        if cur_side == 'ladder':
            continue
        rd.pop('slab params', None)
        rd.pop('tube params', None)
        rd['side'] = 'ladder'


def make_ladder(nd, residues, params):
    """generate links between residues that are hydrogen bonded together"""
    # returns set of residues whose bases are drawn as rungs and
    # and have their atoms hidden
    all_shapes = []

    # Create list of atoms from residues for donors and acceptors
    mol = residues[0].structure

    # make a set for quick inclusion test
    residue_set = set(residues)

    pbg = mol.pseudobond_group(mol.PBG_HYDROGEN_BONDS, create_type=None)
    if not pbg:
        bonds = ()
    else:
        bonds = (p.atoms for p in pbg.pseudobonds)

    # only make one rung between residues even if there is more than one
    # h-bond
    depict_bonds = {}
    for a0, a1 in bonds:
        r0 = a0.residue
        r1 = a1.residue
        if r0 not in residue_set or r1 not in residue_set:
            continue
        non_base = (a0.is_ribose, a1.is_ribose)
        if params.skip_nonbase_Hbonds and any(non_base):
            continue
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
        if params.rung_radius and not any(non_base):
            radius = params.rung_radius
        # elif r0.ribbon_display and r1.ribbon_display:
        #     mgr = mol.ribbon_xs_mgr
        #     radius = min(mgr.scale_nucleic)
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
    if not params.stubs_only:
        for (r0, r1), (c3p0, c3p1, radius, non_base) in depict_bonds.items():
            a0 = r0.find_atom("C2")
            a1 = r1.find_atom("C2")
            r0color = r0.ring_color
            r1color = r1.ring_color
            # choose mid-point to make purine larger
            try:
                is_purine0 = standard_bases[nucleic3to1(r0.name)]['tag'] == PURINE
                is_purine1 = standard_bases[nucleic3to1(r1.name)]['tag'] == PURINE
            except KeyError:
                is_purine0 = False
                is_purine1 = False
            if any(non_base) or is_purine0 == is_purine1:
                mid = 0.5
            elif is_purine0:
                mid = purine_pyrimidine_ratio
            else:
                mid = 1.0 - purine_pyrimidine_ratio
            midpt = c3p0[1] + mid * (c3p1[1] - c3p0[1])
            va, na, ta = get_cylinder(radius, c3p0[1], midpt, top=False)
            all_shapes.append(AtomicShapeInfo(va, na, ta, r0color, r0.atoms, str(r0)))
            va, na, ta = get_cylinder(radius, c3p1[1], midpt, top=False)
            all_shapes.append(AtomicShapeInfo(va, na, ta, r1color, r1.atoms, str(r1)))
            if not non_base[0]:
                matched_residues.add(r0)
            if not non_base[1]:
                matched_residues.add(r1)

    if not params.show_stubs:
        if params.hide:
            return all_shapes, matched_residues
        return all_shapes, ()
    # draw stubs for unmatched nucleotide residues
    for r in residues:
        if r in matched_residues:
            continue
        c3p = _c3pos(r)
        if not c3p:
            continue
        ep0 = c3p[1]
        a = r.find_atom("C2")
        color = r.ring_color
        ep1 = None
        name = nucleic3to1(r.name)
        if name in standard_bases:
            is_purine = standard_bases[name]['tag'] == PURINE
            if is_purine:
                a = r.find_atom('N1')
                if a:
                    ep1 = a.coord
            else:
                # pyrimidine
                a = r.find_atom('N3')
                if a:
                    ep1 = a.coord
        if ep1 is None:
            # find farthest heavy atom from C3'
            from chimerax.atomic import Element
            H = Element.get_element(1)
            dist_atom = (0, None)
            for a in r.atoms:
                if a.element == H:
                    continue
                dist = distance_squared(ep0, a.coord)
                if dist > dist_atom[0]:
                    dist_atom = (dist, a)
            ep1 = dist_atom[1].coord
        if ep1 is None:
            continue
        va, na, ta = get_cylinder(params.rung_radius, ep0, ep1)
        all_shapes.append(AtomicShapeInfo(va, na, ta, color, r.atoms, str(r)))
        # make exposed end rounded (TODO: use a hemisphere)
        va, na, ta = get_sphere(params.rung_radius, ep1)
        all_shapes.append(AtomicShapeInfo(va, na, ta, color, r.atoms, str(r)))
        matched_residues.add(r)
    if params.hide:
        return all_shapes, matched_residues
    return all_shapes, ()
