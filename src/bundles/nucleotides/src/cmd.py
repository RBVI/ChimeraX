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

# chimera nucleotides commands

from . import _data as NA
from . import default

from chimerax.core.errors import UserError
from chimerax.core.commands import (
    CmdDesc, EnumOf, TupleOf, BoolArg, FloatArg, DynamicEnum, Or,
    EmptyArg, StringArg, ObjectsArg, all_objects
)
from chimerax.core.undo import UndoAction, UndoAggregateAction, UndoState

AnchorArg = EnumOf(("base", "ribose"))
Float4Arg = TupleOf(FloatArg, 4, name="lower left x, y, upper right x, y")
nucleotides_dimensions_desc = CmdDesc(
    required=[("name", StringArg)],
    keyword=[
        ("anchor", AnchorArg),
        ("purine", Float4Arg),
        ("pyrimidine", Float4Arg),
    ],
    synopsis="Create custom nucleotide dimensions")


def nucleotides_dimensions(session, name, anchor=None, purine=None, pyrimidine=None):
    if anchor is None or purine is None or pyrimidine is None:
        raise UserError("Incomplete dimensions")
    info = {
        NA.ANCHOR: anchor,
        NA.PURINE: (purine[0:2], purine[2:4]),
        NA.PYRIMIDINE: (pyrimidine[0:2], pyrimidine[2:4]),
        NA.PSEUDO_PYRIMIDINE: (pyrimidine[0:2], pyrimidine[2:4]),
    }
    NA.add_dimensions(name, info, session=session)


nucleotides_dimensions_list_desc = CmdDesc(
    synopsis="List custom nucleotide dimensions")


def nucleotides_dimensions_list(session):
    dimensions = NA.list_dimensions()
    from chimerax.core.commands import commas
    session.logger.info("Nucleotides dimensions: " + commas(dimensions, 'and'))


nucleotides_dimensions_delete_desc = CmdDesc(
    required=[('name', DynamicEnum(
        lambda: NA.list_dimensions(custom_only=True),
        name="Custom nucleotides dimensions"))],
    synopsis="Delete custom nucleotide dimensions")


def nucleotides_dimensions_delete(session, name):
    NA.remove_dimensions(name)


ShapeArg = EnumOf(('box', 'muffler', 'ellipsoid'))
DimensionsArg = DynamicEnum(NA.list_dimensions)
ReprArg = EnumOf(('atoms', 'fill', 'slab', 'tube/slab', 'ladder', 'stubs'))


nucleotides_desc = CmdDesc(
    required=[
        ("objects", Or(ObjectsArg, EmptyArg)),
        ("representation", ReprArg)
    ],
    keyword=[
        ("show_orientation", BoolArg),
        ("shape", ShapeArg),
        ("dimensions", DimensionsArg),
        ("thickness", FloatArg),
        ("hide_atoms", BoolArg),
        ("glycosidic", BoolArg),
        ("base_only", BoolArg),
        ("show_stubs", BoolArg),
        ("radius", FloatArg),
    ],
    synopsis="Manipulate nucleotide representations")


def nucleotides(session, representation, *,
                glycosidic=default.GLYCOSIDIC, show_orientation=default.ORIENT,
                thickness=default.THICKNESS, hide_atoms=default.HIDE,
                shape=default.SHAPE, dimensions=default.DIMENSIONS, radius=None,
                show_stubs=default.SHOW_STUBS, base_only=default.BASE_ONLY,
                stubs_only=default.STUBS_ONLY, objects=None, create_undo=True):

    if objects is None:
        objects = all_objects(session)
    residues = objects.atoms.unique_residues
    from chimerax.atomic import Residue
    residues = residues.filter(residues.polymer_types == Residue.PT_NUCLEIC)
    if len(residues) == 0:
        return

    if create_undo:
        undo_state = UndoState('nucleotides %s' % representation)
        nucleic_undo = _NucleicUndo(
                'nucleotides %s' % representation, session, representation, glycosidic,
                show_orientation, thickness, hide_atoms, shape, dimensions, radius,
                show_stubs, base_only, stubs_only, residues)
        undo = UndoAggregateAction('nucleotides %s' % representation, [undo_state, nucleic_undo])

    if representation == 'atoms':
        # hide filled rings
        if create_undo:
            undo_state.add(residues, "ring_displays", residues.ring_displays, False)
        residues.ring_displays = False
        # reset nucleotide info
        NA.set_normal(residues)
    elif representation == 'fill':
        # show filled rings
        if create_undo:
            undo_state.add(residues, "ring_displays", residues.ring_displays, True)
        residues.ring_displays = True
        # set nucleotide info
        if show_orientation:
            NA.set_orient(residues)
        else:
            NA.set_normal(residues)
    elif representation.endswith('slab'):
        if radius is None:
            radius = default.TUBE_RADIUS
        if dimensions is None:
            if shape == 'ellipsoid':
                dimensions = 'small'
            else:
                dimensions = 'long'
        if representation == 'slab':
            if create_undo:
                undo_state.add(residues, "ring_displays", residues.ring_displays, True)
            residues.ring_displays = True
            show_gly = True
        else:
            show_gly = glycosidic
        if show_gly:
            info = NA.find_dimensions(dimensions)
            show_gly = info[NA.ANCHOR] != NA.RIBOSE
        NA.set_slab(representation, residues, dimensions=dimensions,
                    thickness=thickness, orient=show_orientation,
                    shape=shape, show_gly=show_gly, hide=hide_atoms,
                    tube_radius=radius)
    elif representation in ('ladder', 'stubs'):
        if radius is None:
            radius = default.RUNG_RADIUS
        stubs_only = representation == 'stubs'
        NA.set_ladder(residues, rung_radius=radius, stubs_only=stubs_only,
                      show_stubs=show_stubs, skip_nonbase_Hbonds=base_only, hide=hide_atoms)

    if create_undo:
        session.undo.register(undo)


class _NucleicUndo(UndoAction):

    def __init__(self, name, session, representation, glycosidic, show_orientation,
                 thickness, hide_atoms, shape, dimensions, radius, show_stubs,
                 base_only, stubs_only, residues, can_redo=True):
        super().__init__(name, can_redo)
        self.session = session
        self.representation = representation
        self.glycosidic = glycosidic
        self.show_orientation = show_orientation
        self.thickness = thickness
        self.hide_atoms = hide_atoms
        self.shape = shape
        self.dimensions = dimensions
        self.radius = radius
        self.show_stubs = show_stubs
        self.base_only = base_only
        self.stubs_only = stubs_only
        self.residues = residues
        ns = NA._nucleotides(session)
        self.before_state = ns.take_snapshot(session, NA.NucleotideState.SCENE)

    def undo(self):
        NA.NucleotideState.restore_snapshot(self.session, self.before_state)

    def redo(self):
        nucleotides(
            self.session, self.representation,
            glycosidic=self.glycosidic, show_orientation=self.show_orientation,
            thickness=self.thickness, hide_atoms=self.hide_atoms,
            shape=self.shape, dimensions=self.dimensions, radius=self.radius,
            show_stubs=self.show_stubs, base_only=self.base_only,
            stubs_only=self.stubs_only, objects=self.residues,
            create_undo=False)


def run_provider(session, name, display_name):
    from chimerax.shortcuts.shortcuts import run_on_atoms
    if display_name == "Plain":
        run_on_atoms("nucleotides %s atoms; style nucleic & %s stick",
                     "nucleotides atoms; style nucleic stick")(session)
    elif display_name == "Filled":
        run_on_atoms("nucleotides %s fill; style nucleic & %s stick",
                     "nucleotides fill; style nucleic stick")(session)
    elif display_name == "Slab":
        run_on_atoms("nucleotides %s slab; style nucleic & %s stick",
                     "nucleotides slab; style nucleic stick")(session)
    elif display_name == "Tube/\nSlab":
        run_on_atoms("nucleotides %s tube/slab shape box")(session)
    elif display_name == "Tube/\nEllipsoid":
        run_on_atoms("nucleotides %s tube/slab shape ellipsoid")(session)
    elif display_name == "Tube/\nMuffler":
        run_on_atoms("nucleotides %s tube/slab shape muffler")(session)
    elif display_name == "Ladder":
        run_on_atoms("nucleotides %s ladder")(session)
    elif display_name == "Stubs":
        run_on_atoms("nucleotides %s stubs")(session)
    elif display_name == "nucleotide":
        run_on_atoms("color %s bynuc")(session)
    else:
        session.logger.warning("Unknown nucleotides provider: %r" % name)
