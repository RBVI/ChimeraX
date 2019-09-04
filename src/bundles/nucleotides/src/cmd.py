# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2017 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# chimera nucleotides commands

from . import _data as NA
from . import default

from chimerax.core.errors import UserError
from chimerax.core.commands import (
    CmdDesc, EnumOf, TupleOf, BoolArg, FloatArg, DynamicEnum, Or,
    EmptyArg, StringArg, ObjectsArg, all_objects
)

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
                stubs_only=default.STUBS_ONLY, objects=None):

    if objects is None:
        objects = all_objects(session)
    residues = objects.atoms.unique_residues
    from chimerax.atomic import Residue
    residues = residues.filter(residues.polymer_types == Residue.PT_NUCLEIC)
    if len(residues) == 0:
        return

    if representation == 'atoms':
        # hide filled rings
        residues.ring_displays = False
        # reset nucleotide info
        NA.set_normal(residues)
    elif representation == 'fill':
        # show filled rings
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


def run_provider(session, name, display_name):
    from chimerax.shortcuts.shortcuts import if_sel_atoms
    if display_name == "Plain":
        if_sel_atoms("nucleotides sel atoms; style nucleic & sel stick",
                          "nucleotides atoms; style nucleic stick")(session)
    elif display_name == "Filled":
        if_sel_atoms("nucleotides sel fill; style nucleic & sel stick",
                           "nucleotides fill; style nucleic stick")(session)
    elif display_name == "Slab":
        if_sel_atoms("nucleotides sel slab; style nucleic & sel stick",
                           "nucleotides slab; style nucleic stick")(session)
    elif display_name == "Tube/\nSlab":
        if_sel_atoms("nucleotides sel tube/slab shape box")(session)
    elif display_name == "Tube/\nEllipsoid":
        if_sel_atoms("nucleotides sel tube/slab shape ellipsoid")(session)
    elif display_name == "Tube/\nMuffler":
        if_sel_atoms("nucleotides sel tube/slab shape muffler")(session)
    elif display_name == "Ladder":
        if_sel_atoms("nucleotides sel ladder")(session)
    elif display_name == "Stubs":
        if_sel_atoms("nucleotides sel stubs")(session)
    elif display_name == "nucleotide":
        if_sel_atoms("color sel bynuc")(session)
    else:
        session.logger.warning("Unknown nucleotides provider: %r" % name)
