# vim: set expandtab sw=4:
# chimera nucleotides commands
#
#       nucleotides ndbcolor [atom-spec]
#       nucleotides [atom-spec] sidechain_type [keyword options]
#               * show side chains -- sugar/bases
#               * type is one of atoms, fill/fill, fill/slab, tube/slab, ladder
#               * fill/fill is same as:
#                       nuc sidechain atoms atom-spec; fillring thick atom-spec
#               * options vary with sidechain type
#               * fill/fill, fill/slab, tube/slab:
#                       * orient (true|false)
#               * fill/slab, tube/slab:
#                       * shape (box|tube|ellipsoid)
#                       * style slab-style
#                       * thickness 0.5
#                       * hide (true|false)
#               * tube-slab:
#                       * glycosidic (true|false)
#                         (shows separate glycosidic bond)
#               * ladder:
#                       * radius 0.45
#                         (radius of ladder rungs and only applies to base-base
#                         H-bonds -- zero means use defaults)
#                       * stubs (true|false)
#                       * ignore (true|false)
#                         (ignore non-base H-bonds)
#       nucleotides add style-name options
#               * create/modify slab style, all options are required
#               * options are:
#                       anchor (sugar|base)
#                       (purine|pyrimidine) (lower-left|ll|upper-right|ur) x y
#       nucleotides delete style-name
#               * delete style

# look at movie, volume, hbond commands for ideas

from . import _data as NA
from . import default

from chimerax.core.errors import UserError
from chimerax.core.commands import (
    CmdDesc, EnumOf, TupleOf, BoolArg, FloatArg, DynamicEnum, Or,
    EmptyArg, StringArg, ObjectsArg, all_objects
)

AnchorArg = EnumOf(("base", "sugar"))
Float4Arg = TupleOf(FloatArg, 4, name="lower left x, y, upper right x, y")
nucleotides_style_desc = CmdDesc(
    required=[("name", StringArg)],
    keyword=[
        ("anchor", AnchorArg),
        ("purine", Float4Arg),
        ("pyrimidine", Float4Arg),
    ],
    synopsis="Create custom nucleotide style")


def nucleotides_style(session, name, anchor=None, purine=None, pyrimidine=None):
    if anchor is None or purine is None or pyrimidine is None:
        raise UserError("Incomplete style")
    info = {
        NA.ANCHOR: anchor,
        NA.PURINE: (purine[0:2], purine[2:4]),
        NA.PYRIMIDINE: (pyrimidine[0:2], pyrimidine[2:4]),
        NA.PSEUDO_PYRIMIDINE: (pyrimidine[0:2], pyrimidine[2:4]),
    }
    NA.add_style(name, info, session=session)


nucleotides_style_list_desc = CmdDesc(
    synopsis="List custom nucleotide styles")


def nucleotides_style_list(session):
    styles = NA.list_styles()
    from chimerax.core.commands import commas
    session.logger.info("Nucleotides styles: " + commas(styles, ' and'))


nucleotides_style_delete_desc = CmdDesc(
    required=[('name', DynamicEnum(
        lambda: NA.list_styles(custom_only=True),
        name="Custom nucleotides style"))],
    synopsis="Delete custom nucleotide style")


def nucleotides_style_delete(session, name):
    NA.remove_style(name)


ShapeArg = EnumOf(('box', 'tube', 'ellipsoid'))
StyleArg = DynamicEnum(NA.list_styles, name='a nucleotide style')
# ReprArg = EnumOf(('atoms', 'fill/fill', 'fill/slab', 'tube/slab', 'ladder'))
ReprArg = EnumOf(('atoms', 'slab', 'tube/slab', 'ladder'))


nucleotides_desc = CmdDesc(
    required=[
        ("objects", Or(ObjectsArg, EmptyArg)),
        ("representation", ReprArg)
    ],
    keyword=[
        ("orient", BoolArg),
        ("shape", ShapeArg),
        ("style", StyleArg),
        ("thickness", FloatArg),
        ("hide", BoolArg),
        ("glycosidic", BoolArg),
        ("ignore", BoolArg),
        ("stubs", BoolArg),
        ("radius", FloatArg),
    ],
    synopsis="Manipulate nucleotide representations")


def nucleotides(session, representation, *, glycosidic=default.GLYCOSIDIC, orient=default.ORIENT,
                thickness=default.THICKNESS, hide=default.HIDE,
                shape=default.SHAPE, style=default.STYLE, radius=default.RADIUS,
                stubs=default.STUBS, ignore=default.IGNORE,
                useexisting=default.USE_EXISTING, objects=None):

    if objects is None:
        objects = all_objects(session)
    residues = objects.atoms.unique_residues
    from chimerax.core.atomic import Residue
    residues = residues.filter(residues.polymer_types == Residue.PT_NUCLEIC)
    molecules = residues.unique_structures
    if representation == 'atoms':
        # TODO: residues.fill_rings = False
        NA.set_normal(molecules, residues)
    elif representation == 'fill/fill':
        # TODO
        # residues.fill_rings = True
        if orient:
            NA.set_orient(molecules, residues)
        else:
            NA.set_normal(molecules, residues)
    elif representation.endswith('slab'):
        if representation.startswith('fill'):
            # TODO: residues.fill_rings = True
            show_gly = True
        else:
            show_gly = glycosidic
        if show_gly:
            info = NA.find_style(style)
            show_gly = info[NA.ANCHOR] != NA.SUGAR
        NA.set_slab(representation, molecules, residues, style=style,
                    thickness=thickness, orient=orient,
                    shape=shape, show_gly=show_gly, hide=hide)
    elif representation == 'ladder':
        NA.set_ladder(molecules, residues, rung_radius=radius,
                      show_stubs=stubs, skip_nonbase_Hbonds=ignore)
    NA._rebuild(None, None)  # TODO


nucleotides_ndbcolor_desc = CmdDesc(
    required=[
        ("objects", Or(ObjectsArg, EmptyArg)),
    ],
    synopsis="Color residues according to Nucleic Acid Database conventions")


def nucleotides_ndbcolor(session, objects=None):
    if objects is None:
        objects = all_objects(session)
    residues = objects.atoms.unique_residues
    from chimerax.core.atomic import Residue
    residues = residues.filter(residues.polymer_types == Residue.PT_NUCLEIC)
    NA.ndb_color(residues)
