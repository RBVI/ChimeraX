# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def item_options(session, name, **kw):
    from .triggers import get_triggers
    def make_tuple(option, reason_type, triggers=get_triggers()):
        return (option, (triggers, "changes", lambda changes, *, attr=option.attr_name, rt=reason_type:
            attr + ' changed' in getattr(changes, rt + "_reasons")()))
    return {
        'atoms': [make_tuple(opt, "atom") for opt in [AtomBFactorOption, AtomColorOption,
            AtomIdatmTypeOption, AtomOccupancyOption, AtomRadiusOption, AtomShownOption, AtomStyleOption]],
        'bonds': [make_tuple(opt, "bond") for opt in [BondColorOption, BondHalfBondOption,
            BondLengthOption, BondRadiusOption, BondShownOption]],
        'pseudobond groups': [make_tuple(opt, "pseudobond_group") for opt in [PBGColorOption,
            PBGDashesOption, PBGHalfBondOption, PBGNameOption, PBGRadiusOption]],
        'pseudobonds': [make_tuple(opt, "pseudobond") for opt in [PBondColorOption, PBondHalfBondOption,
            PBondLengthOption, PBondRadiusOption, PBondShownOption]],
        'residues': [make_tuple(opt, "residue") for opt in [ResidueChi1Option, ResidueChi2Option,
            ResidueChi3Option, ResidueChi4Option, ResidueFilledRingOption, ResidueOmegaOption,
            ResiduePhiOption, ResiduePsiOption, ResidueRibbonColorOption, ResidueRibbonHidesBackboneOption,
            ResidueRibbonShownOption, ResidueRingColorOption, ResidueSSIDOption, ResidueSSTypeOption,
            ResidueThinRingsOption]],
        'structures': [make_tuple(opt, "structure") for opt in [StructureAutochainOption,
            StructureBallScaleOption, StructureNameOption, StructureShownOption]],
    }[name]

from chimerax.ui.options import BooleanOption, ColorOption, EnumOption, FloatOption, IntOption, \
    SymbolicEnumOption, StringOption
from chimerax.core.colors import color_name
from chimerax.core.commands import StringArg
from chimerax.core.utils import CustomSortString
from . import Atom, Element, Residue, Structure

color_arg = lambda x: StringArg.unparse(color_name(x))

class AtomBFactorOption(FloatOption):
    attr_name = "bfactor"
    default = 1.0
    name = "B-factor"
    @property
    def command_format(self):
        return "setattr %%s a bfactor %g" % self.value

    def __init__(self, *args, **kw):
        if 'decimal_places' not in kw:
            kw['decimal_places'] = 2
        if 'step' not in kw:
            kw['step'] = 0.5
        super().__init__(*args, **kw)

class AtomColorOption(ColorOption):
    attr_name = "color"
    default = "white"
    name = "Color"
    @property
    def command_format(self):
        return "color %%s %s atoms" % color_arg(self.value)

idatm_entries = list(Atom.idatm_info_map.keys()) + [nm for nm in Element.names if len(nm) < 3]
class AtomIdatmTypeOption(EnumOption):
    values = sorted(idatm_entries)
    attr_name = "idatm_type"
    default = "C3"
    name = "IDATM type"
    @property
    def command_format(self):
        return "setattr %%s a idatmType %s" % self.value

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.enabled = False

class AtomOccupancyOption(FloatOption):
    attr_name = "occupancy"
    default = 1.0
    name = "Occupancy"
    @property
    def command_format(self):
        return "setattr %%s a occupancy %g" % self.value

    def __init__(self, *args, **kw):
        if 'decimal_places' not in kw:
            kw['decimal_places'] = 2
        if 'step' not in kw:
            kw['step'] = 0.1
        super().__init__(*args, min=0.0, max=1.0, **kw)

class AtomRadiusOption(FloatOption):
    attr_name = "radius"
    default = 1.4
    name = "Radius"
    @property
    def command_format(self):
        return "size %%s atomRadius %g" % self.value

    def __init__(self, *args, **kw):
        super().__init__(*args, min='positive', **kw)

class AtomShownOption(BooleanOption):
    attr_name = "display"
    default = True
    name = "Shown"
    @property
    def command_format(self):
        return "%s %%s atoms" % ("show" if self.value else "hide")

class AtomStyleOption(SymbolicEnumOption):
    values = (1, 0, 2)
    labels = ("ball", "sphere", "stick")
    attr_name = "draw_mode"
    default = 0
    name = "Style"
    @property
    def command_format(self):
        return "style %%s %s" % self.labels[self.values.index(self.value)]

class BaseBondColorOption(ColorOption):
    def __init_subclass__(cls, **kwargs):
        cls.prefix = "pseudo" if cls.__name__.startswith("PB") else ""
        cls.balloon = "If not in halfbond mode, the color of the %sbond" % (cls.prefix)

    attr_name = "color"
    default = "white"
    name = "Color"

    @property
    def command_format(self):
        return "color =%%s %s %sbonds" % (color_arg(self.value), self.prefix)

class BaseBondHalfBondOption(BooleanOption):
    def __init_subclass__(cls, **kwargs):
        cls.prefix = "pseudo" if cls.__name__.startswith("PB") else ""
        cls.balloon = "If true, each half of the %sbond is colored the same as the neighboring atom.\n" \
            "Otherwise, the %sbond uses its own color attribute for the whole %sbond." % (cls.prefix,
            cls.prefix, cls.prefix)

    attr_name = "halfbond"
    default = True
    name = "Halfbond mode"
    @property
    def command_format(self):
        return "setattr =%%s %s halfbond %s" % ("p" if self.prefix else "b", str(self.value).lower())

class BaseBondLengthOption(FloatOption):
    def __init_subclass__(cls, **kwargs):
        cls.prefix = "pseudo" if cls.__name__.startswith("PB") else ""
        cls.balloon = "The length of the %sbond, in angstroms" % (cls.prefix)

    attr_name = "length"
    default = 0.0
    name = "Length"
    @property
    def command_format(self):
        raise ValueError("Cannot set %sbond length" % self.prefix)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.enabled = False

class BaseBondRadiusOption(FloatOption):
    def __init_subclass__(cls, **kwargs):
        cls.prefix = "pseudo" if cls.__name__.startswith("PB") else ""

    attr_name = "radius"
    balloon = "stick radius"
    default = 0.2
    name = "Radius"
    @property
    def command_format(self):
        return "size =%%s %s %g" % (("pseudobondRadius" if self.prefix == "pseudo" else "stickRadius"),
            self.value)

    def __init__(self, *args, **kw):
        super().__init__(*args, min='positive', **kw)

class BaseBondShownOption(BooleanOption):
    def __init_subclass__(cls, **kwargs):
        cls.prefix = "pseudo" if cls.__name__.startswith("PB") else ""
        cls.balloon = "If true, the %sbond is shown if both its neighboring atoms are shown.\n" \
            "If false, the %sbond is not shown." % (cls.prefix, cls.prefix)

    attr_name = "display"
    default = True
    name = "Shown"
    @property
    def command_format(self):
        # have to use setattr because "show" will also display flanking atoms if needed
        return "setattr =%%s %s display %s" % ("p" if self.prefix else "b", str(self.value).lower())

class BondColorOption(BaseBondColorOption):
    pass

class BondHalfBondOption(BaseBondHalfBondOption):
    pass

class BondLengthOption(BaseBondLengthOption):
    pass

class BondRadiusOption(BaseBondRadiusOption):
    pass

class BondShownOption(BaseBondShownOption):
    pass

class PBondColorOption(BaseBondColorOption):
    pass

class PBondHalfBondOption(BaseBondHalfBondOption):
    pass

class PBondLengthOption(BaseBondLengthOption):
    pass

class PBondRadiusOption(BaseBondRadiusOption):
    pass

class PBondShownOption(BaseBondShownOption):
    pass

class AngleOption(FloatOption):
    def __init__(self, *args, **kw):
        if 'decimal_places' not in kw:
            kw['decimal_places'] = 1
        if 'step' not in kw:
            kw['step'] = 5.0
        super().__init__(*args, **kw)

class PBGColorOption(ColorOption):
    attr_name = "color"
    balloon = "Pseudobond model color.  Setting it will set all member pseudobonds\n" \
        "to that color, and newly created pseudobonds will be that color."
    default = "gold"
    name = "Color"
    @property
    def command_format(self):
        return "setattr %%s g color %s" % color_arg(self.value)

class PBGDashesOption(IntOption):
    attr_name = "dashes"
    balloon = "Number of dashes per pseudobond.  Zero gives a solid stick.\n" \
        "Currently odd values are rounded down to the next even value."
    default = 9
    name = "Dashes"
    @property
    def command_format(self):
        return "style %%s dashes %d" % self.value

    def __init__(self, *args, **kw):
        super().__init__(*args, min=0, **kw)

class PBGHalfBondOption(BooleanOption):
    balloon = "If true, each half of the pseudobonds are colored the same as their neighboring atoms.\n" \
            "Otherwise, the pseudobonds use their own color attribute for the whole pseudobond."
    attr_name = "halfbond"
    default = True
    name = "Halfbond mode"
    @property
    def command_format(self):
        return "setattr %%s g halfbond %s" % str(self.value).lower()

class PBGNameOption(StringOption):
    attr_name = "name"
    default = "unknown"
    name = "Name"
    @property
    def command_format(self):
        return "setattr %%s g name %s" % StringArg.unparse(self.value)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.enabled = False

class PBGRadiusOption(FloatOption):
    attr_name = "radius"
    balloon = "Pseudobond radius"
    default = 1.4
    name = "Radius"
    @property
    def command_format(self):
        return "setattr %%s g radius %g" % self.value

    def __init__(self, *args, **kw):
        super().__init__(*args, min='positive', **kw)

class ResidueChi1Option(AngleOption):
    attr_name = "chi1"
    balloon = "Side chain \N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT ONE} (chi1) angle"
    default = 0.0
    name = CustomSortString("\N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT ONE} angle", 4)
    @property
    def command_format(self):
        return "setattr %%s r chi1 %g" % self.value

class ResidueChi2Option(AngleOption):
    attr_name = "chi2"
    balloon = "Side chain \N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT TWO} (chi2) angle"
    default = 0.0
    name = CustomSortString("\N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT TWO} angle", 5)
    @property
    def command_format(self):
        return "setattr %%s r chi2 %g" % self.value

class ResidueChi3Option(AngleOption):
    attr_name = "chi3"
    balloon = "Side chain \N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT THREE} (chi3) angle"
    default = 0.0
    name = CustomSortString("\N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT THREE} angle", 6)
    @property
    def command_format(self):
        return "setattr %%s r chi3 %g" % self.value

class ResidueChi4Option(AngleOption):
    attr_name = "chi4"
    balloon = "Side chain \N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT FOUR} (chi4) angle"
    default = 0.0
    name = CustomSortString("\N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT FOUR} angle", 7)
    @property
    def command_format(self):
        return "setattr %%s r chi4 %g" % self.value

class ResidueOmegaOption(AngleOption):
    attr_name = "omega"
    balloon = "Backbone \N{GREEK SMALL LETTER OMEGA} (omega) angle"
    default = 0.0
    name = CustomSortString("\N{GREEK SMALL LETTER OMEGA} angle", 3)
    @property
    def command_format(self):
        return "setattr %%s r omega %g" % self.value

class ResiduePhiOption(AngleOption):
    attr_name = "phi"
    balloon = "Backbone \N{GREEK SMALL LETTER PHI} (phi) angle"
    default = 0.0
    name = CustomSortString("\N{GREEK SMALL LETTER PHI} angle", 1)
    @property
    def command_format(self):
        return "setattr %%s r phi %g" % self.value

class ResiduePsiOption(AngleOption):
    attr_name = "psi"
    balloon = "Backbone \N{GREEK SMALL LETTER PSI} (psi) angle"
    default = 0.0
    name = CustomSortString("\N{GREEK SMALL LETTER PSI} angle", 2)
    @property
    def command_format(self):
        return "setattr %%s r psi %g" % self.value

class ResidueFilledRingOption(BooleanOption):
    attr_name = "ring_display"
    balloon = "Whether ring fill should be thick or thin"
    default = False
    name = "Fill rings"
    @property
    def command_format(self):
        return "setattr %%s r ring_display %s" % str(self.value).lower()

class ResidueRibbonColorOption(ColorOption):
    attr_name = "ribbon_color"
    default = "white"
    name = "Cartoon color"
    @property
    def command_format(self):
        return "color %%s %s target r" % color_arg(self.value)

class ResidueRibbonHidesBackboneOption(BooleanOption):
    attr_name = "ribbon_hide_backbone"
    balloon = "Whether showing a cartoon depiction automatically hides backbone atoms.\n" \
        "Even with this off, the backbone atoms themselves may still need to be 'shown' to appear."
    default = True
    name = "Cartoon hides backbone"
    @property
    def command_format(self):
        return "setattr %%s r ribbon_hide_backbone %s" % str(self.value).lower()

class ResidueRibbonShownOption(BooleanOption):
    attr_name = "ribbon_display"
    default = False
    name = "Show cartoon"
    @property
    def command_format(self):
        return "cartoon%s %%s" % ("" if self.value else " hide")

class ResidueRingColorOption(ColorOption):
    attr_name = "ring_color"
    balloon = "The fill color of a filled ring"
    default = "white"
    name = "Ring fill color"
    @property
    def command_format(self):
        return "color %%s %s target f" % color_arg(self.value)

class ResidueSSIDOption(IntOption):
    attr_name = "ss_id"
    balloon = "Secondary structures elements that are to be depicted as continuous\n" \
        "should have the same secondary structure ID number. Typically coil is\n" \
        "assigned a value of 0 and non-protein residues -1."
    default = -1
    name = "Secondary structure ID"
    @property
    def command_format(self):
        return "setattr %%s r ss_id %d" % self.value

class ResidueSSTypeOption(SymbolicEnumOption):
    values = (Residue.SS_COIL, Residue.SS_HELIX, Residue.SS_STRAND)
    labels = ("coil", "helix", "strand")
    attr_name = "ss_type"
    default = 0.0
    name = "Secondary structure type"
    @property
    def command_format(self):
        return "setattr %%s r ss_type %d" % self.value

class ResidueThinRingsOption(SymbolicEnumOption):
    values = (False, True)
    labels = ("thick", "thin")
    attr_name = "thin_rings"
    balloon = "Whether to depict filled rings as thin or fat"
    default = False
    name = "Ring fill style"
    @property
    def command_format(self):
        return "setattr %%s r thin_rings %s" % str(self.value).lower()

class StructureAutochainOption(BooleanOption):
    attr_name = "autochain"
    balloon = "Fraction of atomic radius to use in ball-and-stick style"
    default = True
    name = "Autochain"
    @property
    def command_format(self):
        return "setattr %%s structures autochain %s" % str(self.value).lower()

class StructureBallScaleOption(FloatOption):
    attr_name = "ball_scale"
    balloon = "Fraction of atomic radius to use in ball-and-stick style"
    default = 0.25
    name = "Ball scale"
    @property
    def command_format(self):
        return "size %%s ballScale %g" % self.value

    def __init__(self, *args, **kw):
        if 'decimal_places' not in kw:
            kw['decimal_places'] = 2
        if 'step' not in kw:
            kw['step'] = .01
        super().__init__(*args, **kw)

class StructureNameOption(StringOption):
    attr_name = "name"
    default = "unknown"
    name = "Name"
    @property
    def command_format(self):
        return "setattr %%s structures name %s" % StringArg.unparse(self.value)

class StructureRibbonTetherOpacityOption(FloatOption):
    attr_name = "ribbon_tether_opacity"
    balloon = "How opaque (non-transparent) the cartoon tether is"
    default = 0.5
    name = "Cartoon tether opacity"
    @property
    def command_format(self):
        return "cartoon tether %%s opacity %g" % self.value

    def __init__(self, *args, **kw):
        if 'decimal_places' not in kw:
            kw['decimal_places'] = 2
        if 'step' not in kw:
            kw['step'] = .05
        super().__init__(*args, **kw)

class StructureRibbonTetherScaleOption(FloatOption):
    attr_name = "ribbon_tether_scale"
    balloon = "Size of tether base radius relative to\nthe display radius of the corresponding α-carbon"
    default = 1.0
    name = "Cartoon tether scale"
    @property
    def command_format(self):
        return "cartoon tether %%s scale %g" % self.value

    def __init__(self, *args, **kw):
        if 'decimal_places' not in kw:
            kw['decimal_places'] = 2
        if 'step' not in kw:
            kw['step'] = .05
        super().__init__(*args, **kw)

class StructureRibbonTetherShapeOption(SymbolicEnumOption):
    values = (Structure.TETHER_CONE, Structure.TETHER_REVERSE_CONE, Structure.TETHER_CYLINDER)
    labels = ("cone", "steeple", "cylinder")
    attr_name = "ribbon_tether_shape"
    default = Structure.TETHER_CONE
    name = "Cartoon tether shape"
    @property
    def command_format(self):
        return "cartoon tether %%s shape %s" % self.labels[self.value]

class StructureRibbonTetherSidesOption(IntOption):
    attr_name = "ribbon_tether_sides"
    balloon = "Number of planar facets used to draw a tether"
    default = 4
    name = "Cartoon tether sides"
    @property
    def command_format(self):
        return "cartoon tether %%s sides %d" % self.value

    def __init__(self, *args, **kw):
        super().__init__(*args, min=3, max=10, **kw)

class StructureShownOption(BooleanOption):
    attr_name = "display"
    default = True
    name = "Shown"
    @property
    def command_format(self):
        return "%s %%s models" % ("show" if self.value else "hide")

class StructureHelixModeOption(SymbolicEnumOption):
    values = (Structure.RIBBON_MODE_DEFAULT, Structure.RIBBON_MODE_ARC, Structure.RIBBON_MODE_WRAP)
    labels = ("default", "tube", "wrap")
    attr_name = "ribbon_mode_helix"
    balloon = "How peptide helices are depicted in cartoons.  If 'default', depicted as\n" \
        "a ribbon whose flat surface faces outward from the local curvature of the\n" \
        "peptide chain.  If 'tube', as a cylinder that follows the overall curvature\n" \
        "of the helix path.  If 'wrap', similar to 'default' except following the same\n" \
        "curved path as 'tube'."
    default = Structure.RIBBON_MODE_DEFAULT
    name = "Cartoon helix style"
    @property
    def command_format(self):
        return "cartoon style %%s modeHelix %s" % self.labels[self.values.index(self.value)]

