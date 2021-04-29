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
        'atoms': [make_tuple(opt, "atom") for opt in [AtomColorOption, AtomIdatmTypeOption, AtomRadiusOption,
            AtomShownOption, AtomStyleOption]],
        'bonds': [make_tuple(opt, "bond") for opt in [BondColorOption, BondHalfBondOption,
            BondLengthOption, BondRadiusOption, BondShownOption]],
        'pseudobonds': [make_tuple(opt, "pseudobond") for opt in [PBondColorOption, PBondHalfBondOption,
            PBondLengthOption, PBondRadiusOption, PBondShownOption]],
        'residues': [make_tuple(opt, "residue") for opt in [ResidueChi1Option, ResidueChi2Option,
            ResidueChi3Option, ResidueChi4Option, ResidueFilledRingOption, ResidueOmegaOption,
            ResiduePhiOption, ResiduePsiOption, ResidueRibbonColorOption, ResidueRibbonHidesBackboneOption,
            ResidueRibbonShownOption, ResidueRingColorOption, ResidueSSIDOption, ResidueSSTypeOption,
            ResidueThinRingsOption]],
    }[name]

from chimerax.ui.options import BooleanOption, ColorOption, EnumOption, FloatOption, IntOption, \
    SymbolicEnumOption
from chimerax.core.colors import color_name
from . import Atom, Element, Residue

class AtomColorOption(ColorOption):
    attr_name = "color"
    balloon = "Atom color"
    default = "white"
    name = "Color"
    @property
    def command_format(self):
        return "color %%s %s atoms" % color_name(self.value)

idatm_entries = list(Atom.idatm_info_map.keys()) + [nm for nm in Element.names if len(nm) < 3]
class AtomIdatmTypeOption(EnumOption):
    values = sorted(idatm_entries)
    attr_name = "idatm_type"
    balloon = "IDATM type"
    default = "C3"
    name = "IDATM type"
    @property
    def command_format(self):
        return "setattr %%s a idatmType %s" % self.value

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.enabled = False

class AtomRadiusOption(FloatOption):
    attr_name = "radius"
    balloon = "Atomic radius"
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
    balloon = "Atom/bond display style"
    default = 0
    name = "Style"
    @property
    def command_format(self):
        return "style %%s %s" % self.labels[self.values.index(self.value)]

class BaseBondColorOption(ColorOption):
    def __init_subclass__(cls, **kwargs):
        cls.prefix = "pseudo" if cls.__name__.startswith("PB") else ""
        cls.balloon = "If not in half bond mode, the color of the %sbond" % (cls.prefix)

    attr_name = "color"
    default = "white"
    name = "Color"

    @property
    def command_format(self):
        return "color %%s %s %sbonds" % (color_name(self.value), self.prefix)

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
        return "setattr %%s %s halfbond %s" % ("p" if self.prefix else "b", str(self.value).lower())

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
        cls.balloon = "%sbond radius" % cls.prefix

    attr_name = "radius"
    default = 0.2
    name = "Radius"
    @property
    def command_format(self):
        return "size %%s %s %g" % (("pseudobondRadius" if self.prefix == "pseudo" else "stickRadius"),
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
        return "%s %%s %sbonds" % (("show" if self.value else "hide"), self.prefix)

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

class ResidueChi1Option(FloatOption):
    attr_name = "chi1"
    balloon = "Side chain \N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT ONE} (chi1) angle"
    default = 0.0
    name = "\N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT ONE} angle"
    @property
    def command_format(self):
        return "setattr %%s r chi1 %g" % self.value

    def __init__(self, *args, **kw):
        if 'step' not in kw:
            kw['step'] = 1.0
        super().__init__(*args, **kw)

class ResidueChi2Option(FloatOption):
    attr_name = "chi2"
    balloon = "Side chain \N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT TWO} (chi2) angle"
    default = 0.0
    name = "\N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT ONE} angle"
    @property
    def command_format(self):
        return "setattr %%s r chi2 %g" % self.value

    def __init__(self, *args, **kw):
        if 'step' not in kw:
            kw['step'] = 1.0
        super().__init__(*args, **kw)

class ResidueChi3Option(FloatOption):
    attr_name = "chi3"
    balloon = "Side chain \N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT THREE} (chi3) angle"
    default = 0.0
    name = "\N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT ONE} angle"
    @property
    def command_format(self):
        return "setattr %%s r chi3 %g" % self.value

    def __init__(self, *args, **kw):
        if 'step' not in kw:
            kw['step'] = 1.0
        super().__init__(*args, **kw)

class ResidueChi4Option(FloatOption):
    attr_name = "chi4"
    balloon = "Side chain \N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT FOUR} (chi4) angle"
    default = 0.0
    name = "\N{GREEK SMALL LETTER CHI}\N{SUBSCRIPT ONE} angle"
    @property
    def command_format(self):
        return "setattr %%s r chi4 %g" % self.value

    def __init__(self, *args, **kw):
        if 'step' not in kw:
            kw['step'] = 1.0
        super().__init__(*args, **kw)

class ResidueOmegaOption(FloatOption):
    attr_name = "omega"
    balloon = "Backbone \N{GREEK SMALL LETTER OMEGA} (omega) angle"
    default = 0.0
    name = "\N{GREEK SMALL LETTER OMEGA} angle"
    @property
    def command_format(self):
        return "setattr %%s r omega %g" % self.value

    def __init__(self, *args, **kw):
        if 'step' not in kw:
            kw['step'] = 1.0
        super().__init__(*args, **kw)

class ResiduePhiOption(FloatOption):
    attr_name = "phi"
    balloon = "Backbone \N{GREEK SMALL LETTER PHI} (phi) angle"
    default = 0.0
    name = "\N{GREEK SMALL LETTER PHI} angle"
    @property
    def command_format(self):
        return "setattr %%s r phi %g" % self.value

    def __init__(self, *args, **kw):
        if 'step' not in kw:
            kw['step'] = 1.0
        super().__init__(*args, **kw)

class ResiduePsiOption(FloatOption):
    attr_name = "psi"
    balloon = "Backbone \N{GREEK SMALL LETTER PSI} (psi) angle"
    default = 0.0
    name = "\N{GREEK SMALL LETTER PSI} angle"
    @property
    def command_format(self):
        return "setattr %%s r psi %g" % self.value

    def __init__(self, *args, **kw):
        if 'step' not in kw:
            kw['step'] = 1.0
        super().__init__(*args, **kw)

class ResidueFilledRingOption(BooleanOption):
    attr_name = "ring_display"
    balloon = "Whether to depict rings as filled/solid"
    default = False
    name = "Fill rings"
    @property
    def command_format(self):
        return "setattr %%s r ring_display %s" % str(self.value).lower()

class ResidueRibbonColorOption(ColorOption):
    attr_name = "ribbon_color"
    default = "white"
    name = "Ribbon color"
    @property
    def command_format(self):
        return "color %%s %s target r" % color_name(self.value)

class ResidueRibbonHidesBackboneOption(BooleanOption):
    attr_name = "ribbon_hide_backbone"
    balloon = "Whether showing a ribbon depiction automatically hides backbone atoms.\n" \
        "Even with this off, the backbone atoms themselves may still need to be 'shown' to appear."
    default = True
    name = "Ribbon hides backbone"
    @property
    def command_format(self):
        return "setattr %%s r ribbon_hide_backbone %s" % str(self.value).lower()

class ResidueRibbonShownOption(BooleanOption):
    attr_name = "ribbon_display"
    default = False
    name = "Show ribbon"
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
        return "color %%s %s target f" % color_name(self.value)

class ResidueSSIDOption(IntOption):
    attr_name = "ss_id"
    balloon = "Secondary structures elements that are to be depicted as continuous\n" \
        "should have the same secondary structure ID number."
    default = -1
    name = "Secondary structure ID #"
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
        return "setattr %%s r ss_type %s" % self.labels[self.value]

class ResidueThinRingsOption(BooleanOption):
    attr_name = "thin_rings"
    balloon = "Whether to depict filled rings as thin or fat"
    default = False
    name = "Thin filled rings"
    @property
    def command_format(self):
        return "setattr %%s r thin_rings %s" % str(self.value).lower()

