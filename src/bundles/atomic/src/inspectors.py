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
        'atoms': [make_tuple(opt, "atom") for opt in [AtomColorOption, AtomStyleOption]],
        'bonds': [make_tuple(opt, "bond") for opt in []]
    }[name]

from chimerax.ui.options import RGBAOption, SymbolicEnumOption

class AtomColorOption(RGBAOption):
    attr_name = "color"
    balloon = "Atom color"
    default = "white"
    name = "Color"
    @property
    def command_format(self):
        return "color %%s %g,%g,%g,%g atoms" % tuple([100.0 * x for x in self.value])

class AtomStyleOption(SymbolicEnumOption):
    values = (0, 1, 2)
    labels = ("sphere", "ball", "stick")
    attr_name = "draw_mode"
    balloon = "Atom/bond display style"
    default = 0
    name = "Style"
    @property
    def command_format(self):
        return "style %%s %s" % self.labels[self.value]

