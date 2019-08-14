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
    return {
        'atoms': [AtomColorOption]
    }[name]

from chimerax.ui.options import RGBAOption

class AtomColorOption(RGBAOption):
    balloon = "Atom color\n\nattribute: color"
    @property
    def command_format(self):
        return "color %%s %g,%g,%g,%g atoms" % self.value
