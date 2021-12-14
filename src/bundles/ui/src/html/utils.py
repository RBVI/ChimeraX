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

def disclosure(contents, *, summary=None, background_color=None, open=False):
	return '<details%s%s>%s%s</details>' % (
		' open' if open else "",
		(' style="background-color: %s"' % background_color) if background_color is not None else "",
		('<summary>%s</summary>' % summary) if summary is not None else "",
		contents
	)
