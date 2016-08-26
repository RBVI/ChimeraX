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

"""
idatm: info relevant to use of IDATM types
===============================

TODO
"""

from bond_geom import tetrahedral, planar, linear, single, geometry_name
import chimera

typeInfo = chimera.Atom.getIdatmInfoMap()

registrant = "idatm"
selCategory = 'IDATM type'
try:
	import chimera
	from chimera.selection.manager import selMgr
except ImportError:
	pass
else:
	for idatmType in typeInfo.keys():
		selectorText = """\
selAdd = []
for mol in molecules:
        for atom in mol.atoms:
		if atom.idatmType == '%s':
			selAdd.append(atom)
sel.add(selAdd)
""" % (idatmType,)
		selMgr.addSelector(registrant, [selMgr.CHEMISTRY, selCategory,
			idatmType], selectorText,
			description=typeInfo[idatmType].description)
	selMgr.makeCallbacks()

	del idatmType, selectorText, selMgr
