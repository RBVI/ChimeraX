# vim: set expandtab shiftwidth=4 softtabstop=4:
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
