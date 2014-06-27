# run as:
#
#	chimera --nogui --silent pdb:3fx2 mol2bild.py > 3fx2.bild
#

import chimera

_cur_color = None
_cur_opacity = None

def _emit_color(color):
	global _cur_color, _cur_opacity
	rgba = color.rgba()
	if rgba[0:3] != _cur_color:
		_cur_color = rgba[0:3]
		print '.color', rgba[0], rgba[1], rgba[2]
	if rgba[3] != _cur_opacity:
		_cur_opacity = rgba[3]
		print '.transparency', 1 - _cur_opacity

def cvtMolecule(m):
	print '.comment', m.name
	mrad = m.DefaultBondRadius
	for a in m.atoms:
		# save color in singleton
		color = a.color
		if color is None:
			color = m.color
		_emit_color(color)
		coord = a.xformCoord()
		radius = a.radius * mrad
		print '.sphere %g %g %g %g' % (coord[0], coord[1], coord[2], radius)
	for b in m.bonds:
		# save color in singleton
		a0, a1 = b.atoms
		p0 = a0.xformCoord()
		p1 = a1.xformCoord()
		c0 = a0.color
		if c0 is None:
			c0 = m.color
		c1 = a1.color
		if c1 is None:
			c1 = m.color
		if c0 == c1:
			_emit_color(c0)
			print '.cylinder %g %g %g %g %g %g %g open' % (
				p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], mrad)
		else:
			mid = chimera.Point([p0, p1])
			_emit_color(c0)
			print '.cylinder %g %g %g %g %g %g %g open' % (
				p0[0], p0[1], p0[2], mid[0], mid[1], mid[2], mrad)
			_emit_color(c1)
			print '.cylinder %g %g %g %g %g %g %g open' % (
				p1[0], p1[1], p1[2], mid[0], mid[1], mid[2], mrad)


for m in chimera.openModels.list(modelTypes=[chimera.Molecule]):
	cvtMolecule(m)
