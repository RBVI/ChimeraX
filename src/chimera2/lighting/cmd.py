# chimera lighting command
#
#	lighting mode [mode]
#		If no argument, show the current mode, otherwise, set it.
#		mode can be one of ambient, single, two-point, three-point
#	lighting brightness [brightness]
#		If no argument, show the current brightness, otherwise, set it.
#		brightness must be > 0
#	lighting contrast [contrast]
#		If no argument, show the current contrast, otherwise, set it.
#		contrast must be >= 0 and <= 1
#	lighting ratio [ratio]
#		If no argument, show the current ratio, otherwise, set it.
#		ratio must be >= 1
#	lighting key|fill|back color [color_spec]
#	lighting key|fill|back direction [x y z]
#	lighting key|fill|back specular_intensity [intensity]
#		change specific light properties
#	lighting restore name
#		restore named lighting style
#	lighting save name
#		save current settings in named lighting style
#	lighting delete name
#		delete named lighting style
#	lighting sharpness [sharpness]
#	lighting reflectivity [reflectivity]

from chimera2 import UserError, lighting
from chimera2.color import Color

# CmdInfo is initialized at the end of this file

def mode(name=None):
	if name is None:
		return "Current lighting mode is %s" % lighting.mode()
	# TODO: remove autocompletion code belong when annotation types
	#   can give autocompletion hints
	for m in lighting.MODES:
		if m.startswith(name):
			name = m
	lighting.set_mode(name)

def brightness(brightness: float=None):
	if brightness is None:
		return "Current brightness is %s" % lighting.brightness()
	try:
		lighting.set_brightness(brightness)
	except ValueError as e:
		raise UserError(str(e))

def contrast(contrast: float=None):
	if contrast is not None:
		try:
			lighting.set_contrast(contrast)
			return
		except ValueError as e:
			raise UserError(str(e))
	mode = lighting.mode()
	if mode == lighting.AMBIENT:
		return "Not applicable"
	return "Current contrast is %s" % lighting.contrast()

def ratio(ratio: float=None):
	if ratio is not None:
		try:
			lighting.set_ratio(ratio)
			return
		except ValueError as e:
			raise UserError(str(e))
	mode = lighting.mode()
	if mode in (lighting.AMBIENT, lighting.ONE):
		return "Not applicable"
	maxr = lighting.maximum_ratio(lighting.contrast())
	if mode == lighting.ONE:
		msg = "Effective key-fill ratio is %s" % maxr
	else:
		ratio = lighting.ratio()
		msg = "Current key-fill ratio is %s" % ratio
		if ratio > maxr:
			msg += " (limited to %s)" % maxr
	return msg

def light_color(light, color: Color=None):
	if color is None:
		color = lighting.light_color(light)
		return "%s light color is (%g, %g, %g)" % ((light,) + color)
	lighting.set_light_color(light, color)

def light_direction(light, x: float=None, y: float=None, z: float=None):
	if x is None and y is None and z is None:
		dir = lighting.light_direction(light)
		return "%s light direction is (%g, %g, %g)" % ((light,) + dir)
	if x is None or y is None or z is None:
		raise UserError("none or all of x, y, z are required")
	try:
		lighting.set_light_direction(light, (x, y, z))
	except ValueError as e:
		raise UserError(str(e))

def light_specular_intensity(light, intensity: float=None):
	if intensity is None:
		i = lighting.light_specular_intensity(light)
		return "%s light specular intensity is %s" % (light, i)
	if not (0 <= intensity <= 1):
		raise UserError("expecting a number between 0 and 1 inclusive")
	lighting.set_light_specular_intensity(light, i)

def sharpness(sharpness: float=None):
	if sharpness is None:
		s = lighting.sharpness()
		return "material sharpness is %s" % s
	try:
		lighting.set_sharpness(sharpness)
	except ValueError as e:
		raise UserError(str(e))

def reflectivity(reflectivity: float=None):
	if reflectivity is None:
		r = lighting.reflectivity()
		return "material reflectivity is %s" % r
	try:
		lighting.set_reflectivity(reflectivity)
	except ValueError as e:
		raise UserError(str(e))

def restore(style):
	lighting.restore(style)

def save(style):
	lighting.save(style)

def delete(style):
	lighting.delete(style)

from chimera2 import cmds
cmds.register('lighting mode', mode)
cmds.register('lighting brightness', brightness)
cmds.register('lighting contrast', contrast)
cmds.register('lighting ratio', ratio)
cmds.register('lighting sharpness', sharpness)
cmds.register('lighting reflectivity', reflectivity)
cmds.register('lighting restore', restore)
cmds.register('lighting save', save)
cmds.register('lighting delete', delete)

LIGHT_NAMES = (lighting.KEY, lighting.FILL, lighting.BACK)
for light in LIGHT_NAMES:
	def set_color(color: Color=None, _light=light):
		return light_color(_light, color)
	cmds.register("lighting %s color" % light, set_color)
	def set_direction(x: float=None, y: float=None, z: float=None, _light=light):
		return light_direction(_light, x, y, z)
	cmds.register("lighting %s direction" % light, set_direction)
	def set_intensity(intensity: float=None, _light=light):
		return light_specular_intensity(_light, intensity)
	cmds.register("lighting %s specular_intensity" % light, set_intensity)
del light, set_color, set_direction, set_intensity
