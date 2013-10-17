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

from chimera2 import lighting
from chimera2.color import Color
from chimera2.cmds import UserError, Optional, register as register_cmd

# CmdInfo is initialized at the end of this file

class _Mode:

	def __call__(self, text):
		for m in lighting.MODES:
			if m.startswith(text):
				return m
		raise UserError("invalid lighting mode")

	def completions(self, text):
		return [m for m in lighting.MODES if m.startswith(text)]
Mode = _Mode()

def mode(mode: Optional(Mode)=None):
	if mode is None:
		return "Current lighting mode is %s" % lighting.mode()
	try:
		return lighting.set_mode(mode)
	except ValueError as e:
		raise UserError(e)

def brightness(brightness: Optional(float)=None):
	if brightness is None:
		return "Current brightness is %s" % lighting.brightness()
	try:
		return lighting.set_brightness(brightness)
	except ValueError as e:
		raise UserError(e)

def contrast(contrast: Optional(float)=None):
	if contrast is not None:
		try:
			return lighting.set_contrast(contrast)
		except ValueError as e:
			raise UserError(e)
	mode = lighting.mode()
	if mode == lighting.AMBIENT:
		return "Not applicable"
	return "Current contrast is %s" % lighting.contrast()

def ratio(ratio: Optional(float)=None):
	if ratio is not None:
		try:
			return lighting.set_ratio(ratio)
		except ValueError as e:
			raise UserError(e)
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

def light_color(light, color: Optional(Color)=None):
	if color is None:
		color = lighting.light_color(light)
		return "%s light color is (%g, %g, %g)" % ((light,) + color)
	try:
		return lighting.set_light_color(light, color)
	except ValueError as e:
		raise UserError(e)

def light_direction(light, x: Optional(float)=None, y: Optional(float)=None, z: Optional(float)=None):
	if x is None and y is None and z is None:
		dir = lighting.light_direction(light)
		return "%s light direction is (%g, %g, %g)" % ((light,) + dir)
	if x is None or y is None or z is None:
		raise ValueError("none or all of x, y, z are required")
	try:
		return lighting.set_light_direction(light, (x, y, z))
	except ValueError as e:
		raise UserError(e)

def light_specular_intensity(light, intensity: Optional(float)=None):
	if intensity is None:
		i = lighting.light_specular_intensity(light)
		return "%s light specular intensity is %s" % (light, i)
	if not (0 <= intensity <= 1):
		raise UserError("expecting a number between 0 and 1 inclusive")
	try:
		return lighting.set_light_specular_intensity(light, i)
	except ValueError as e:
		raise UserError(e)

def sharpness(sharpness: Optional(float)=None):
	if sharpness is None:
		s = lighting.sharpness()
		return "material sharpness is %s" % s
	try:
		return lighting.set_sharpness(sharpness)
	except ValueError as e:
		raise UserError(e)

def reflectivity(reflectivity: Optional(float)=None):
	if reflectivity is None:
		r = lighting.reflectivity()
		return "material reflectivity is %s" % r
	try:
		return lighting.set_reflectivity(reflectivity)
	except ValueError as e:
		raise UserError(e)

def restore(style):
	return lighting.restore(style)

def save(style):
	return lighting.save(style)

def delete(style):
	return lighting.delete(style)

def register():
	register_cmd('lighting mode', mode)
	register_cmd('lighting brightness', brightness)
	register_cmd('lighting contrast', contrast)
	register_cmd('lighting ratio', ratio)
	register_cmd('lighting sharpness', sharpness)
	register_cmd('lighting reflectivity', reflectivity)
	register_cmd('lighting restore', restore)
	register_cmd('lighting save', save)
	register_cmd('lighting delete', delete)

	LIGHT_NAMES = (lighting.KEY, lighting.FILL, lighting.BACK)
	for light in LIGHT_NAMES:
		def set_color(color: Optional(Color)=None, _light=light):
			return light_color(_light, color)
		register_cmd("lighting %s color" % light, set_color)
		def set_direction(x: Optional(float)=None, y: Optional(float)=None, z: Optional(float)=None, _light=light):
			return light_direction(_light, x, y, z)
		register_cmd("lighting %s direction" % light, set_direction)
		def set_intensity(intensity: Optional(float)=None, _light=light):
			return light_specular_intensity(_light, intensity)
		register_cmd("lighting %s specular_intensity" % light, set_intensity)
