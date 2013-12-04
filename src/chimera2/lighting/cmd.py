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

from chimera2.lighting import lighting, maximum_ratio
from chimera2.color import Color_arg
from chimera2.cli import UserError, CmdInfo, Enum_of, string_arg, float_arg, float3_arg, register as register_cmd

# CmdInfo is initialized at the end of this file

Mode_arg = Enum_of(lighting.MODES)

mode_info = CmdInfo(optional=[('mode', Mode_arg)])
def mode(mode=None):
	if mode is None:
		return "Current lighting mode is %s" % lighting.mode
	try:
		lighting.mode = mode
	except ValueError as e:
		raise UserError(e)

brightness_info = CmdInfo(optional=[('brightness', float_arg)])
def brightness(brightness=None):
	if brightness is None:
		return "Current brightness is %s" % lighting.brightness
	try:
		lighting.brightness = brightness
	except ValueError as e:
		raise UserError(e)

contrast_info = CmdInfo(optional=[('contrast', float_arg)])
def contrast(contrast=None):
	if contrast is None:
		mode = lighting.mode
		if mode == lighting.AMBIENT:
			return "Not applicable"
		return "Current contrast is %s" % lighting.contrast
	try:
		lighting.contrast = contrast
	except ValueError as e:
		raise UserError(e)

ratio_info = CmdInfo(optional=[('ratio', float_arg)])
def ratio(ratio=None):
	if ratio is not None:
		try:
			lighting.ratio = ratio
			return
		except ValueError as e:
			raise UserError(e)
	mode = lighting.mode
	if mode in (lighting.AMBIENT, lighting.ONE):
		return "Not applicable"
	maxr = maximum_ratio(lighting.contrast)
	if mode == lighting.ONE:
		msg = "Effective key-fill ratio is %s" % maxr
	else:
		ratio = lighting.ratio
		msg = "Current key-fill ratio is %s" % ratio
		if ratio > maxr:
			msg += " (limited to %s)" % maxr
	return msg

color_info = CmdInfo(optional=[('color', Color_arg)])
def light_color(light, color=None):
	if color is None:
		color = lighting.light_color(light)
		return "%s light color is (%g, %g, %g)" % ((light,) + color)
	try:
		return lighting.set_light_color(light, color)
	except ValueError as e:
		raise UserError(e)

direction_info = CmdInfo(optional=[('xyz', float3_arg)])
def light_direction(light, xyz=None):
	if xyz is None:
		dir = lighting.light_direction(light)
		return "%s light direction is (%g, %g, %g)" % ((light,) + dir)
	try:
		return lighting.set_light_direction(light, xyz)
	except ValueError as e:
		raise UserError(e)

intensity_info = CmdInfo(optional=[('instensity', float_arg)])
def light_specular_intensity(light, intensity=None):
	if intensity is None:
		i = lighting.light_specular_intensity(light)
		return "%s light specular intensity is %s" % (light, i)
	if not (0 <= intensity <= 1):
		raise UserError("expecting a number between 0 and 1 inclusive")
	try:
		return lighting.set_light_specular_intensity(light, i)
	except ValueError as e:
		raise UserError(e)

sharpness_info = CmdInfo(optional=[('sharpness', float_arg)])
def sharpness(sharpness=None):
	if sharpness is None:
		s = lighting.sharpness()
		return "material sharpness is %s" % s
	try:
		return lighting.set_sharpness(sharpness)
	except ValueError as e:
		raise UserError(e)

reflectivity_info = CmdInfo(optional=[('reflectivity', float_arg)])
def reflectivity(reflectivity=None):
	if reflectivity is None:
		r = lighting.reflectivity
		return "material reflectivity is %s" % r
	try:
		lighting.reflectivity = reflectivity
	except ValueError as e:
		raise UserError(e)

restore_info = CmdInfo(required=[('style', string_arg)])
def restore(style):
	return lighting.restore(style)

save_info = restore_info.copy()
def save(style):
	return lighting.save(style)

delete_info = restore_info.copy()
def delete(style):
	return lighting.delete(style)

def register():
	register_cmd('lighting mode', mode_info, mode)
	register_cmd('lighting brightness', brightness_info, brightness)
	register_cmd('lighting contrast', contrast_info, contrast)
	register_cmd('lighting ratio', ratio_info, ratio)
	register_cmd('lighting sharpness', sharpness_info, sharpness)
	register_cmd('lighting reflectivity', reflectivity_info, reflectivity)
	register_cmd('lighting restore', restore_info, restore)
	register_cmd('lighting save', save_info, save)
	register_cmd('lighting delete', delete_info, delete)

	for light in lighting.LIGHT_NAMES:
		def set_color(color=None, _light=light):
			return light_color(_light, color)
		register_cmd("lighting %s color" % light, color_info.copy(),
								set_color)
		def set_direction(xyz=None, _light=light):
			return light_direction(_light, xyz)
		register_cmd("lighting %s direction" % light,
					direction_info.copy(), set_direction)
		def set_intensity(intensity=None, _light=light):
			return light_specular_intensity(_light, intensity)
		register_cmd("lighting %s specular_intensity" % light,
					intensity_info.copy(), set_intensity)
