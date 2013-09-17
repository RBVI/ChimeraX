"""
Lighting module
---------------
Provide high-level interface to lighting

4 types of lighting are supported:
	1. Ambient only
	2. A single light will ambient fill
	3. Separate key and fill lights (without ambient)
	4. Separate key, fill, and back lights (without ambient)
"""
#
#	In the two light (key and fill) interface:
#
#		scale = brightness / ratio
#		fill_brightness = scale - ambient
#		key_brightness = 1 - scale
#
#	except the light's brightness is increased when off-center:
#
#		adjust = 2 - abs(light_direction * eye_direction)
#
#	The single light interface replaces the fill light with an
#	(omnidirectional) ambient light.
#

from chimera2.math3d import Vector
from chimera2.color import Color
import copy

# Styles
SystemDefault = "Chimera default"
UserDefault = "User default"

# Modes
AMBIENT = 'ambient'
ONE = 'single'		# key + ambient
TWO = 'two-point'	# key + fill
THREE = 'three-point'	# key + fill + back/rim
MODES = (AMBIENT, ONE, TWO, THREE)

MODE = 'mode'
BRIGHTNESS = 'brightness'
CONTRAST = 'contrast'
RATIO = 'ratio'
KEY = 'key'
FILL = 'fill'
BACK = 'back'
_DIR = 0
_COLOR = 1
_SPEC = 2

MATERIAL = 'material'
_SHARP = 0
#_COLOR = 1
_REFLECT = 2

_OPENGL_DEFAULT_AMBIENT = 0.2	# OpenGL default of 0.2 ambient light

_prefs = {}	# TODO: preferences mechanism

# serializable parameters
_default_params = {}
_params = {}

# exported expanded parameters
key_light = None	#: DirectionalLight
fill_light = None	#: DirectionalLight
back_light = None	#: DirectionalLight
ambient = 0		#: float

# Chimera 1 default material:
class Material:

	def __init__(self):
		self.ambient_diffuse = (1.0, 1.0, 1.0)
		self.specular = (0.85, 0.85, 0.85)
		self.shininess = 30.0
		self.opacity = 1.0

_default_material = Material()

class DirectionalLight:

	def __init__(self):
		self.direction = (0, 0, -1)
		self.color = (1, 1, 1)
		self.specular_scale = 1.

def mode():
	return _params[MODE]

def set_mode(mode, _do_update=True):
	if mode == _params[MODE]:
		return
	_params[MODE] = mode
	global key_light, fill_light, back_light, ambient
	if mode == AMBIENT:
		key_light = None
		fill_light = None
		back_light = None
	elif mode == ONE:
		key_light = DirectionalLight()
		key_light.direction = Vector(_params[KEY][_DIR])
		key_light.color = Color(_params[KEY][_COLOR])
		key_light.specular_scale = _params[KEY][_SPEC]
		fill_light = None
		back_light = None
	elif mode == TWO:
		key_light = DirectionalLight()
		key_light.direction = Vector(_params[KEY][_DIR])
		key_light.color = Color(_params[KEY][_COLOR])
		key_light.specular_scale = _params[KEY][_SPEC]
		fill_light = DirectionalLight()
		fill_light.direction = Vector(_params[FILL][_DIR])
		fill_light.color = Color(_params[FILL][_COLOR])
		fill_light.specular_scale = _params[FILL][_SPEC]
		back_light = None
	elif mode == THREE:
		key_light = DirectionalLight()
		key_light.direction = Vector(_params[KEY][_DIR])
		key_light.color = Color(_params[KEY][_COLOR])
		key_light.specular_scale = _params[KEY][_SPEC]
		fill_light = DirectionalLight()
		fill_light.direction = Vector(_params[FILL][_DIR])
		fill_light.color = Color(_params[FILL][_COLOR])
		fill_light.specular_scale = _params[FILL][_SPEC]
		back_light = DirectionalLight()
		back_light.direction = Vector(_params[BACK][_DIR])
		back_light.color = Color(_params[BACK][_COLOR])
		back_light.specular_scale = _params[BACK][_SPEC]
	else:
		raise ValueError('unknown lighting mode')
	if _do_update:
		_update_lights()

def _update_lights():
	global ambient
	brightness = _params[BRIGHTNESS]
	contrast = _params[CONTRAST]
	if _params[MODE] == AMBIENT:
		ambient = brightness
	elif _params[MODE] == ONE:
		ambient = (1 - contrast) * brightness
		eye_dir = Vector([0, 0, 1])
		adjust = 2 - abs(key_light.direction * eye_dir)
		key_light.diffuse_scale = (brightness - ambient) * adjust
		if key_light.diffuse_scale < 1e-6:
			# prevent from disappearing in interface
			key_light.diffuse_scale = 1e-6
	elif _params[MODE] in (TWO, THREE):
		ratio = _params[RATIO]
		maxr = maximum_ratio(contrast)
		if ratio > maxr:
			ratio = maxr
		eye_dir = Vector([0, 0, 1])
		ambient = (1 - contrast) * brightness
		scale = brightness / ratio

		# F = fill_light.diffuse_scale + ambient
		# K = key_light.diffuse_scale + ambient
		# F = brightness / ratio
		# ratio = K / F
		# With lights in eye direction:
		#   brightness = fill_light.diffuse_scale + key_light.diffuse_scale + ambient
		# (The above equation insures that the brightness doesn't
		# change appreciatively when switching between single and two
		# light modes.)
		fill_light.diffuse_scale = (brightness - ratio * ambient) / (ratio + 1)
		key_light.diffuse_scale = (brightness - ambient - fill_light.diffuse_scale)

		# adjust values are used to maintain a constant
		# brightness as the lights are moved around.
		adjust = 2 - abs(fill_light.direction * eye_dir)
		fill_light.diffuse_scale *= adjust
		if fill_light.diffuse_scale < 1e-6:
			# prevent from disappearing in interface
			fill_light.diffuse_scale = 1e-6
		adjust = 2 - abs(key_light.direction * eye_dir)
		key_light.diffuse_scale *= adjust
		if key_light.diffuse_scale < 1e-6:
			# prevent from disappearing in interface
			key_light.diffuse_scale = 1e-6

def brightness():
	return _params[BRIGHTNESS]

def set_brightness(brightness, _do_update=True):
	if brightness < 0:
		raise ValueError("brightness must be >= 0")
	if brightness == _params[BRIGHTNESS]:
		return
	_params[BRIGHTNESS] = brightness
	if _do_update:
		_update_lights()

def contrast():
	return _params[CONTRAST]

def set_contrast(contrast, _do_update=True):
	if contrast < 0 or contrast > 1:
		raise ValueError("contrast must be between 0 and 1 inclusive")
	if contrast == _params[CONTRAST]:
		return
	_params[CONTRAST] = contrast
	if _do_update:
		_update_lights()

def maximum_ratio(contrast):
	if contrast >= 1:
		return float("inf")	# larger than any reasonable value
	if contrast < 0:
		return 1
	return 1 / (1 - contrast)

def ratio():
	return _params[RATIO]

def set_ratio(ratio, clamp=False, _do_update=True):
	maxr = maximum_ratio(_params[CONTRAST])
	if ratio < 1:
		if not clamp:
			raise ValueError("ratio must be >= 1")
		ratio = 1
	if clamp and ratio > maxr:
		ratio = maxr
	if ratio == _params[RATIO]:
		return
	_params[RATIO] = ratio
	if _do_update:
		_update_lights()

def _get_light(light, ambient_okay=False):
	mode = _params[MODE]
	if light == KEY and (ambient_okay or mode != AMBIENT):
		return key_light
	if light == FILL and mode != AMBIENT:
		return fill_light
	if light == BACK and mode != AMBIENT:
		return back_light
	return None

def light_color(name):
	try:
		return _params[name][_COLOR]
	except KeyError:
		raise ValueError('unknown light: %s' % name)

def set_light_color(name, color):
	if not isinstance(color, Color):
		color = Color(color)
	try:
		_params[name][_COLOR] = color.rgb
	except KeyError:
		raise ValueError('unknown light: %s' % name)
	light = _get_light(name, ambient_okay=True)
	if light:
		light.color = color

def light_direction(name):
	try:
		return _params[name][_DIR]
	except KeyError:
		raise ValueError('unknown light: %s' % name)

def set_light_direction(name, dir, _do_update=True):
	if not isinstance(dir, Vector):
		dir = Vector(dir)
	dir.normalize()
	try:
		_params[name][_DIR] = tuple(dir)
	except KeyError:
		raise ValueError('unknown light: %s' % name)
	if _do_update:
		_update_lights()
	light = _get_light(name)
	if light:
		light.direction = dir

def light_specular_intensity(name):
	try:
		return _params[name][_SPEC]
	except KeyError:
		raise ValueError('unknown light: %s' % name)

def set_light_specular_intensity(name, i):
	try:
		_params[name][_SPEC] = i
	except KeyError:
		raise ValueError('unknown light: %s' % name)
	light = _get_light(name)
	if light:
		light.specular_scale = i

def sharpness():
	return _params[MATERIAL][_SHARP]

def set_sharpness(sharpness):
	if sharpness == _params[MATERIAL][_SHARP]:
		return
	if sharpness < 0 or sharpness > 128:
		raise ValueError("sharpness must be between 0 and 128 inclusive")
	_params[MATERIAL][_SHARP] = sharpness
	_update_materials()

def reflectivity():
	return _params[MATERIAL][_REFLECT]

def set_reflectivity(reflectivity):
	if reflectivity == _params[MATERIAL][_REFLECT]:
		return
	if reflectivity < 0:
		raise ValueError('reflectivity must be non-negative')
	_params[MATERIAL][_REFLECT] = reflectivity
	_update_materials()

def shiny_color():
	return Color(_params[MATERIAL][_COLOR])

#def set_shiny_color(color):
#	_params[MATERIAL][_COLOR] = color.rgba()[:3]
#	_update_materials()

def material():
	return _params[MATERIAL]

def set_material(sharpness, specular_color, reflectivity):
	s, sc, r = _params[MATERIAL]
	if sharpness == s and specular_color == sc and reflectivity == r:
		return
	# repeat error checking from set_sharpness and set_reflectivity
	if sharpness < 0 or sharpness > 128:
		raise ValueError("sharpness must be between 0 and 128 inclusive")
	if reflectivity < 0:
		raise ValueError('reflectivity must be non-negative')
	_params[MATERIAL] = [sharpness, specular_color, reflectivity]
	_update_materials()

def _update_materials():
	sharpness, specular_color, reflectivity = _params[MATERIAL]
	mat = _default_material
	mat.shininess = sharpness
	mat.specular = tuple(x * reflectivity for x in specular_color)
	# TODO: broadcast change

def _init():
	# create defaults after graphics is initialized
	global _default_params, _params
	mat = _default_material
	rgb, reflect = _normalized_color(mat.specular)
	from math import sin, cos, radians
	a15 = radians(15)
	fill_direction = Vector([sin(a15), sin(a15), cos(a15)])
	a45 = radians(45)
	key_direction = Vector([-sin(a45 / 2), sin(a45), cos(a45)])
	back_direction = Vector([sin(a45 / 2), sin(a45), -cos(a45)])
	brightness = 1.16
	_default_params = {
		MODE: TWO,
		BRIGHTNESS: brightness,
		#CONTRAST: 1 - _OPENGL_DEFAULT_AMBIENT / brightness,
		CONTRAST: 0.83,
		RATIO: 1.25,
		KEY: [
			tuple(key_direction),
			(1, 1, 1),		# color
			1.0			# specular
		],
		FILL: [
			tuple(fill_direction),
			(1, 1, 1),		# color
			0.0			# specular
		],
		BACK: [
			tuple(back_direction),
			(1, 1, 1),		# color
			0.0			# specular
		],
		MATERIAL: [mat.shininess, rgb, reflect],
	}
	_params = copy.deepcopy(_default_params)
	_params[MODE] = 'uninitialized'
	if UserDefault in _prefs:
		_set_from_params(_prefs[UserDefault])
	else:
		_set_from_params(_default_params)

def _calculate_brightness_ratio(key, fill, ambient):
	# compute brightness and ratio from light settings
	eye_dir = Vector([0, 0, 1])
	fadjust = 2 - abs(fill.direction * eye_dir)
	fd = fill.diffuse_scale / fadjust
	kadjust = 2 - abs(key.direction * eye_dir)
	kd = key.diffuse_scale / kadjust
	brightness = fd + kd + ambient
	if brightness < 0:
		brightness = 0
	ratio = (kd + ambient) / (fd + ambient)
	if ratio < 1:
		ratio = 1
	maxr = maximum_ratio(1 - ambient)
	if ratio > maxr:
		ratio = maxr
	return brightness, ratio

def _set_from_params(p):
	set_brightness(p[BRIGHTNESS], _do_update=False)
	set_contrast(p[CONTRAST], _do_update=False)
	set_ratio(p[RATIO], clamp=True, _do_update=False)
	_set_light_params(KEY, p[KEY], _do_update=False)
	_set_light_params(FILL, p[FILL], _do_update=False)
	if BACK in p:
		_set_light_params(BACK, p[BACK], _do_update=False)
	set_material(*p[MATERIAL])
	set_mode(p[MODE], _do_update=False)
	_update_lights()

def _set_light_params(name, values, _do_update=True):
	set_light_direction(name, values[_DIR], _do_update=_do_update)
	set_light_color(name, values[_COLOR])
	set_light_specular_intensity(name, values[_SPEC])

def _normalized_color(rgb, brightness=None):
	# Given rgb may have components > 1.  In this case scale to
	# produce maximum color component of 1.
	b = max(rgb)
	if brightness is None:
		if b <= 1:
			return rgb, 1.0
	elif b < brightness:
		# Only use the given brightness if it is high enough to
		# yield a specular color with components <= 1
		b = brightness
	return tuple([ c / b for c in rgb ]), b

def restore(style):
	if style == SystemDefault:
		_set_from_params(_default_params)
	elif style in _prefs:
		_set_from_params(_prefs[style])
	else:
		raise ValueError("unknown lighting style: %s" % style)

def save(style):
	import copy
	_prefs[style] = copy.deepcopy(_params)
	#TODO: _prefs.saveToFile()

def delete(style):
	del _prefs[style]
	#TODO: _prefs.saveToFile()

_init()
