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
#	except the light's brightness is increased when off-center
#	to keep the apparent brightness constant:
#
#		adjust = 2 - abs(light_direction * eye_direction)
#
#	The single light interface replaces the fill light with an
#	(omnidirectional) ambient light.
#

from chimera2.math3d import Vector
from chimera2.color import Color
from ..trackchanges import track

# Styles
SystemDefault = "Chimera default"
UserDefault = "User default"

# Modes
AMBIENT = 'ambient'
ONE = 'single'		# key + ambient
TWO = 'two-point'	# key + fill
THREE = 'three-point'	# key + fill + back/rim

VERSION = 'version'
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

_default_lighting = None	# system default
lighting = None			# current lighting parameters

def maximum_ratio(contrast):
	if contrast >= 1:
		return float("inf")	# larger than any reasonable value
	if contrast < 0:
		return 1
	return 1 / (1 - contrast)

# Chimera default material:
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

@track.register_data_type
class Lighting:

	# lighting mode: make global constants available
	AMBIENT = AMBIENT 
	ONE = ONE
	TWO = TWO 
	THREE = THREE 
	MODES = (AMBIENT, ONE, TWO, THREE)
	LIGHT_NAMES = (KEY, FILL, BACK)

	# modified reasons
	LIGHTING_CHANGE = 'lighting change'
	COLOR_CHANGE = 'light color change'
	SPECULAR_CHANGE = 'light specular change'
	MATERIAL_CHANGE = 'material change'

	def __init__(self):
		# outputs:
		self.key_light = None
		self.fill_light = None
		self.back_light = None
		self.ambient = 0
		# private:
		mat = _default_material
		rgb, reflect = _normalized_color(mat.specular)
		from math import sin, cos, radians
		a15 = radians(15)
		fill_direction = Vector([sin(a15), sin(a15), cos(a15)])
		a45 = radians(45)
		key_direction = Vector([-sin(a45 / 2), sin(a45), cos(a45)])
		back_direction = Vector([sin(a45 / 2), sin(a45), -cos(a45)])
		brightness = 1.16
		self._params = {
			VERSION: 1,
			MODE: "uninitialized",	# temporary, see below
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
		self.mode = TWO		# finally initialize mode
		track.created(Lighting, [self])

	def as_POD(self):
		"""Return Plain-Old-Data version of lighting information"""
		return self._params

	def from_POD(self, pod):
		try:
			assert(pod[VERSION] == 1)
		except (KeyError, AssertionError):
			raise ValueError("unsupported version of lighting data")
		import copy
		self._params = copy.deepcopy(pod)
		mode = self._params[MODE]
		self._params[MODE] = "uninitalized"
		self.mode = mode
		self._update_lights()

	@property
	def mode(self):
		return self._params[MODE]

	@mode.setter
	def mode(self, mode):
		# set lighting mode and configure lights
		if mode == self._params[MODE]:
			return
		self._params[MODE] = mode
		if mode == AMBIENT:
			self.key_light = None
			self.fill_light = None
			self.back_light = None
		elif mode == ONE:
			kl = self.key_light = DirectionalLight()
			kl.direction = Vector(self._params[KEY][_DIR])
			kl.color = Color(self._params[KEY][_COLOR])
			kl.specular_scale = self._params[KEY][_SPEC]
			self.fill_light = None
			self.back_light = None
		elif mode == TWO:
			kl = self.key_light = DirectionalLight()
			kl.direction = Vector(self._params[KEY][_DIR])
			kl.color = Color(self._params[KEY][_COLOR])
			kl.specular_scale = self._params[KEY][_SPEC]
			fl = self.fill_light = DirectionalLight()
			fl.direction = Vector(self._params[FILL][_DIR])
			fl.color = Color(self._params[FILL][_COLOR])
			fl.specular_scale = self._params[FILL][_SPEC]
			self.back_light = None
		elif mode == THREE:
			kl = self.key_light = DirectionalLight()
			kl.direction = Vector(self._params[KEY][_DIR])
			kl.color = Color(self._params[KEY][_COLOR])
			kl.specular_scale = self._params[KEY][_SPEC]
			fl = self.fill_light = DirectionalLight()
			fl.direction = Vector(self._params[FILL][_DIR])
			fl.color = Color(self._params[FILL][_COLOR])
			fl.specular_scale = self._params[FILL][_SPEC]
			bl = self.back_light = DirectionalLight()
			bl.direction = Vector(self._params[BACK][_DIR])
			bl.color = Color(self._params[BACK][_COLOR])
			bl.specular_scale = self._params[BACK][_SPEC]
		else:
			raise ValueError('unknown lighting mode')
		self._update_lights()

	@property
	def brightness(self):
		return self._params[BRIGHTNESS]

	@brightness.setter
	def brightness(self, brightness):
		if brightness < 0:
			raise ValueError("brightness must be >= 0")
		if brightness == self._params[BRIGHTNESS]:
			return
		self._params[BRIGHTNESS] = brightness
		self._update_lights()

	@property
	def contrast(self):
		return self._params[CONTRAST]

	@contrast.setter
	def contrast(self, contrast):
		if contrast < 0 or contrast > 1:
			raise ValueError("contrast must be between 0 and 1 inclusive")
		if contrast == self._params[CONTRAST]:
			return
		self._params[CONTRAST] = contrast
		self._update_lights()

	@property
	def ratio(self):
		return self._params[RATIO]

	@ratio.setter
	def ratio(self, ratio, clamp=False):
		maxr = maximum_ratio(self._params[CONTRAST])
		if ratio < 1:
			if not clamp:
				raise ValueError("ratio must be >= 1")
			ratio = 1
		if clamp and ratio > maxr:
			ratio = maxr
		if ratio == self._params[RATIO]:
			return
		self._params[RATIO] = ratio
		self._update_lights()

	def _update_lights(self):
		brightness = self._params[BRIGHTNESS]
		contrast = self._params[CONTRAST]
		if self._params[MODE] == AMBIENT:
			self.ambient = brightness
		elif self._params[MODE] == ONE:
			self.ambient = (1 - contrast) * brightness
			eye_dir = Vector([0, 0, 1])
			kl = self.key_light
			adjust = 2 - abs(kl.direction * eye_dir)
			kl.diffuse_scale = (brightness - self.ambient) * adjust
			if kl.diffuse_scale < 1e-6:
				# prevent from disappearing in interface
				kl.diffuse_scale = 1e-6
		elif self._params[MODE] in (TWO, THREE):
			ratio = self._params[RATIO]
			maxr = maximum_ratio(contrast)
			if ratio > maxr:
				ratio = maxr
			eye_dir = Vector([0, 0, 1])
			self.ambient = (1 - contrast) * brightness
			#scale = brightness / ratio
			kl = self.key_light
			fl = self.fill_light

			# F = fill_light.diffuse_scale + ambient
			# K = key_light.diffuse_scale + ambient
			# F = brightness / ratio
			# ratio = K / F
			# With lights in eye direction:
			#   brightness = fill_light.diffuse_scale + key_light.diffuse_scale + ambient
			# (The above equation insures that the brightness
			# doesn't change appreciatively when switching between
			# single and two light modes.)
			fl.diffuse_scale = (brightness - ratio * self.ambient) / (ratio + 1)
			kl.diffuse_scale = (brightness - self.ambient - fl.diffuse_scale)

			# adjust values are used to maintain a constant
			# brightness as the lights are moved around.
			adjust = 2 - abs(fl.direction * eye_dir)
			fl.diffuse_scale *= adjust
			if fl.diffuse_scale < 1e-6:
				# prevent from disappearing in interface
				fl.diffuse_scale = 1e-6
			adjust = 2 - abs(kl.direction * eye_dir)
			kl.diffuse_scale *= adjust
			if kl.diffuse_scale < 1e-6:
				# prevent from disappearing in interface
				kl.diffuse_scale = 1e-6
		track.modified(Lighting, [self], self.LIGHTING_CHANGE)

	def _get_light(self, light, ambient_okay=False):
		mode = self._params[MODE]
		if light == KEY and (ambient_okay or mode != AMBIENT):
			return self.key_light
		if light == FILL and mode != AMBIENT:
			return self.fill_light
		if light == BACK and mode != AMBIENT:
			return self.back_light
		return None

	def light_color(self, name):
		try:
			return self._params[name][_COLOR]
		except KeyError:
			raise ValueError('unknown light: %s' % name)

	def set_light_color(self, name, color):
		if not isinstance(color, Color):
			color = Color(color)
		try:
			self._params[name][_COLOR] = color.rgb
		except KeyError:
			raise ValueError('unknown light: %s' % name)
		light = self._get_light(name, ambient_okay=True)
		if light:
			light.color = color
			track.modified(Lighting, [self], self.COLOR_CHANGE)

	def light_direction(self, name):
		try:
			return self._params[name][_DIR]
		except KeyError:
			raise ValueError('unknown light: %s' % name)

	def set_light_direction(self, name, dir):
		if not isinstance(dir, Vector):
			dir = Vector(dir)
		dir.normalize()
		try:
			self._params[name][_DIR] = tuple(dir)
		except KeyError:
			raise ValueError('unknown light: %s' % name)
		self._update_lights()
		light = self._get_light(name)
		if light:
			light.direction = dir
			self._update_lights()

	def light_specular_intensity(self, name):
		try:
			return self._params[name][_SPEC]
		except KeyError:
			raise ValueError('unknown light: %s' % name)

	def set_light_specular_intensity(self, name, i):
		try:
			self._params[name][_SPEC] = i
		except KeyError:
			raise ValueError('unknown light: %s' % name)
		light = self._get_light(name)
		if light:
			light.specular_scale = i
			track.modified(Lighting, [self], self.SPECULAR_CHANGE)

	@property
	def sharpness(self):
		return self._params[MATERIAL][_SHARP]

	@sharpness.setter
	def set_sharpness(self, sharpness):
		if sharpness == self._params[MATERIAL][_SHARP]:
			return
		if sharpness < 0 or sharpness > 128:
			raise ValueError("sharpness must be between 0 and 128 inclusive")
		self._params[MATERIAL][_SHARP] = sharpness
		self._update_materials()

	@property
	def reflectivity(self):
		return self._params[MATERIAL][_REFLECT]

	@reflectivity.setter
	def reflectivity(self, reflectivity):
		if reflectivity == self._params[MATERIAL][_REFLECT]:
			return
		if reflectivity < 0:
			raise ValueError('reflectivity must be non-negative')
		self._params[MATERIAL][_REFLECT] = reflectivity
		self._update_materials()

	@property
	def shiny_color(self):
		return Color(self._params[MATERIAL][_COLOR])

	#@shiny_color.setter
	#def shiny_color(self, color):
	#	self._params[MATERIAL][_COLOR] = color.rgba()[:3]
	#	self._update_materials()

	@property
	def material(self):
		return self._params[MATERIAL]

	@material.setter
	def material(self, sharpness, specular_color, reflectivity):
		s, sc, r = self._params[MATERIAL]
		if sharpness == s and specular_color == sc and reflectivity == r:
			return
		# repeat error checking from set_sharpness and set_reflectivity
		if sharpness < 0 or sharpness > 128:
			raise ValueError("sharpness must be between 0 and 128 inclusive")
		if reflectivity < 0:
			raise ValueError('reflectivity must be non-negative')
		self._params[MATERIAL] = [sharpness, specular_color, reflectivity]
		self._update_materials()

	def _update_materials(self):
		sharpness, specular_color, reflectivity = self._params[MATERIAL]
		mat = _default_material
		mat.shininess = sharpness
		mat.specular = tuple(x * reflectivity for x in specular_color)
		track.modified(Lighting, [self], self.MATERIAL_CHANGE)

	def restore(self, style):
		if style == SystemDefault:
			self.from_POD(_default_lighting.as_POD())
		elif style in _prefs:
			self.from_POD(_prefs[style])
		else:
			raise ValueError("unknown lighting style: %s" % style)

	def save(self, style):
		import copy
		_prefs[style] = copy.deepcopy(self.as_POD())
		#TODO: _prefs.saveToFile()

	def delete(self, style):
		del _prefs[style]
		#TODO: _prefs.saveToFile()

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


def _init():
	global _default_lighting, lighting
	_default_lighting = Lighting()
	lighting = Lighting()
	if UserDefault in _prefs:
		lighting.from_POD(_prefs[UserDefault])
_init()
