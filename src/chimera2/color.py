
Builtin_Colors = {
	# subset of X11's rgb.txt
	"aquamarine":	(127, 255, 212),
	"black":	(0, 0, 0),
	"blue":		(0, 0, 255),
	"brown":	(165, 42, 42),
	"chartreuse":	(127, 255, 0),
	"coral":	(255, 127, 80),
	"cornflower blue":	(100, 149, 237),
	"cyan":		(0, 255, 255),
	"dark cyan":	(0, 139, 139),
	"dark gray":	(169, 169, 169),
	"dark grey":	(169, 169, 169),
	"dark green":	(0, 100, 0),
	"dark khaki":	(189, 183, 107),
	"dark magenta":	(139, 0, 139),
	"dark olive green":	(85, 107, 47),
	"dark red":	(139, 0, 0),
	"dark slate blue":	(72, 61, 139),
	"dark slate gray":	(47, 79, 79),
	"dark slate grey":	(47, 79, 79),
	"deep pink":	(255, 20, 147),
	"deep sky blue":	(0, 191, 255),
	"dim gray":	(105, 105, 105),
	"dim grey":	(105, 105, 105),
	"dodger blue":	(30, 144, 255),
	"firebrick":	(178, 34, 34),
	"forest green":	(34, 139, 34),
	"gold":		(255, 215, 0),
	"goldenrod":	(218, 165, 32),
	"gray":		(190, 190, 190),
	"grey":		(190, 190, 190),
	"green":	(0, 255, 0),
	"hot pink":	(255, 105, 180),
	"khaki":	(240, 230, 140),
	"light blue":	(173, 216, 230),
	"light gray":	(211, 211, 211),
	"light grey":	(211, 211, 211),
	"light green":	(144, 238, 144),
	"light sea green":	(32, 178, 170),
	"lime green":	(50, 205, 50),
	"magenta":	(255, 0, 255),
	"medium blue":	(50, 50, 205),
	"medium purple":	(147, 112, 219),
	"navy blue":	(0, 0, 128),
	"olive drab":	(107, 142, 35),
	"orange red":	(255, 69, 0),
	"orange":	(255, 127, 0),
	"orchid":	(218, 112, 214),
	"pink":		(255, 192, 203),
	"plum":		(221, 160, 221),
	"purple":	(160, 32, 240),
	"red":		(255, 0, 0),
	"rosy brown":	(188, 143, 143),
	"salmon":	(250, 128, 114),
	"sandy brown":	(244, 164, 96),
	"sea green":	(46, 139, 87),
	"sienna":	(160, 82, 45),
	"sky blue":	(135, 206, 235),
	"slate gray":	(112, 128, 144),
	"slate grey":	(112, 128, 144),
	"spring green":	(0, 255, 127),
	"steel blue":	(70, 130, 180),
	"tan":		(210, 180, 140),
	"turquoise":	(64, 224, 208),
	"violet red":	(208, 32, 144),
	"white":	(255, 255, 255),
	"yellow":	(255, 255, 0),
}

class Color:

	def __init__(self, rgb=None):
		if isinstance(rgb, (tuple, list)):
			if len(rgb) != 3:
				raise ValueError("expected 3 floats")
			self.rgb = list(rgb)
		elif isinstance(rgb, Color):
			self.rgb = rgb.rgb
		else:
			raise ValueError("Not a color")

from chimera2 import cli
class Color_arg(cli.Annotation):
	multiword = True
	name = 'a color'

	@staticmethod
	def parse(text):
		if text[0] != '#':
			text = text.casefold()
			try:
				color = Builtin_Colors[text]
			except KeyError:
				raise ValueError("Invalid color name")
			return Color([x / 255.0 for x in color])
		# Hex: DDD, DDDDDD, or DDDDDDDDDDDD
		try:
			int(text[1:], 16)
		except ValueError:
			raise ValueError("expected hex digits in color constant")
		if len(text) == 3:
			digits = (x for x in text[1:])
			return Color([int(x, 16) / 15.0 for x in digits])
		elif len(text) == 6:
			digits = (text[x:x+1] for x in range(1, 7, 2))
			return Color([int(x, 16) / 255.0 for x in digits])
		elif len(text) == 12:
			digits = (text[x:x+1] for x in range(1, 13, 4))
			return Color([int(x, 16) / 65535.0 for x in digits])
		else:
			raise ValueError("Color constant should have 3, 6, or 12 hex digits")

	@staticmethod
	def completions(text):
		# TODO: add user defined colors
		text = text.casefold()
		names = [n for n in Builtin_Colors if n.startswith(text)]
		return names
