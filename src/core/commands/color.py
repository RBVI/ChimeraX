# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
color: basic color support
===========================

A :py:class:`Color` class that hold a :py:class:`numpy.array` of
four 32-bit floats.

Also, a :py:`ColorArg` :py:mod:`cli` :py:class:`Annotation` for
command line parsing.

CSS3 colors are supported with the addition of the gray() specification
from the CSS4 draft and the CSS4 color names.
"""
import re
from sortedcontainers import SortedDict
from . import cli
from ..state import State, RestoreError

# CSS4 colors
_BuiltinColors = SortedDict({
    'aliceblue': (240, 248, 255, 255),
    'antiquewhite': (250, 235, 215, 255),
    'aqua': (0, 255, 255, 255),
    'aquamarine': (127, 255, 212, 255),
    'azure': (240, 255, 255, 255),
    'beige': (245, 245, 220, 255),
    'bisque': (255, 228, 196, 255),
    'black': (0, 0, 0, 255),
    'blanchedalmond': (255, 235, 205, 255),
    'blue': (0, 0, 255, 255),
    'blueviolet': (138, 43, 226, 255),
    'brown': (165, 42, 42, 255),
    'burlywood': (222, 184, 135, 255),
    'cadetblue': (95, 158, 160, 255),
    'chartreuse': (127, 255, 0, 255),
    'chocolate': (210, 105, 30, 255),
    'coral': (255, 127, 80, 255),
    'cornflowerblue': (100, 149, 237, 255),
    'cornsilk': (255, 248, 220, 255),
    'crimson': (220, 20, 60, 255),
    'cyan': (0, 255, 255, 255),
    'darkblue': (0, 0, 139, 255),
    'darkcyan': (0, 139, 139, 255),
    'darkgoldenrod': (184, 134, 11, 255),
    'darkgray': (169, 169, 169, 255),
    'darkgreen': (0, 100, 0, 255),
    'darkgrey': (169, 169, 169, 255),
    'darkkhaki': (189, 183, 107, 255),
    'darkmagenta': (139, 0, 139, 255),
    'darkolivegreen': (85, 107, 47, 255),
    'darkorange': (255, 140, 0, 255),
    'darkorchid': (153, 50, 204, 255),
    'darkred': (139, 0, 0, 255),
    'darksalmon': (233, 150, 122, 255),
    'darkseagreen': (143, 188, 143, 255),
    'darkslateblue': (72, 61, 139, 255),
    'darkslategray': (47, 79, 79, 255),
    'darkslategrey': (47, 79, 79, 255),
    'darkturquoise': (0, 206, 209, 255),
    'darkviolet': (148, 0, 211, 255),
    'deeppink': (255, 20, 147, 255),
    'deepskyblue': (0, 191, 255, 255),
    'dimgray': (105, 105, 105, 255),
    'dimgrey': (105, 105, 105, 255),
    'dodgerblue': (30, 144, 255, 255),
    'firebrick': (178, 34, 34, 255),
    'floralwhite': (255, 250, 240, 255),
    'forestgreen': (34, 139, 34, 255),
    'fuchsia': (255, 0, 255, 255),
    'gainsboro': (220, 220, 220, 255),
    'ghostwhite': (248, 248, 255, 255),
    'gold': (255, 215, 0, 255),
    'goldenrod': (218, 165, 32, 255),
    'gray': (128, 128, 128, 255),
    'green': (0, 128, 0, 255),
    'greenyellow': (173, 255, 47, 255),
    'grey': (128, 128, 128, 255),
    'honeydew': (240, 255, 240, 255),
    'hotpink': (255, 105, 180, 255),
    'indianred': (205, 92, 92, 255),
    'indigo': (75, 0, 130, 255),
    'ivory': (255, 255, 240, 255),
    'khaki': (240, 230, 140, 255),
    'lavender': (230, 230, 250, 255),
    'lavenderblush': (255, 240, 245, 255),
    'lawngreen': (124, 252, 0, 255),
    'lemonchiffon': (255, 250, 205, 255),
    'lightblue': (173, 216, 230, 255),
    'lightcoral': (240, 128, 128, 255),
    'lightcyan': (224, 255, 255, 255),
    'lightgoldenrodyellow': (250, 250, 210, 255),
    'lightgray': (211, 211, 211, 255),
    'lightgreen': (144, 238, 144, 255),
    'lightgrey': (211, 211, 211, 255),
    'lightpink': (255, 182, 193, 255),
    'lightsalmon': (255, 160, 122, 255),
    'lightseagreen': (32, 178, 170, 255),
    'lightskyblue': (135, 206, 250, 255),
    'lightslategray': (119, 136, 153, 255),
    'lightslategrey': (119, 136, 153, 255),
    'lightsteelblue': (176, 196, 222, 255),
    'lightyellow': (255, 255, 224, 255),
    'lime': (0, 255, 0, 255),
    'limegreen': (50, 205, 50, 255),
    'linen': (250, 240, 230, 255),
    'magenta': (255, 0, 255, 255),
    'maroon': (128, 0, 0, 255),
    'mediumaquamarine': (102, 205, 170, 255),
    'mediumblue': (0, 0, 205, 255),
    'mediumorchid': (186, 85, 211, 255),
    'mediumpurple': (147, 112, 219, 255),
    'mediumseagreen': (60, 179, 113, 255),
    'mediumslateblue': (123, 104, 238, 255),
    'mediumspringgreen': (0, 250, 154, 255),
    'mediumturquoise': (72, 209, 204, 255),
    'mediumvioletred': (199, 21, 133, 255),
    'midnightblue': (25, 25, 112, 255),
    'mintcream': (245, 255, 250, 255),
    'mistyrose': (255, 228, 225, 255),
    'moccasin': (255, 228, 181, 255),
    'navajowhite': (255, 222, 173, 255),
    'navy': (0, 0, 128, 255),
    'oldlace': (253, 245, 230, 255),
    'olive': (128, 128, 0, 255),
    'olivedrab': (107, 142, 35, 255),
    'orange': (255, 165, 0, 255),
    'orangered': (255, 69, 0, 255),
    'orchid': (218, 112, 214, 255),
    'palegoldenrod': (238, 232, 170, 255),
    'palegreen': (152, 251, 152, 255),
    'paleturquoise': (175, 238, 238, 255),
    'palevioletred': (219, 112, 147, 255),
    'papayawhip': (255, 239, 213, 255),
    'peachpuff': (255, 218, 185, 255),
    'peru': (205, 133, 63, 255),
    'pink': (255, 192, 203, 255),
    'plum': (221, 160, 221, 255),
    'powderblue': (176, 224, 230, 255),
    'purple': (128, 0, 128, 255),
    'rebeccapurple': (102, 51, 153, 255),
    'red': (255, 0, 0, 255),
    'rosybrown': (188, 143, 143, 255),
    'royalblue': (65, 105, 225, 255),
    'saddlebrown': (139, 69, 19, 255),
    'salmon': (250, 128, 114, 255),
    'sandybrown': (244, 164, 96, 255),
    'seagreen': (46, 139, 87, 255),
    'seashell': (255, 245, 238, 255),
    'sienna': (160, 82, 45, 255),
    'silver': (192, 192, 192, 255),
    'skyblue': (135, 206, 235, 255),
    'slateblue': (106, 90, 205, 255),
    'slategray': (112, 128, 144, 255),
    'slategrey': (112, 128, 144, 255),
    'snow': (255, 250, 250, 255),
    'springgreen': (0, 255, 127, 255),
    'steelblue': (70, 130, 180, 255),
    'tan': (210, 180, 140, 255),
    'teal': (0, 128, 128, 255),
    'thistle': (216, 191, 216, 255),
    'tomato': (255, 99, 71, 255),
    'turquoise': (64, 224, 208, 255),
    'violet': (238, 130, 238, 255),
    'wheat': (245, 222, 179, 255),
    'white': (255, 255, 255, 255),
    'whitesmoke': (245, 245, 245, 255),
    'yellow': (255, 255, 0, 255),
    'yellowgreen': (154, 205, 50, 255),
})
_BuiltinColors['transparent'] = (0, 0, 0, 0)
_SpecialColors = ["byatom", "byelement", "byhetero", "bychain"]

_BuiltinColormaps = SortedDict()
_CmapRanges = ["full"]

_SequentialLevels = ["residues", "helix", "helices", "strands",
                     "SSEs", "chains", "molmodels", 
                     "volmodels", "allmodels"]


class UserColors(SortedDict, State):
    """Support for per-session colors.

    Accessed through the session object as ``session.user_colors``.
    """

    USER_COLORS_VERSION = 1

    def take_snapshot(self, phase, session, flags):
        if phase != self.SAVE_PHASE:
            return
        return [self.USER_COLORS_VERSION, dict(self)]

    def restore_snapshot(self, phase, session, version, data):
        if version != self.USER_COLORS_VERSION:
            raise RestoreError("Unexpected UserColors version")
        if phase == self.CREATE_PHASE:
            self.clear()
            self.update(data)

    def reset_state(self):
        """Reset state to data-less state"""
        self.clear()


class Color:
    """Basic color support.

    The color components are stored as a 4-element float32 numpy array
    in RGBA order: red, green, blue, and alpha.
    Alpha is the opacity.

    Parameters
    ----------
    rgba : color components
        3- or 4-component array of integers (0-255), or floating point (0-1),
        or # followed by 3 (4), 6 (8), or 12 (16) hex digits (with alpha).
    limit : True
        Clip color array values to [0, 1] inclusive.
    """

    def __init__(self, rgba=None, limit=True):
        from numpy import array, float32, uint8, clip, ndarray, empty
        if isinstance(rgba, ndarray):
            is_uint8 = rgba.dtype == uint8
            a = self.rgba = empty(4, dtype=float32)
            if len(rgba) == 3:
                a[0:3] = rgba
                a[3] = 255 if is_uint8 else 1
            elif len(rgba) == 4:
                a[0:4] = rgba
            else:
                raise ValueError("expected 3 or 4 element array")
            if is_uint8:
                self.rgba /= 255
        elif isinstance(rgba, (tuple, list)):
            if len(rgba) == 3:
                self.rgba = array(list(rgba) + [1.0], dtype=float32)
            elif len(rgba) == 4:
                self.rgba = array(rgba, dtype=float32)
            else:
                raise ValueError("expected 3 or 4 floats")
            if limit:
                clip(self.rgba, 0, 1, out=self.rgba)
        elif isinstance(rgba, Color):
            self.rgba = rgba.rgba[:]    # copy
            if limit:
                clip(self.rgba, 0, 1, out=self.rgba)
        elif isinstance(rgba, str):
            # Hex: #DDD, #DDDDDD, or #DDDDDDDDDDDD
            try:
                if rgba[0] != '#':
                    raise ValueError
                int(rgba[1:], 16)
            except ValueError:
                raise ValueError("expected hexadecimal digits after #")
            if len(rgba) == 4:
                digits = (x for x in rgba[1:])
                values = [int(x, 16) / 15 for x in digits] + [1.0]
            elif len(rgba) == 5:
                digits = (x for x in rgba[1:])
            elif len(rgba) == 7:
                digits = (rgba[x:x + 2] for x in range(1, 7, 2))
                values = [int(x, 16) / 255 for x in digits] + [1.0]
            elif len(rgba) == 9:
                digits = (rgba[x:x + 2] for x in range(1, 9, 2))
                values = [int(x, 16) / 255 for x in digits]
            elif len(rgba) == 13:
                digits = (rgba[x:x + 4] for x in range(1, 13, 4))
                values = [int(x, 16) / 65535 for x in digits] + [1.0]
            elif len(rgba) == 17:
                digits = (rgba[x:x + 4] for x in range(1, 17, 4))
                values = [int(x, 16) / 65535 for x in digits]
            else:
                raise ValueError(
                    "Color constant should have 3 (4), 6 (8), or 12 (16)"
                    " hexadecimal digits")
            self.rgba = array(values, dtype=float32)
        else:
            raise ValueError("Not a color")

    def __eq__(self, other):
        if not isinstance(other, Color):
            return False
        import numpy
        return numpy.array_equal(self.rgba, other.rgba)

    def __ne__(self, other):
        if not isinstance(other, Color):
            return False
        import numpy
        return not numpy.array_equal(self.rgba, other.rgba)

    def opaque(self):
        """Return if the color is opaque."""
        return self.rgba[3] >= 1.0

    def __repr__(self):
        return '%s' % self.rgba

    def uint8x4(self):
        """Return uint8x4 version color"""
        from numpy import trunc, uint8
        return trunc(self.rgba * 255 + .5).astype(uint8)

    def hex(self):
        """Return CSS hex representation (no alpha)"""
        red, green, blue, alpha = self.uint8x4()
        return '#%02x%02x%02x' % (red, green, blue)

    def hex_with_alpha(self):
        """Return hex representation"""
        return '#%02x%02x%02x%02x' % tuple(self.uint8x4())


_color_func = re.compile(r"^(rgb|rgba|hsl|hsla|gray)\s*\(([^)]*)\)")
_number = re.compile(r"\s*[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)")
_units = re.compile(r"\s*(%|deg|grad|rad|turn|)\s*")


def _parse_numbers(text):
    # parse comma separated list of number [units]
    result = []
    start = 0
    while 1:
        m = _number.match(text, start)
        if not m:
            raise cli.AnnotationError("Expected a number", start)
        n = float(m.group())
        n_pos = start
        start = m.end()
        m = _units.match(text, start)
        u = m.group(1)
        u_pos = start
        start = m.end()
        result.append((n, n_pos, u, u_pos))
        if start == len(text):
            return result
        if text[start] != ',':
            raise cli.AnnotationError("Expected a comma", start)
        start += 1


def _convert_number(number, maximum=255, require_percent=False):
    """Return number scaled to 0 <= n <= 1"""
    n, n_pos, u, u_pos = number
    if require_percent and u != '%':
        raise cli.AnnotationError("Must give a percentage", u_pos)
    if u == '':
        return n / maximum
    if u == '%':
        return n / 100
    raise cli.AnnotationError("Unexpected units", u_pos)


def _convert_angle(number):
    n, n_pos, u, u_pos = number
    if u in ('', 'deg'):
        return n / 360
    if u == 'rad':
        from math import pi
        return n / (2 * pi)
    if u == 'grad':
        return n / 400
    if u == 'turn':
        return n
    raise cli.AnnotationError("Unexpected units", u_pos)


class ColorArg(cli.Annotation):
    """Support color names and CSS3 color specifications.

    The CSS3 color specifications supported are: rgb, rgba, hsl, hsla, and
    gray from CSS4.

    The following examples are all ``red``, except for the gray ones::

        red
        #f00
        #0xff0000
        rgb(255, 0, 0)
        rgb(100%, 0, 0)
        rgba(100%, 0, 0, 1)
        rgba(100%, 0, 0, 100%)
        hsl(0, 100%, 50%)
        gray(128)
        gray(50%)
    """
    name = 'a color'

    @staticmethod
    def parse(text, session):
        if text[0] == '#':
            token, text, rest = cli.next_token(text)
            return Color(token), text, rest
        m = _color_func.match(text)
        if m is None:
            token, text, rest = cli.next_token(text)
            if session is not None:
                i = session.user_colors.bisect_left(token)
                if i < len(session.user_colors):
                    name = session.user_colors.iloc[i]
                    if name.startswith(token):
                        return session.user_colors[name], name, rest
            i = _BuiltinColors.bisect_left(token)
            if i >= len(_BuiltinColors):
                raise ValueError("Invalid color name")
            name = _BuiltinColors.iloc[i]
            if not name.startswith(token):
                raise ValueError("Invalid color name")
            color = _BuiltinColors[name]
            return Color([x / 255 for x in color]), name, rest
        color_space = m.group(1)
        numbers = _parse_numbers(m.group(2))
        rest = text[m.end():]
        if color_space == 'gray' and len(numbers) in (1, 2):
            # gray( number [%], [ number [%] ])
            try:
                x = _convert_number(numbers[0])
                if len(numbers) == 2:
                    alpha = _convert_number(numbers[1], maximum=1)
                else:
                    alpha = 1
            except cli.AnnotationError as err:
                err.offset += m.end(1)
                raise
            return Color([x, x, x, alpha]), m.group(), rest
        if color_space == 'rgb' and len(numbers) == 3:
            # rgb( number [%], number [%], number [%])
            try:
                red = _convert_number(numbers[0])
                green = _convert_number(numbers[1])
                blue = _convert_number(numbers[2])
            except cli.AnnotationError as err:
                err.offset += m.end(1)
                raise
            return Color([red, green, blue, 1]), m.group(), rest
        if color_space == 'rgba' and len(numbers) == 4:
            # rgba( number [%], number [%], number [%], number [%])
            try:
                red = _convert_number(numbers[0])
                green = _convert_number(numbers[1])
                blue = _convert_number(numbers[2])
                alpha = _convert_number(numbers[3], maximum=1)
            except cli.AnnotationError as err:
                err.offset += m.end(1)
                raise
            return Color([red, green, blue, alpha]), m.group(), rest
        if color_space == 'hsl' and len(numbers) == 3:
            # hsl( number [%], number [%], number [%])
            try:
                hue = _convert_angle(numbers[0])
                sat = _convert_number(numbers[1], require_percent=True)
                light = _convert_number(numbers[2], require_percent=True)
            except cli.AnnotationError as err:
                err.offset += m.end(1)
                raise
            if sat < 0:
                sat = 0
            if light < 0:
                light = 0
            elif light > 1:
                light = 1
            import colorsys
            red, green, blue = colorsys.hls_to_rgb(hue, light, sat)
            return Color([red, green, blue, 1]), m.group(), rest
        if color_space == 'hsla' and len(numbers) == 4:
            # hsla( number [%], number [%], number [%], number [%])
            try:
                hue = _convert_angle(numbers[0])
                sat = _convert_number(numbers[1], require_percent=True)
                light = _convert_number(numbers[2], require_percent=True)
                alpha = _convert_number(numbers[3], maximum=1)
            except cli.AnnotationError as err:
                err.offset += m.end(1)
                raise
            if sat < 0:
                sat = 0
            if light < 0:
                light = 0
            elif light > 1:
                light = 1
            import colorsys
            red, green, blue = colorsys.hls_to_rgb(hue, light, sat)
            return Color([red, green, blue, alpha]), m.group(), rest
        raise ValueError("Unknown color description")

# -----------------------------------------------------------------------------


class UserColormaps(SortedDict, State):
    """Support for per-session colormaps.

    Accessed through the session object as ``session.user_colormaps``.
    """

    USER_COLORMAPS_VERSION = 1

    def take_snapshot(self, phase, session, flags):
        if phase != self.SAVE_PHASE:
            return
        return [self.USER_COLORMAPS_VERSION, dict(self)]

    def restore_snapshot(self, phase, session, version, data):
        if version != self.USER_COLORMAPS_VERSION:
            raise RestoreError("Unexpected UserColormaps version")
        if phase == self.CREATE_PHASE:
            self.clear()
            self.update(data)

    def reset_state(self):
        """Reset state to data-less state"""
        self.clear()


class Colormap:
    """Basic colormap support.

    Colormaps keep track of two parallel arrays: values and colors.
    When given a value, the colors corresponding to tracked values below
    and above the target value; the target color is computed by
    interpolating in RGB space.  Two additional colors are tracked
    by colormap for values above and below the minimum and maximum
    tracked values.  A "no value" color is also tracked.

    Attributes
    ----------
        data_values : 

    Parameters
    ----------
    data_values : array of floating point values
        sorted list or numpy array of floating point values.
    colors : array of colors
        list or numpy array of Color instances.
    color_above_value_range : color for values above maximum value
        instance of Color.
    color_below_value_range : color for values below minimum value
        instance of Color.
    color_no_value : default color when no value is defined
        instance of Color.
    """
    def __init__(self, data_values, colors,
                 color_above_value_range = None,
                 color_below_value_range = None,
                 color_no_value= None):
        from numpy import array, float32, ndarray
        if not data_values:
            import numpy
            self.data_values = numpy.linspace(0.0, 1.0, len(colors))
        elif isinstance(data_values, ndarray):
            self.data_values = data_values
        else:
            self.data_values = array(data_values, dtype=float32)
        c = colors[0]
        if isinstance(c, Color):
            self.colors = array([c.rgba for c in colors])
        elif isinstance(colors, ndarray):
            self.colors = colors
        else:
            self.colors = array(colors, dtype=float32)

        if color_above_value_range == None:
            color_above_value_range = colors[-1]
        if color_below_value_range == None:
            color_below_value_range = colors[0]
        if color_no_value == None:
            color_no_value = (.5,.5,.5,1)

        self.color_above_value_range = color_above_value_range
        self.color_below_value_range = color_below_value_range
        self.color_no_value = color_no_value

    def get_colors_for(self, values):
        """Return numpy array of rgba for given values.

        Parameter
        ---------
        values : numpy array of float32

        Return Value
        ------------
        numpy array of rgba (Nx4 where N is the length of "values".)
        """
        from .. import map
        colors = map.interpolate_colormap(values, self.data_values, self.colors,
                                          self.color_above_value_range,
                                          self.color_below_value_range)
        return colors


# Initialize built-in colormaps
_BuiltinColormaps['rainbow'] = Colormap(None, ((1,0,0,1), (1,1,0,1), (0,1,0,1), (0,1,1,1), (0,0,1,1)))
_BuiltinColormaps['grayscale'] = Colormap(None, ((0,0,0,1), (1,1,1,1)))
# _BuiltinColormaps['red-white-blue'] = Colormap(None, ((1,0,0,1), (1,1,1,1), (0,0,1,1)))
_BuiltinColormaps['red-white-blue'] = Colormap(None, ((1,0,0,1), (.7,.7,.7,1), (0,0,1,1)))
_BuiltinColormaps['blue-white-red'] = Colormap(None, ((0,0,1,1), (1,1,1,1), (1,0,0,1)))
_BuiltinColormaps['cyan-white-maroon'] = Colormap(None, ((0.059,0.78,0.81,1), (1,1,1,1), (0.62,0.125,0.37,1)))
# Add some aliases
_BuiltinColormaps['redblue'] = _BuiltinColormaps['red-white-blue']
_BuiltinColormaps['bluered'] = _BuiltinColormaps['blue-white-red']
_BuiltinColormaps['gray'] = _BuiltinColormaps['grayscale']
_BuiltinColormaps['cyanmaroon'] = _BuiltinColormaps['cyan-white-maroon']


class ColormapArg(cli.Annotation):
    """Support color map names and value-color pairs specifications.
    """
    name = 'a colormap'

    @staticmethod
    def parse(text, session):
        token, text, rest = cli.next_token(text)
        parts = token.split(':')
        if len(parts) > 1:
            values = []
            colors = []
            for p in parts:
                vc = p.split(',')
                if len(vc) == 1:
                    color, t, r = ColorArg.parse(vc[0], session)
                elif len(vc) == 2:
                    values.append(float(vc[0]))
                    color, t, r = ColorArg.parse(vc[1], session)
                else:
                    raise ValueError("Too many fields in colormap")
                if r:
                    raise ValueError("Bad color in colormap")
                colors.append(color)
            if len(values) != len(colors):
                raise ValueError("Number of values and color must match in colormap")
            return Colormap(values, colors), text, rest
        else:
            if session is not None:
                i = session.user_colormaps.bisect_left(token)
                if i < len(session.user_colormaps):
                    name = session.user_colormaps.iloc[i]
                    if name.startswith(token):
                        return session.user_colormaps[name], name, rest
            i = _BuiltinColormaps.bisect_left(token)
            if i >= len(_BuiltinColormaps):
                raise ValueError("Invalid colormap name")
            name = _BuiltinColormaps.iloc[i]
            if not name.startswith(token):
                raise ValueError("Invalid colormap name")
            return _BuiltinColormaps[name], name, rest

# -----------------------------------------------------------------------------


def define_color(session, name, color=None):
    """Create a user defined color."""
    if ' ' in name:
        from ..errors import UsetError
        raise UserError("Sorry, spaces are not alllowed in color names")
    if color is None:
        if session is not None:
            i = session.user_colors.bisect_left(name)
            if i < len(session.user_colors):
                real_name = session.user_colors.iloc[i]
                if real_name.startswith(name):
                    color = session.user_colors[real_name]
        if color is None:
            i = _BuiltinColors.bisect_left(name)
            if i < len(_BuiltinColors):
                real_name = _BuiltinColors.iloc[i]
                if real_name.startswith(name):
                    color = Color([x / 255 for x in _BuiltinColors[real_name]])
        if color is None:
            session.logger.status('Unknown color %r' % name)
            return

        def percent(x):
            if x == 1:
                return 100
            return ((x * 10000) % 10000) / 100
        red, green, blue, alpha = color.rgba
        if alpha >= 1:
            transmit = 'opaque'
        elif alpha <= 0:
            transmit = 'transparent'
        else:
            transmit = '%g%% transparent' % percent(1 - alpha)

        msg = 'Color %r is %s, %.4g%% red, %.4g%% green, and %.4g%% blue' % (
            real_name, transmit, percent(red), percent(green),
            percent(blue))
        if session is None:
            print(msg)
            return
        session.logger.status(msg)
        session.logger.info(
            msg +
            '<div style="width:1em; height:.4em;'
            ' display:inline-block;'
            ' border:1px solid #000; background-color:%s"/>'
            % color.hex())
        return
    session.user_colors[name] = color


def undefine_color(session, name):
    """Remove a user defined color."""
    del session.user_colors[name]


def color(session, color, spec=None):
    """Color an object specification."""
    from . import atomspec
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)

    rgba8 = color.uint8x4()
    atoms = results.atoms
    if atoms is None:
        na = 0
    else:
        atoms.colors = rgba8
        na = len(atoms)

    ns = 0
    from ..structure import AtomicStructure
    for m in results.models:
        if not isinstance(m, AtomicStructure):
            m.color = rgba8
            ns += 1

    what = []
    if na > 0:
        what.append('%d atoms' % na)
    if ns > 0:
        what.append('%d surfaces' % ns)
    if na == 0 and ns == 0:
        what.append('nothing')
    session.logger.status('Colored %s' % ', '.join(what))


def rcolor(session, color, spec=None):
    """Color ribbons for an object specification."""
    from . import atomspec
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)

    rgba8 = color.uint8x4()
    residues = results.atoms.unique_residues
    if residues is None:
        nr = 0
    else:
        residues.ribbon_colors = rgba8
        nr = len(residues)

    what = []
    if nr > 0:
        what.append('%d residues' % nr)
    else:
        what.append('nothing')
    session.logger.status('Colored %s' % ', '.join(what))


def ecolor(session, spec, color=None, target=None,
           sequential=None, cmap=None, cmap_range=None):
    """Color an object specification."""
    from . import atomspec
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    if sequential is not None:
        try:
            f = _SequentialColor[sequential]
        except KeyError:
            from ..errors import UserError
            raise UserError("sequential \"%s\" not implemented yet"
                            % sequential)
        else:
            f(results, cmap, target)
            return
    what = []

    if target is None or 'a' in target:
        # atoms/bonds
        atoms = results.atoms
        if atoms is not None:
            if color in _SpecialColors:
                if color == "byelement":
                    _set_element_colors(atoms, False)
                elif color == "byhetero":
                    _set_element_colors(atoms, True)
                elif color == "bychain":
                    color_atoms_by_chain(atoms)
                else:
                    # Other "colors" do not apply to atoms
                    pass
            else:
                atoms.colors = color.uint8x4()
            what.append('%d atoms' % len(atoms))

    if target is None or 'l' in target:
        if target is not None:
            session.logger.warning('Label colors not supported yet')

    if target is None or 's' in target:
        from .scolor import scolor
        if color in _SpecialColors:
            ns = scolor(session, results.atoms, byatom=True)
        else:
            ns = scolor(session, results.atoms, color)
        what.append('%d surfaces' % ns)

    if target is None or 'c' in target:
        residues = results.atoms.unique_residues
        if color not in _SpecialColors:
            residues.ribbon_colors = color.uint8x4()
        elif color == 'bychain':
            color_ribbons_by_chain(residues)
        what.append('%d residues' % len(residues))

    if target is None or 'r' in target:
        if target is not None:
            session.logger.warning('Residue label colors not supported yet')

    if target is None or 'n' in target:
        if target is not None:
            session.logger.warning('Non-molecular model-level colors not supported yet')

    if target is None or 'm' in target:
        if target is not None:
            session.logger.warning('Model-level colors not supported yet')

    if target is None or 'b' in target:
        if target is not None:
            session.logger.warning('Bond colors not supported yet')

    if target is None or 'p' in target:
        if target is not None:
            session.logger.warning('Pseudobond colors not supported yet')

    if target is None or 'd' in target:
        if target is not None:
            session.logger.warning('Distances colors not supported yet')

    if not what:
        what.append('nothing')
    session.logger.status('Colored %s' % ', '.join(what))


def _set_element_colors(atoms, skip_carbon):
    import numpy
    en = atoms.element_numbers
    for e in numpy.unique(en):
        if not skip_carbon or e != 6:
            ae = atoms.filter(en == e)
            atoms.filter(en == e).colors = element_colors(e)

# -----------------------------------------------------------------------------
#
def color_by_element(atoms):
    atoms.colors = element_colors(atoms.element_numbers)

# -----------------------------------------------------------------------------
#
element_rgba_256 = None
def element_colors(element_numbers):
    global element_rgba_256
    if element_rgba_256 is None:
        from numpy import empty, uint8
        element_rgba_256 = ec = empty((256, 4), uint8)
        ec[:,:3] = 180
        ec[:,3] = 255
        # jmol element colors
        colors = (
            (1,	(255,255,255)),	 # H
            (2,	(217,255,255)),	 # He
            (3,	(204,128,255)),	 # Li
            (4,	(194,255,0)),	 # Be  
            (5,	(255,181,181)),	 # B
            (6,	(144,144,144)),	 # C
            (7,	(48,80,248)),	 # N  
            (8,	(255,13,13)),	 # O  
            (9, (144,224,80)),	 # F 
            (10, (179,227,245)), # Ne
            (11, (171,92,242)),	 # Na 
            (12, (138,255,0)),	 # Mg  
            (13, (191,166,166)), # Al
            (14, (240,200,160)), # Si
            (15, (255,128,0)),	 # P  
            (16, (255,255,48)),	 # S 
            (17, (31,240,31)),	 # Cl  
            (18, (128,209,227)), # Ar
            (19, (143,64,212)),	 # K 
            (20, (61,255,0)),	 # Ca   
            (21, (230,230,230)), # Sc
            (22, (191,194,199)), # Ti
            (23, (166,166,171)), # V
            (24, (138,153,199)), # Cr
            (25, (156,122,199)), # Mn
            (26, (224,102,51)),	 # Fe 
            (27, (240,144,160)), # Co
            (28, (80,208,80)),	 # Ni  
            (29, (200,128,51)),	 # Cu 
            (30, (125,128,176)), # Zn
            (31, (194,143,143)), # Ga
            (32, (102,143,143)), # Ge
            (33, (189,128,227)), # As
            (34, (255,161,0)),	 # Se  
            (35, (166,41,41)),	 # Br  
            (36, (92,184,209)),	 # Kr 
            (37, (112,46,176)),	 # Rb 
            (38, (0,255,0)),	 # Sr    
            (39, (148,255,255)), # Y
            (40, (148,224,224)), # Zr
            (41, (115,194,201)), # Nb
            (42, (84,181,181)),	 # Mo 
            (43, (59,158,158)),	 # Tc 
            (44, (36,143,143)),	 # Ru 
            (45, (10,125,140)),	 # Rh 
            (46, (0,105,133)),	 # Pd  
            (47, (192,192,192)), # Ag
            (48, (255,217,143)), # Cd
            (49, (166,117,115)), # In
            (50, (102,128,128)), # Sn
            (51, (158,99,181)),	 # Sb 
            (52, (212,122,0)),	 # Te  
            (53, (148,0,148)),	 # I  
            (54, (66,158,176)),	 # Xe 
            (55, (87,23,143)),	 # Cs  
            (56, (0,201,0)),	 # Ba    
            (57, (112,212,255)), # La
            (58, (255,255,199)), # Ce
            (59, (217,255,199)), # Pr
            (60, (199,255,199)), # Nd
            (61, (163,255,199)), # Pm
            (62, (143,255,199)), # Sm
            (63, (97,255,199)),	 # Eu 
            (64, (69,255,199)),	 # Gd 
            (65, (48,255,199)),	 # Tb 
            (66, (31,255,199)),	 # Dy 
            (67, (0,255,156)),	 # Ho  
            (68, (0,230,117)),	 # Er  
            (69, (0,212,82)),	 # Tm   
            (70, (0,191,56)),	 # Yb   
            (71, (0,171,36)),	 # Lu   
            (72, (77,194,255)),	 # Hf 
            (73, (77,166,255)),	 # Ta 
            (74, (33,148,214)),	 # W 
            (75, (38,125,171)),	 # Re 
            (76, (38,102,150)),	 # Os 
            (77, (23,84,135)),	 # Ir  
            (78, (208,208,224)), # Pt
            (79, (255,209,35)),	 # Au 
            (80, (184,184,208)), # Hg
            (81, (166,84,77)),	 # Tl  
            (82, (87,89,97)),	 # Pb   
            (83, (158,79,181)),	 # Bi 
            (84, (171,92,0)),	 # Po   
            (85, (117,79,69)),	 # At  
            (86, (66,130,150)),	 # Rn 
            (87, (66,0,102)),	 # Fr   
            (88, (0,125,0)),	 # Ra    
            (89, (112,171,250)), # Ac
            (90, (0,186,255)),	 # Th  
            (91, (0,161,255)),	 # Pa  
            (92, (0,143,255)),	 # U  
            (93, (0,128,255)),	 # Np  
            (94, (0,107,255)),	 # Pu  
            (95, (84,92,242)),	 # Am  
            (96, (120,92,227)),	 # Cm 
            (97, (138,79,227)),	 # Bk 
            (98, (161,54,212)),	 # Cf 
            (99, (179,31,212)),	 # Es 
            (100, (179,31,186)), # Fm 
            (101, (179,13,166)), # Md 
            (102, (189,13,135)), # No 
            (103, (199,0,102)),	 # Lr  
            (104, (204,0,89)),	 # Rf   
            (105, (209,0,79)),	 # Db   
            (106, (217,0,69)),	 # Sg   
            (107, (224,0,56)),	 # Bh   
            (108, (230,0,46)),	 # Hs   
            (109, (235,0,38)),	 # Mt   
        )
        for e, rgb in colors:
            ec[e,:3] = rgb

    colors = element_rgba_256[element_numbers]
    return colors

# -----------------------------------------------------------------------------
#
def color_by_chain(atoms):
    color_atoms_by_chain(atoms)
    color_ribbons_by_chain(atoms.unique_residues)
def color_atoms_by_chain(atoms):
    atoms.colors = chain_colors(atoms.residues.chain_ids)
def color_ribbons_by_chain(residues):
    residues.ribbon_colors = chain_colors(residues.chain_ids)

# -----------------------------------------------------------------------------
#
rgba_256 = None
def chain_colors(cids):

    global rgba_256
    if rgba_256 is None:
        rgba_256 = {
          'a':(123,104,238,255),
          'b':(240,128,128,255),
          'c':(143,188,143,255),
          'd':(222,184,135,255),
          'e':(255,127,80,255),
          'f':(128,128,128,255),
          'g':(107,142,35,255),
          'h':(100,100,100,255),
          'i':(255,255,0,255),
          'j':(55,19,112,255),
          'k':(255,255,150,255),
          'l':(202,62,94,255),
          'm':(205,145,63,255),
          'n':(12,75,100,255),
          'o':(255,0,0,255),
          'p':(175,155,50,255),
          'q':(105,205,48,255),
          'r':(37,70,25,255),
          's':(121,33,135,255),
          't':(83,140,208,255),
          'u':(0,154,37,255),
          'v':(178,220,205,255),
          'w':(255,152,213,255),
          'x':(200,90,174,255),
          'y':(175,200,74,255),
          'z':(63,25,12,255),
          '1': (87, 87, 87,255),
          '2': (173, 35, 35,255),
          '3': (42, 75, 215,255),
          '4': (29, 105, 20,255),
          '5': (129, 74, 25,255),
          '6': (129, 38, 192,255),
          '7': (160, 160, 160,255),
          '8': (129, 197, 122,255),
          '9': (157, 175, 255,255),
          '0': (41, 208, 208,255),
        }

    for cid in set(cids):
        c = str(cid).lower()
        if not c in rgba_256:
            from random import randint, seed
            seed(c)
            rgba_256[c] = (randint(128,255),randint(128,255),randint(128,255),255)

    from numpy import array, uint8
    c = array(tuple(rgba_256[cid.lower()] for cid in cids), uint8)
    return c

# -----------------------------------------------------------------------------
#
def chain_rgba(cid):
    return tuple(float(c/255.0) for c in chain_colors([cid])[0])

# -----------------------------------------------------------------------------
#
def chain_rgba8(cid):
    return chain_colors([cid])[0]

# -----------------------------------------------------------------------------
#
def _set_sequential_chain(selected, cmap, target):
    # Organize selected atoms by structure and then chain
    sa = selected.atoms
    chain_atoms = sa.filter(sa.in_chains)
    structures = {}
    for structure, chain_id, atoms in chain_atoms.by_chain:
        try:
            sl = structures[structure]
        except KeyError:
            sl = []
            structures[structure] = sl
        sl.append((chain_id, atoms))
    # Make sure there is a colormap
    if cmap is None:
        cmap = _BuiltinColormaps["rainbow"]
    # Each structure is colored separately with cmap applied by chain
    import numpy
    for sl in structures.values():
        colors = cmap.get_colors_for(numpy.linspace(0.0, 1.0, len(sl)))
        for color, (chain_id, atoms) in zip(colors, sl):
            c = Color(color).uint8x4()
            if target is None or 'a' in target:
                atoms.colors = c
            if target is None or 'c' in target:
                atoms.unique_residues.ribbon_colors = c

_SequentialColor = {
    "chains": _set_sequential_chain,
}


# -----------------------------------------------------------------------------
#
def register_commands():
    from . import atomspec
    cli.register(
        'color',
        cli.CmdDesc(required=[("color", ColorArg)],
                    optional=[("spec", atomspec.AtomSpecArg)],
                    synopsis="color specified objects"),
        color
    )
    cli.register(
        'rcolor',
        cli.CmdDesc(required=[("color", ColorArg)],
                    optional=[("spec", atomspec.AtomSpecArg)],
                    synopsis="color specified ribbons"),
        rcolor
    )
    cli.register(
        'colordef',
        cli.CmdDesc(required=[('name', cli.StringArg)],
                    optional=[('color', ColorArg)],
                    synopsis="define a custom color"),
        define_color
    )
    cli.register(
        '~colordef',
        cli.CmdDesc(required=[('name', cli.StringArg)],
                    synopsis="remove color definition"),
        undefine_color
    )
    cli.register(
        'ecolor',
        cli.CmdDesc(required=[('spec', cli.Or(atomspec.AtomSpecArg, cli.EmptyArg))],
                    optional=[('color', cli.Or(ColorArg, cli.EnumOf(_SpecialColors)))],
                    keyword=[('target', cli.StringArg),
                             ('sequential', cli.EnumOf(_SequentialLevels)),
                             ('cmap', ColormapArg),
                             ('cmap_range', cli.Or(cli.TupleOf(cli.FloatArg, 2),
                                                    cli.EnumOf(_CmapRanges)))],
                    synopsis="testing real color syntax"),
        ecolor
    )


def test():
    tests = [
        "0x00ff00",
        "#0f0",
        "#00ffff",
        "gray(50)",
        "gray(50%)",
        "rgb(0, 0, 255)",
        "rgb(100%, 0, 0)",
        "red",
        "hsl(0, 100%, 50%)",  # red
        "lime",
        "hsl(120deg, 100%, 50%)",  # lime
        "darkgreen",
        "hsl(120, 100%, 20%)",  # darkgreen
        "lightgreen",
        "hsl(120, 75%, 75%)",  # lightgreen
    ]
    for t in tests:
        print(t)
        try:
            print(ColorArg.parse(t))
        except ValueError as err:
            print(err)
    print('same:', ColorArg.parse('white')[0] == Color('#ffffff'))
