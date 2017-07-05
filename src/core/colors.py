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
color: basic color support
===========================

A :py:class:`Color` class that hold a :py:class:`numpy.array` of
four 32-bit floats.

CSS3 colors are supported with the addition of the gray() specification
from the CSS4 draft and the CSS4 color names.
"""
from sortedcontainers import SortedDict
from .state import State, CORE_STATE_VERSION

BuiltinColormaps = SortedDict()


class UserColors(SortedDict, State):
    """Support for per-session colors.

    Accessed through the session object as ``session.user_colors``.
    """

    def __init__(self):
        SortedDict.__init__(self)
        self.update(BuiltinColors)

    def take_snapshot(self, session, flags):
        # only save differences from builtin colors
        cmap = {name: color for name, color in self.items()
                if name not in BuiltinColors or color != BuiltinColors[name]}
        data = {'colors':cmap,
                'version': CORE_STATE_VERSION}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        c = UserColors()
        c.update(data['colors'])
        return c

    def reset_state(self, session):
        """Reset state to data-less state"""
        self.clear()
        self.update(BuiltinColors)

    def list(self, all=False):
        if all:
            return list(self.keys())
        return [name for name, color in self.items()
                if name not in BuiltinColors or color != BuiltinColors[name]]

    def add(self, key, value):
        if key in BuiltinColors:
            raise ValueError('Can not override builtin color')
        self[key] = value

    def remove(self, key):
        if key in BuiltinColors:
            raise ValueError('Can not remove builtin color')
        del self[key]


class Color(State):
    """Basic color support.

    The color components are stored as a 4-element float32 numpy array
    in RGBA order: red, green, blue, and alpha.
    Alpha is the opacity.

    Parameters
    ----------
    rgba : color components
        3- or 4-component array of integers (0-255), or floating point (0-1),
        or # followed by 3 (4), 6 (8), or 12 (16) hex digits (with alpha).
    limit : bool
        Clip color array values to [0, 1] inclusive.
    mutable : bool
        Whether color components can be changed.
    """

    def __init__(self, rgba=None, limit=True, mutable=True):
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
                values = [int(x, 16) / 15 for x in digits]
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
        if not mutable:
            self.rgba.flags.writeable = False

    def __eq__(self, other):
        if not isinstance(other, Color):
            return False
        import numpy
        return numpy.array_equal(self.rgba, other.rgba)

    def __ne__(self, other):
        if not isinstance(other, Color):
            return True
        import numpy
        return not numpy.array_equal(self.rgba, other.rgba)

    def take_snapshot(self, session, flags):
        data = {'rgba': self.rgba}
        return CORE_STATE_VERSION, data

    @staticmethod
    def restore_snapshot(session, data):
        return Color(data['rgba'], limit=False)

    def reset_state(self, session):
        pass

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

# -----------------------------------------------------------------------------


class UserColormaps(SortedDict, State):
    """Support for per-session colormaps.

    Accessed through the session object as ``session.user_colormaps``.
    """

    def take_snapshot(self, session, flags):
        data = {'colormaps': dict(self),
                'version': CORE_STATE_VERSION,}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        c = UserColormaps()
        c.update(data['colormaps'])
        return c

    def reset_state(self, session):
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
                 color_above_value_range=None,
                 color_below_value_range=None,
                 color_no_value=None):
        self.values_specified = (data_values is not None)
        from numpy import array, float32, ndarray, argsort
        if data_values is None:
            import numpy
            v = numpy.linspace(0.0, 1.0, len(colors))
        elif isinstance(data_values, ndarray):
            v = data_values
        else:
            v = array(data_values, dtype=float32)
        order = argsort(v)
        self.data_values = v[order]
        if isinstance(colors[0], Color):
            c = array([c.rgba for c in colors])
        elif isinstance(colors, ndarray):
            c = colors
        else:
            c = array(colors, dtype=float32)
        self.colors = c[order]

        if color_above_value_range is None:
            color_above_value_range = self.colors[-1]
        if color_below_value_range is None:
            color_below_value_range = self.colors[0]
        if color_no_value is None:
            color_no_value = (.5, .5, .5, 1)

        self.color_above_value_range = color_above_value_range
        self.color_below_value_range = color_below_value_range
        self.color_no_value = color_no_value

    def interpolated_rgba(self, values):
        """Return numpy array of float rgba for given values.

        Parameter
        ---------
        values : numpy array of float32

        Return Value
        ------------
        numpy array of rgba (Nx4 where N is the length of "values".)
        """
        from . import map
        colors = map.interpolate_colormap(values, self.data_values, self.colors,
                                          self.color_above_value_range,
                                          self.color_below_value_range)
        return colors

    def interpolated_rgba8(self, values):
        c = self.interpolated_rgba(values)
        c *= 255
        from numpy import uint8
        c8 = c.astype(uint8)
        return c8

    def value_range(self):
        v = self.data_values
        return (v[0], v[-1])
    
    def linear_range(self, min_value, max_value):
        import numpy
        v = numpy.linspace(min_value, max_value, len(self.colors))
        cmap = Colormap(v, self.colors,
                        self.color_above_value_range,
                        self.color_below_value_range,
                        self.color_no_value)
        return cmap
    
    def rescale_range(self, value0, value1):
        '''Return new colormap with [0,1] range becoming [value0,value1].'''
        v = self.data_values.copy()
        v *= (value1 - value0)
        v += value0
        cmap = Colormap(v, self.colors,
                        self.color_above_value_range,
                        self.color_below_value_range,
                        self.color_no_value)
        return cmap


# Initialize built-in colormaps
# Rainbow is blue to red instead of red to blue so that N-terminus to C-terminus rainbow coloring
# produces the conventional blue to red.
BuiltinColormaps['rainbow'] = Colormap(None, ((0, 0, 1, 1), (0, 1, 1, 1), (0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1)))
BuiltinColormaps['grayscale'] = Colormap(None, ((0, 0, 0, 1), (1, 1, 1, 1)))
BuiltinColormaps['red-white-blue'] = Colormap(None, ((1, 0, 0, 1), (1, 1, 1, 1), (0, 0, 1, 1)))
#BuiltinColormaps['red-white-blue'] = Colormap(None, ((1, 0, 0, 1), (.7, .7, .7, 1), (0, 0, 1, 1)))
BuiltinColormaps['blue-white-red'] = Colormap(None, ((0, 0, 1, 1), (1, 1, 1, 1), (1, 0, 0, 1)))
BuiltinColormaps['cyan-white-maroon'] = Colormap(None, ((0.059, 0.78, 0.81, 1), (1, 1, 1, 1), (0.62, 0.125, 0.37, 1)))
#BuiltinColormaps['lipophilicity'] = Colormap(None, ((.118,.565,1,1), (1,1,1,1), (1,.271,0,1)))	# dodger blue, white, orange red
BuiltinColormaps['lipophilicity'] = Colormap(None, ((0,139/255,139/255,1), (1,1,1,1), (184/255,134/255,11/255,1)))	# dark cyan, white, dark goldenrod

# Add some aliases
BuiltinColormaps['redblue'] = BuiltinColormaps['red-white-blue']
BuiltinColormaps['bluered'] = BuiltinColormaps['blue-white-red']
BuiltinColormaps['gray'] = BuiltinColormaps['grayscale']
BuiltinColormaps['cyanmaroon'] = BuiltinColormaps['cyan-white-maroon']


_df_state = {}


def distinguish_from(rgbs, *, num_candidates=3, seed=None, save_state=True):
    """Best effort to return an RGB that perceptually differs from the given RGB(A)s"""
    if rgbs and len(rgbs[0]) > 3:
        rgbs = [rgba[:3] for rgba in rgbs]

    max_diff = None
    import random
    global _df_state
    if seed is not None:
        if save_state and seed in _df_state:
            random.setstate(_df_state[seed])
        else:
            random.seed(seed)
    for i in range(num_candidates):
        candidate = tuple([random.random() for i in range(3)])
        if not rgbs:
            if save_state and seed is not None:
                _df_state[seed] = random.getstate()
            return candidate
        min_diff = None
        for rgb in rgbs:
            diff = abs(rgb[0] - candidate[0]) + abs(rgb[1] - candidate[1]) \
                + 0.5 * abs(rgb[2] - candidate[2])
            if min_diff is None or diff < min_diff:
                min_diff = diff
        if max_diff is None or min_diff > max_diff:
            max_diff = min_diff
            best_candidate = candidate
    if save_state and seed is not None:
        _df_state[seed] = random.getstate()
    return best_candidate

def contrast_with(rgb):
    """Depending on which contrasts best with the given RGB(A), return white or black (RGB)"""
    if rgb[0]*2 + rgb[1]*3 + rgb[2] < 0.417:
        return (1.0, 1.0, 1.0)
    return (0.0, 0.0, 0.0)

# -----------------------------------------------------------------------------
#

# CSS4 colors + multiword color names
BuiltinColors = SortedDict({
    'aliceblue': (240, 248, 255, 255),
    'alice blue': (240, 248, 255, 255),
    'antiquewhite': (250, 235, 215, 255),
    'antique white': (250, 235, 215, 255),
    'aqua': (0, 255, 255, 255),
    'aquamarine': (127, 255, 212, 255),
    'azure': (240, 255, 255, 255),
    'beige': (245, 245, 220, 255),
    'bisque': (255, 228, 196, 255),
    'black': (0, 0, 0, 255),
    'blanchedalmond': (255, 235, 205, 255),
    'blanched almond': (255, 235, 205, 255),
    'blue': (0, 0, 255, 255),
    'blueviolet': (138, 43, 226, 255),
    'blue violet': (138, 43, 226, 255),
    'brown': (165, 42, 42, 255),
    'burlywood': (222, 184, 135, 255),
    'burly wood': (222, 184, 135, 255),
    'cadetblue': (95, 158, 160, 255),
    'cadet blue': (95, 158, 160, 255),
    'chartreuse': (127, 255, 0, 255),
    'chocolate': (210, 105, 30, 255),
    'coral': (255, 127, 80, 255),
    'cornflowerblue': (100, 149, 237, 255),
    'cornflower blue': (100, 149, 237, 255),
    'cornsilk': (255, 248, 220, 255),
    'crimson': (220, 20, 60, 255),
    'cyan': (0, 255, 255, 255),
    'darkblue': (0, 0, 139, 255),
    'dark blue': (0, 0, 139, 255),
    'darkcyan': (0, 139, 139, 255),
    'dark cyan': (0, 139, 139, 255),
    'darkgoldenrod': (184, 134, 11, 255),
    'dark goldenrod': (184, 134, 11, 255),
    'darkgray': (169, 169, 169, 255),
    'dark gray': (169, 169, 169, 255),
    'darkgreen': (0, 100, 0, 255),
    'dark green': (0, 100, 0, 255),
    'darkgrey': (169, 169, 169, 255),
    'dark grey': (169, 169, 169, 255),
    'darkkhaki': (189, 183, 107, 255),
    'dark khaki': (189, 183, 107, 255),
    'darkmagenta': (139, 0, 139, 255),
    'dark magenta': (139, 0, 139, 255),
    'darkolivegreen': (85, 107, 47, 255),
    'dark olive green': (85, 107, 47, 255),
    'darkorange': (255, 140, 0, 255),
    'dark orange': (255, 140, 0, 255),
    'darkorchid': (153, 50, 204, 255),
    'dark orchid': (153, 50, 204, 255),
    'darkred': (139, 0, 0, 255),
    'dark red': (139, 0, 0, 255),
    'darksalmon': (233, 150, 122, 255),
    'dark salmon': (233, 150, 122, 255),
    'darkseagreen': (143, 188, 143, 255),
    'dark seagreen': (143, 188, 143, 255),
    'dark sea green': (143, 188, 143, 255),
    'darkslateblue': (72, 61, 139, 255),
    'dark slate blue': (72, 61, 139, 255),
    'darkslategray': (47, 79, 79, 255),
    'dark slate gray': (47, 79, 79, 255),
    'darkslategrey': (47, 79, 79, 255),
    'dark slate grey': (47, 79, 79, 255),
    'darkturquoise': (0, 206, 209, 255),
    'dark turquoise': (0, 206, 209, 255),
    'darkviolet': (148, 0, 211, 255),
    'dark violet': (148, 0, 211, 255),
    'deeppink': (255, 20, 147, 255),
    'deep pink': (255, 20, 147, 255),
    'deepskyblue': (0, 191, 255, 255),
    'deep skyblue': (0, 191, 255, 255),
    'deep sky blue': (0, 191, 255, 255),
    'dimgray': (105, 105, 105, 255),
    'dim gray': (105, 105, 105, 255),
    'dimgrey': (105, 105, 105, 255),
    'dim grey': (105, 105, 105, 255),
    'dodgerblue': (30, 144, 255, 255),
    'dodger blue': (30, 144, 255, 255),
    'firebrick': (178, 34, 34, 255),
    'fire brick': (178, 34, 34, 255),
    'floralwhite': (255, 250, 240, 255),
    'floral white': (255, 250, 240, 255),
    'forestgreen': (34, 139, 34, 255),
    'forest green': (34, 139, 34, 255),
    'fuchsia': (255, 0, 255, 255),
    'gainsboro': (220, 220, 220, 255),
    'ghostwhite': (248, 248, 255, 255),
    'ghost white': (248, 248, 255, 255),
    'gold': (255, 215, 0, 255),
    'goldenrod': (218, 165, 32, 255),
    'gray': (128, 128, 128, 255),
    'green': (0, 128, 0, 255),
    'greenyellow': (173, 255, 47, 255),
    'green yellow': (173, 255, 47, 255),
    'grey': (128, 128, 128, 255),
    'honeydew': (240, 255, 240, 255),
    'hotpink': (255, 105, 180, 255),
    'hot pink': (255, 105, 180, 255),
    'indianred': (205, 92, 92, 255),
    'indian red': (205, 92, 92, 255),
    'indigo': (75, 0, 130, 255),
    'ivory': (255, 255, 240, 255),
    'khaki': (240, 230, 140, 255),
    'lavender': (230, 230, 250, 255),
    'lavenderblush': (255, 240, 245, 255),
    'lavender blush': (255, 240, 245, 255),
    'lawngreen': (124, 252, 0, 255),
    'lawn green': (124, 252, 0, 255),
    'lemonchiffon': (255, 250, 205, 255),
    'lemon chiffon': (255, 250, 205, 255),
    'lightblue': (173, 216, 230, 255),
    'light blue': (173, 216, 230, 255),
    'lightcoral': (240, 128, 128, 255),
    'light coral': (240, 128, 128, 255),
    'lightcyan': (224, 255, 255, 255),
    'light cyan': (224, 255, 255, 255),
    'lightgoldenrodyellow': (250, 250, 210, 255),
    'light goldenrod yellow': (250, 250, 210, 255),
    'lightgray': (211, 211, 211, 255),
    'light gray': (211, 211, 211, 255),
    'lightgreen': (144, 238, 144, 255),
    'light green': (144, 238, 144, 255),
    'lightgrey': (211, 211, 211, 255),
    'light grey': (211, 211, 211, 255),
    'lightpink': (255, 182, 193, 255),
    'light pink': (255, 182, 193, 255),
    'lightsalmon': (255, 160, 122, 255),
    'light salmon': (255, 160, 122, 255),
    'lightseagreen': (32, 178, 170, 255),
    'light seagreen': (32, 178, 170, 255),
    'light sea green': (32, 178, 170, 255),
    'lightskyblue': (135, 206, 250, 255),
    'light skyblue': (135, 206, 250, 255),
    'light sky blue': (135, 206, 250, 255),
    'lightslategray': (119, 136, 153, 255),
    'light slate gray': (119, 136, 153, 255),
    'lightslategrey': (119, 136, 153, 255),
    'light slate grey': (119, 136, 153, 255),
    'lightsteelblue': (176, 196, 222, 255),
    'light steel blue': (176, 196, 222, 255),
    'lightyellow': (255, 255, 224, 255),
    'light yellow': (255, 255, 224, 255),
    'lime': (0, 255, 0, 255),
    'limegreen': (50, 205, 50, 255),
    'lime green': (50, 205, 50, 255),
    'linen': (250, 240, 230, 255),
    'magenta': (255, 0, 255, 255),
    'maroon': (128, 0, 0, 255),
    'mediumaquamarine': (102, 205, 170, 255),
    'medium aquamarine': (102, 205, 170, 255),
    'mediumblue': (0, 0, 205, 255),
    'medium blue': (0, 0, 205, 255),
    'mediumorchid': (186, 85, 211, 255),
    'medium orchid': (186, 85, 211, 255),
    'mediumpurple': (147, 112, 219, 255),
    'medium purple': (147, 112, 219, 255),
    'mediumseagreen': (60, 179, 113, 255),
    'medium seagreen': (60, 179, 113, 255),
    'medium sea green': (60, 179, 113, 255),
    'mediumslateblue': (123, 104, 238, 255),
    'medium slate blue': (123, 104, 238, 255),
    'mediumspringgreen': (0, 250, 154, 255),
    'medium spring green': (0, 250, 154, 255),
    'mediumturquoise': (72, 209, 204, 255),
    'medium turquoise': (72, 209, 204, 255),
    'mediumvioletred': (199, 21, 133, 255),
    'medium violet red': (199, 21, 133, 255),
    'midnightblue': (25, 25, 112, 255),
    'midnight blue': (25, 25, 112, 255),
    'mintcream': (245, 255, 250, 255),
    'mint cream': (245, 255, 250, 255),
    'mistyrose': (255, 228, 225, 255),
    'misty rose': (255, 228, 225, 255),
    'moccasin': (255, 228, 181, 255),
    'navajowhite': (255, 222, 173, 255),
    'navajo white': (255, 222, 173, 255),
    'navy': (0, 0, 128, 255),
    'oldlace': (253, 245, 230, 255),
    'old lace': (253, 245, 230, 255),
    'olive': (128, 128, 0, 255),
    'olivedrab': (107, 142, 35, 255),
    'olive drab': (107, 142, 35, 255),
    'orange': (255, 165, 0, 255),
    'orangered': (255, 69, 0, 255),
    'orange red': (255, 69, 0, 255),
    'orchid': (218, 112, 214, 255),
    'palegoldenrod': (238, 232, 170, 255),
    'pale goldenrod': (238, 232, 170, 255),
    'palegreen': (152, 251, 152, 255),
    'pale green': (152, 251, 152, 255),
    'paleturquoise': (175, 238, 238, 255),
    'pale turquoise': (175, 238, 238, 255),
    'palevioletred': (219, 112, 147, 255),
    'pale violet red': (219, 112, 147, 255),
    'papayawhip': (255, 239, 213, 255),
    'papaya whip': (255, 239, 213, 255),
    'peachpuff': (255, 218, 185, 255),
    'peach puff': (255, 218, 185, 255),
    'peru': (205, 133, 63, 255),
    'pink': (255, 192, 203, 255),
    'plum': (221, 160, 221, 255),
    'powderblue': (176, 224, 230, 255),
    'powder blue': (176, 224, 230, 255),
    'purple': (128, 0, 128, 255),
    'rebeccapurple': (102, 51, 153, 255),
    'rebecca purple': (102, 51, 153, 255),
    'red': (255, 0, 0, 255),
    'rosybrown': (188, 143, 143, 255),
    'rosy brown': (188, 143, 143, 255),
    'royalblue': (65, 105, 225, 255),
    'royal blue': (65, 105, 225, 255),
    'saddlebrown': (139, 69, 19, 255),
    'saddle brown': (139, 69, 19, 255),
    'salmon': (250, 128, 114, 255),
    'sandybrown': (244, 164, 96, 255),
    'sandy brown': (244, 164, 96, 255),
    'seagreen': (46, 139, 87, 255),
    'sea green': (46, 139, 87, 255),
    'seashell': (255, 245, 238, 255),
    'sienna': (160, 82, 45, 255),
    'silver': (192, 192, 192, 255),
    'skyblue': (135, 206, 235, 255),
    'sky blue': (135, 206, 235, 255),
    'slateblue': (106, 90, 205, 255),
    'slate blue': (106, 90, 205, 255),
    'slategray': (112, 128, 144, 255),
    'slate gray': (112, 128, 144, 255),
    'slategrey': (112, 128, 144, 255),
    'slate grey': (112, 128, 144, 255),
    'snow': (255, 250, 250, 255),
    'springgreen': (0, 255, 127, 255),
    'spring green': (0, 255, 127, 255),
    'steelblue': (70, 130, 180, 255),
    'steel blue': (70, 130, 180, 255),
    'tan': (210, 180, 140, 255),
    'teal': (0, 128, 128, 255),
    'thistle': (216, 191, 216, 255),
    'tomato': (255, 99, 71, 255),
    'turquoise': (64, 224, 208, 255),
    'violet': (238, 130, 238, 255),
    'wheat': (245, 222, 179, 255),
    'white': (255, 255, 255, 255),
    'whitesmoke': (245, 245, 245, 255),
    'white smoke': (245, 245, 245, 255),
    'yellow': (255, 255, 0, 255),
    'yellowgreen': (154, 205, 50, 255),
    'yellow green': (154, 205, 50, 255),
})
BuiltinColors['transparent'] = (0, 0, 0, 0)


def most_common_color(colors):
    import numpy
    as32 = colors.view(numpy.int32).reshape((len(colors),))
    unique, indices, counts = numpy.unique(as32, return_index=True, return_counts=True)
    max_index = numpy.argmax(counts)
    if counts[max_index] < len(colors) / 10:
        return None
    return colors[indices[max_index]]

def rgba_to_rgba8(rgba):
    return tuple(int(255*r) for r in rgba)

def rgba8_to_rgba(rgba):
    return tuple(r/255.0 for r in rgba)

def _init():
    for name in BuiltinColors:
        rgb = BuiltinColors[name]
        color = Color([x / 255 for x in rgb], mutable=False)
        BuiltinColors[name] = color
_init()
