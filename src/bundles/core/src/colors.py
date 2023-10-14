# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
color: basic color support
===========================

A :py:class:`Color` class that hold a :py:class:`numpy.array` of
four 32-bit floats.

CSS3 colors are supported with the addition of the gray() specification
from the CSS4 color draft, https://www.w3.org/TR/css-color-4/, and CSS4
color names.
"""
from sortedcontainers import SortedDict
from .state import State, StateManager

# If any of the *STATE_VERSIONs change, then increase the (maximum) core session
# number in setup.py.in
COLOR_STATE_VERSION = 1
USER_COLORS_STATE_VERSION = 1
COLORMAP_STATE_VERSION = 1
USER_COLORMAPS_STATE_VERSION = 1

class UserColors(SortedDict, StateManager):
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
        data = {
            'colors': cmap,
            'version': USER_COLORS_STATE_VERSION,
        }
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
            from .errors import UserError
            raise UserError('Can not override builtin color')
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
        or # followed by 3 (4), 6 (8), or 12 (16) hex digits (with alpha),
        or built-in color name.
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
            if rgba.startswith('#'):
                # Hex: #DDD, #DDDDDD, or #DDDDDDDDDDDD
                try:
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
                try:
                    self.rgba = BuiltinColors[rgba].rgba[:]  # copy
                except KeyError:
                    raise ValueError("No built-in color named %s" % rgba)
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
        data = {
            'rgba': self.rgba,
            'version': COLOR_STATE_VERSION,
        }
        return data

    @staticmethod
    def restore_snapshot(session, data):
        return Color(data['rgba'], limit=False)

    def opaque(self):
        """Return if the color is opaque."""
        return self.rgba[3] >= 1.0

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(list(self.rgba)))

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


class UserColormaps(SortedDict, StateManager):
    """Support for per-session colormaps.

    Accessed through the session object as ``session.user_colormaps``.
    """

    def take_snapshot(self, session, flags):
        data = {
            'colormaps': dict(self),
            'version': USER_COLORMAPS_STATE_VERSION,
        }
        return data

    @staticmethod
    def restore_snapshot(session, data):
        c = UserColormaps()
        c.update(data['colormaps'])
        return c

    def reset_state(self, session):
        """Reset state to data-less state"""
        self.clear()


class Colormap(State):
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
                 color_no_value=None, name=None):
        self.name = name
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

        self.is_transparent = not (c[:,3] == 1).all()
        
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
        from chimerax import map
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

    def rescale_range(self, value0, value1, full = False):
        '''
        Return new colormap with [0,1] range becoming [value0,value1].
        Or if full is true rescale current range to new range.
        '''
        v = self.data_values.copy()
        if full:
            cur0, cur1 = self.value_range()
            v -= cur0
            v *= (value1 - value0) / (cur1 - cur0)
            v += value0
        else:
            v *= (value1 - value0)
            v += value0
        cmap = Colormap(v, self.colors,
                        self.color_above_value_range,
                        self.color_below_value_range,
                        self.color_no_value)
        return cmap

    def reversed(self):
        """Return a reversed color ramp.

        Return Value
        ------------
        instance of Colormap
        """
        cmap = Colormap(self.data_values, self.colors[::-1],
                        self.color_below_value_range,
                        self.color_above_value_range,
                        self.color_no_value)
        cmap.values_specified = self.values_specified
        return cmap

    # -------------------------------------------------------------------------
    #
    def take_snapshot(self, session, flags):
        data = {
            'name': self.name,
            'values_specified': self.values_specified,
            'data_values': self.data_values,
            'colors': self.colors,
            'color_above_value_range': self.color_above_value_range,
            'color_below_value_range': self.color_below_value_range,
            'color_no_value': self.color_no_value,
            'version': COLORMAP_STATE_VERSION,
        }
        return data

    # -------------------------------------------------------------------------
    #
    @classmethod
    def restore_snapshot(cls, session, data):
        c = Colormap(data['data_values'], data['colors'],
                     color_above_value_range = data['color_above_value_range'],
                     color_below_value_range = data['color_below_value_range'],
                     color_no_value = data['color_no_value'],
                     name = data['name'])
        c.values_specified = data['values_specified']
        return c

def colormap_with_range(colormap, range, default_colormap_name = 'red-blue', full_range = (0,1)):
    if colormap is None:
        colormap = BuiltinColormaps[default_colormap_name]
    if range == 'full':
        range = full_range
    if range is None:
        cmap = colormap if colormap.values_specified else colormap.rescale_range(*full_range)
        return cmap
    vmin, vmax = range
    if colormap.values_specified:
        cmap = colormap.rescale_range(vmin, vmax, full = True)
    else:
        cmap = colormap.linear_range(vmin, vmax)
    return cmap

def _builtin_colormaps():
    '''Define built-in colormaps'''

    cmaps = SortedDict()
    # Rainbow is blue to red instead of red to blue so that N-terminus to C-terminus rainbow coloring
    # produces the conventional blue to red.
    cmaps['rainbow'] = Colormap(None, ((0, 0, 1, 1), (0, 1, 1, 1), (0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1)))
    cmaps['grayscale'] = Colormap(None, ((0, 0, 0, 1), (1, 1, 1, 1)))
    cmaps['red-white-blue'] = Colormap(None, ((1, 0, 0, 1), (1, 1, 1, 1), (0, 0, 1, 1)))
    cmaps['blue-white-red'] = Colormap(None, ((0, 0, 1, 1), (1, 1, 1, 1), (1, 0, 0, 1)))
    cmaps['cyan-white-maroon'] = Colormap(None, ((0.059, 0.78, 0.81, 1), (1, 1, 1, 1), (0.62, 0.125, 0.37, 1)))
    cmaps['cyan-gray-maroon'] = Colormap(None, ((0.059, 0.78, 0.81, 1), (.7, .7, .7, 1), (0.62, 0.125, 0.37, 1)))
    cmaps['lipophilicity'] = Colormap(None, ((0, 139 / 255, 139 / 255, 1), (1, 1, 1, 1), (184 / 255, 134 / 255, 11 / 255, 1)))  # dark cyan, white, dark goldenrod
    _alphafold_colors = [BuiltinColors[name] for name in
                         ('red', 'orange', 'yellow', 'cornflowerblue', 'blue')]
    cmaps['alphafold'] = Colormap((0, 50, 70, 90, 100), _alphafold_colors)
    cmaps['esmfold'] = Colormap((0, 0.5, 0.7, 0.9, 1.0), _alphafold_colors)
    _pae_colors = [BuiltinColors[name] for name in
                         ('blue', 'cornflowerblue', 'yellow', 'orange', 'gray', 'lightgray', 'white')]
    cmaps['pae'] = Colormap((0, 5, 10, 15, 20, 25, 30), _pae_colors)
    _pae_green_colors = ((0.118,0.275,0.118,1), (0.142,0.571,0.142,1), (0.216,0.693,0.216,1), (0.338,0.788,0.338,1), (0.510,0.867,0.510,1), (0.730,0.937,0.730,1), (1.000,1.000,1.000,1))
    cmaps['paegreen'] = Colormap((0, 5, 10, 15, 20, 25, 30), _pae_green_colors)
    _pae_contacts_colors = [BuiltinColors[name] for name in
                            ('blue', 'cornflowerblue', 'yellow', 'orange', 'red')]
    cmaps['paecontacts'] = Colormap((0, 5, 10, 15, 20), _pae_contacts_colors)

    # Add some aliases
    cmaps['redblue'] = cmaps['red-white-blue']
    cmaps['bluered'] = cmaps['blue-white-red']
    cmaps['gray'] = cmaps['grayscale']
    cmaps['cyanmaroon'] = cmaps['cyan-white-maroon']

    _read_colorbrewer(cmaps)

    return cmaps

# Add colorbrewer palettes
def _read_colorbrewer(colormaps):
    import json
    import os.path
    my_dir = os.path.dirname(__file__)
    # colorbrewer.json is downloaded from
    # http://colorbrewer2.org/export/colorbrewer.json
    brewer_filename = os.path.join(my_dir, "colorbrewer.json")
    try:
        with open(brewer_filename) as f:
            brewer_maps = json.load(f)
    except IOError as e:
        print("%s: %s" % (brewer_filename, str(e)), flush=True)
    else:
        def rgb(r, g, b):
            return (r / 255, g / 255, b / 255, 1)
        gs = {'__builtins__': None, 'rgb': rgb}
        ls = {}
        for scheme, ramps in brewer_maps.items():
            s_type = ramps.pop("type")
            for count, rgbs in ramps.items():
                name = "%s-%s" % (scheme, count)
                colors = tuple([eval(e, gs, ls) for e in rgbs])
                colormaps[name.casefold()] = Colormap(None, colors, name=name)
                if ((s_type == "div" and count == "5") or
                        (s_type == "seq" and count == "5") or
                        (s_type == "qual" and count == "6")):
                    colormaps[scheme.casefold()] = Colormap(None, colors, name=scheme)


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
    if rgb[0] * 0.59 + rgb[1] < 0.826:
        return (1.0, 1.0, 1.0)
    return (0.0, 0.0, 0.0)

def contrast_with_background(session):
    """Contrast with the graphics-window background color"""
    return contrast_with(session.main_view.background_color)

def _builtin_colors():
    # CSS4 colors + multiword color names
    colors = SortedDict({
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
    colors['transparent'] = (0, 0, 0, 0)

    ctable = {}
    for name, rgba in colors.items():
        color = Color([x / 255 for x in rgba], mutable=False)
        color.color_name = name
        ctable[name] = color
    return ctable

def random_colors(n, opacity=255, seed=None):
    from numpy import random, uint8
    if seed is not None:
        random.seed(seed)
    c = random.randint(0, 255, (n,4), dtype = uint8)
    c[:,3] = opacity
    return c

def most_common_color(colors):
    from numpy import ndarray, array, uint8, int32, argmax, unique
    if not isinstance(colors, ndarray):
        colors = array(colors, uint8)
    as32 = colors.view(int32).reshape((len(colors),))
    unique, indices, counts = unique(as32, return_index=True, return_counts=True)
    max_index = argmax(counts)
    if counts[max_index] < len(colors) / 10:
        return None
    return colors[indices[max_index]]

_color_names = None
def color_name(color_or_rgba8, *, always_include_hex_alpha=False):
    '''Return english color name or hex color string.'''
    global _color_names
    if _color_names is None:
        _color_names = {tuple(color.uint8x4()):name
                        for name, color in BuiltinColors.items()}
    if isinstance(color_or_rgba8, Color):
        rgba8 = color_or_rgba8.uint8x4()
    else:
        rgba8 = color_or_rgba8
    c = tuple(rgba8)
    if c in _color_names:
        name = _color_names[c]
    else:
        name = hex_color(c, always_include_alpha=always_include_hex_alpha)
    return name

def palette_equal(p1, p2, *, tolerance=1/512):
    if len(p1) != len(p2):
        return False
    def len4(c):
        if len(c) == 4:
            return c
        else:
            return [x for x in c] + [1.0]
    for c1, c2 in zip(p1, p2):
        for v1, v2 in zip(len4(c1), len4(c2)):
            if abs(v1 - v2) > tolerance:
                return False
    return True

def palette_name(rgbas, *, tolerance=1/512):
    for name, cm in BuiltinColormaps.items():
        if palette_equal(cm.colors, rgbas, tolerance=tolerance):
            return name
    # reversed palettes
    for name, cm in BuiltinColormaps.items():
        if palette_equal(cm.colors, list(reversed(rgbas)), tolerance=tolerance):
            return '^' + name
    return None

def hex_color(rgba8, *, always_include_alpha=False):
    return ('#%02x%02x%02x' % tuple(rgba8[:3])) if rgba8[3] == 255 and not always_include_alpha else (
        '#%02x%02x%02x%02x' % tuple(rgba8))

def rgba_to_rgba8(rgba):
    return tuple(int(255 * r + 0.5) for r in rgba)


def rgba8_to_rgba(rgba):
    return tuple(r / 255.0 for r in rgba)


def rgb_to_hls(rgba):
    """Convert array of RGB(A) values to HLS

    Alpha component is optional and ignored.
    Uses same conventions as colorsys module,
    i.e., all components are floating point (0-1)"""
    from numpy import zeros, float32, amin, amax, remainder

    min = amin(rgba[:, 0:3], axis=1)
    max = amax(rgba[:, 0:3], axis=1)
    delta = max - min

    hls = zeros((len(rgba), 3), dtype=float32)

    # lightness
    hls[:, 1] = (min + max) / 2

    # saturation
    chromatic_mask = delta != 0
    mask = chromatic_mask & (hls[:, 1] <= 0.5)
    hls[mask, 2] = delta[mask] / hls[mask, 1] / 2
    mask = chromatic_mask & (hls[:, 1] > 0.5)
    hls[mask, 2] = delta[mask] / (2 - 2 * hls[mask, 1])

    # hue
    r, g, b = rgba[:, 0], rgba[:, 1], rgba[:, 2]
    mask = chromatic_mask & (b == max)
    hls[mask, 0] = 4 + (r[mask] - g[mask]) / delta[mask]
    mask = chromatic_mask & (g == max)
    hls[mask, 0] = 2 + (b[mask] - r[mask]) / delta[mask]
    mask = chromatic_mask & (r == max)
    hls[mask, 0] = (g[mask] - b[mask]) / delta[mask]

    hls[chromatic_mask, 0] = remainder(hls[chromatic_mask, 0], 6) / 6
    return hls


def _values(m1, m2, hue):
    from numpy import empty, remainder
    hue = remainder(hue, 1)
    values = empty(len(hue))

    mask = hue < (1 / 6)
    values[mask] = m1[mask] + (m2[mask] - m1[mask]) * hue[mask] * 6
    mask = (hue >= (1 / 6)) & (hue < .5)
    values[mask] = m2[mask]
    mask = (hue >= .5) & (hue < (2 / 3))
    values[mask] = m1[mask] + (m2[mask] - m1[mask]) * (2 / 3 - hue[mask]) * 6
    mask = hue >= (2 / 3)
    values[mask] = m1[mask]
    return values


def hls_to_rgb(hls):
    """Convert array of HLS values to RGB

    Uses same conventions as colorsys module,
    i.e., all components are floating point (0-1)"""
    from numpy import zeros, empty, newaxis

    rgb = zeros((len(hls), 3))

    # grays
    mask = hls[:, 2] == 0
    rgb[mask] = hls[mask, 1][:, newaxis]

    cmask = hls[:, 2] != 0  # chromatic mask
    m1 = empty(len(hls))
    m2 = empty(len(hls))
    mask = cmask & (hls[:, 1] <= 0.5)
    m2[mask] = hls[mask, 1] * (1 + hls[mask, 2])
    mask = cmask & (hls[:, 1] > 0.5)
    m2[mask] = hls[mask, 1] + hls[mask, 2] - hls[mask, 1] * hls[mask, 2]
    m1[cmask] = 2 * hls[cmask, 1] - m2[cmask]
    rgb[cmask, 0] = _values(m1[cmask], m2[cmask], hls[cmask, 0] + 1 / 3)
    rgb[cmask, 1] = _values(m1[cmask], m2[cmask], hls[cmask, 0])
    rgb[cmask, 2] = _values(m1[cmask], m2[cmask], hls[cmask, 0] - 1 / 3)
    return rgb


def rgb_to_hwb(rgba):
    """Convert array of RGB(A) values to HWB

    Alpha component is optional and ignored.
    Uses same conventions as colorsys module,
    i.e., all components are floating point (0-1)"""
    from numpy import amin, amax
    hwb = rgb_to_hls(rgba)  # get hue from hls for now
    min = amin(rgba[:, 0:3], axis=1)
    max = amax(rgba[:, 0:3], axis=1)
    hwb[:, 1] = min
    hwb[:, 2] = 1 - max
    return hwb


def hwb_to_rgb2(hwb):
    """Convert HWB values to RGB -- alternate (slower) version"""
    from numpy import empty, newaxis
    hls = empty((len(hwb), 3))
    hls[:, 0] = hwb[:, 0]
    hls[:, 1:3] = [.5, 1]
    rgb = hls_to_rgb(hls)
    rgb = rgb * (1 - hwb[:, 1] - hwb[:, 2])[:, newaxis] + hwb[:, 1][:, newaxis]
    return rgb


def hwb_to_rgb(hwb):
    """Convert HWB values to RGB

    Uses same conventions as colorsys module,
    i.e., all components are floating point (0-1)"""
    from numpy import empty, remainder, floor, newaxis, clip
    rgb = empty((len(hwb), 3))

    max = 1 - hwb[:, 2]
    # normalize hwb as per https://www.w3.org/TR/css-color-4/#the-hwb-notation
    ratio = hwb[:, 1] + hwb[:, 2]
    mask = ratio > 1
    hwb[mask, 1:3] /= ratio[mask, newaxis]

    i = remainder(floor(hwb[:, 0] * 6), 6)
    max = 1 - hwb[:, 2]
    f = 6 * hwb[:, 0] - i
    mask = (i % 2) != 0
    f[mask] = 1 - f[mask]

    n = hwb[:, 1] + f * (max - hwb[:, 1])

    mask = i == 0
    rgb[mask, 0] = max[mask]
    rgb[mask, 1] = n[mask]
    rgb[mask, 2] = hwb[mask, 1]
    mask = i == 1
    rgb[mask, 0] = n[mask]
    rgb[mask, 1] = max[mask]
    rgb[mask, 2] = hwb[mask, 1]
    mask = i == 2
    rgb[mask, 0] = hwb[mask, 1]
    rgb[mask, 1] = max[mask]
    rgb[mask, 2] = n[mask]
    mask = i == 3
    rgb[mask, 0] = hwb[mask, 1]
    rgb[mask, 1] = n[mask]
    rgb[mask, 2] = max[mask]
    mask = i == 4
    rgb[mask, 0] = n[mask]
    rgb[mask, 1] = hwb[mask, 1]
    rgb[mask, 2] = max[mask]
    mask = i == 5
    rgb[mask, 0] = max[mask]
    rgb[mask, 1] = hwb[mask, 1]
    rgb[mask, 2] = n[mask]

    return rgb


def luminance(rgba):
    """Compute luminance assuming sRGB display

    Alpha component is optional and ignored.
    Uses same conventions as colorsys module,
    i.e., all components are floating point (0-1)"""
    # from https://www.w3.org/TR/css-color-4/#contrast-adjuster
    if rgba.shape[1] == 4:
        rgb = rgba[:, 0:3]
    else:
        rgb = rgba
    # first convert to linear sRGB
    mask = rgb < 0.04045
    rgb[mask] = rgb[mask] / 12.92
    rgb[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4
    # then compute luminance
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance

from . import configfile
class ColorValue(configfile.Value):
    """Class to use in settings when the setting is a Color"""

    def convert_from_string(self, session, str_value):
        from chimerax.core.commands import ColorArg
        value, consumed, rest = ColorArg.parse(str_value, session)
        return value

    def convert_to_string(self, session, value):
        if not isinstance(value, Color):
            try:
                value = Color(value)
            except Exception as e:
                raise ValueError("Cannot convert %s to Color instance: %s" % (repr(value), str(e)))
        from chimerax.core.commands import ColorArg
        str_value = ColorArg.unparse(value, session)
        # confirm that value can be restored from disk,
        # by converting to a string and back
        new_value = self.convert_from_string(session, str_value)
        if new_value != value:
            raise ValueError('value changed while saving it')
        return str_value

BuiltinColors = _builtin_colors()
BuiltinColormaps = _builtin_colormaps()

