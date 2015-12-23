# vim: set expandtab shiftwidth=4 softtabstop=4:

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
        data = {name: color for name, color in self.items()
                if name not in BuiltinColors or color != BuiltinColors[name]}
        return CORE_STATE_VERSION, data

    def restore_snapshot_init(self, session, bundle_info, version, data):
        self.__init__()
        self.update(data)

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

    def take_snapshot(self, session, flags):
        data = self.rgba
        return CORE_STATE_VERSION, data

    def restore_snapshot_init(self, session, bundle_info, version, data):
        self.__init__(data, limit=False)

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
        return CORE_STATE_VERSION, dict(self)

    def restore_snapshot_init(self, session, bundle_info, version, data):
        self.__init__()
        self.update(data)

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
        from numpy import array, float32, ndarray, argsort
        if not data_values:
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
            color_above_value_range = colors[-1]
        if color_below_value_range is None:
            color_below_value_range = colors[0]
        if color_no_value is None:
            color_no_value = (.5, .5, .5, 1)

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
        from . import map
        colors = map.interpolate_colormap(values, self.data_values, self.colors,
                                          self.color_above_value_range,
                                          self.color_below_value_range)
        return colors


# Initialize built-in colormaps
# Rainbow is blue to red instead of red to blue so that N-terminus to C-terminus rainbow coloring
# produces the conventional blue to red.
BuiltinColormaps['rainbow'] = Colormap(None, ((0, 0, 1, 1), (0, 1, 1, 1), (0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1)))
BuiltinColormaps['grayscale'] = Colormap(None, ((0, 0, 0, 1), (1, 1, 1, 1)))
# BuiltinColormaps['red-white-blue'] = Colormap(None, ((1, 0, 0, 1), (1, 1, 1, 1), (0, 0, 1, 1)))
BuiltinColormaps['red-white-blue'] = Colormap(None, ((1, 0, 0, 1), (.7, .7, .7, 1), (0, 0, 1, 1)))
BuiltinColormaps['blue-white-red'] = Colormap(None, ((0, 0, 1, 1), (1, 1, 1, 1), (1, 0, 0, 1)))
BuiltinColormaps['cyan-white-maroon'] = Colormap(None, ((0.059, 0.78, 0.81, 1), (1, 1, 1, 1), (0.62, 0.125, 0.37, 1)))
# Add some aliases
BuiltinColormaps['redblue'] = BuiltinColormaps['red-white-blue']
BuiltinColormaps['bluered'] = BuiltinColormaps['blue-white-red']
BuiltinColormaps['gray'] = BuiltinColormaps['grayscale']
BuiltinColormaps['cyanmaroon'] = BuiltinColormaps['cyan-white-maroon']


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
        ec[:, :3] = 180
        ec[:, 3] = 255
        # jmol element colors
        colors = (
            (1, (255, 255, 255)),   # H
            (2, (217, 255, 255)),   # He
            (3, (204, 128, 255)),   # Li
            (4, (194, 255, 0)),     # Be
            (5, (255, 181, 181)),   # B
            (6, (144, 144, 144)),   # C
            (7, (48, 80, 248)),     # N
            (8, (255, 13, 13)),     # O
            (9, (144, 224, 80)),    # F
            (10, (179, 227, 245)),  # Ne
            (11, (171, 92, 242)),   # Na
            (12, (138, 255, 0)),    # Mg
            (13, (191, 166, 166)),  # Al
            (14, (240, 200, 160)),  # Si
            (15, (255, 128, 0)),    # P
            (16, (255, 255, 48)),   # S
            (17, (31, 240, 31)),    # Cl
            (18, (128, 209, 227)),  # Ar
            (19, (143, 64, 212)),   # K
            (20, (61, 255, 0)),     # Ca
            (21, (230, 230, 230)),  # Sc
            (22, (191, 194, 199)),  # Ti
            (23, (166, 166, 171)),  # V
            (24, (138, 153, 199)),  # Cr
            (25, (156, 122, 199)),  # Mn
            (26, (224, 102, 51)),   # Fe
            (27, (240, 144, 160)),  # Co
            (28, (80, 208, 80)),    # Ni
            (29, (200, 128, 51)),   # Cu
            (30, (125, 128, 176)),  # Zn
            (31, (194, 143, 143)),  # Ga
            (32, (102, 143, 143)),  # Ge
            (33, (189, 128, 227)),  # As
            (34, (255, 161, 0)),    # Se
            (35, (166, 41, 41)),    # Br
            (36, (92, 184, 209)),   # Kr
            (37, (112, 46, 176)),   # Rb
            (38, (0, 255, 0)),      # Sr
            (39, (148, 255, 255)),  # Y
            (40, (148, 224, 224)),  # Zr
            (41, (115, 194, 201)),  # Nb
            (42, (84, 181, 181)),   # Mo
            (43, (59, 158, 158)),   # Tc
            (44, (36, 143, 143)),   # Ru
            (45, (10, 125, 140)),   # Rh
            (46, (0, 105, 133)),    # Pd
            (47, (192, 192, 192)),  # Ag
            (48, (255, 217, 143)),  # Cd
            (49, (166, 117, 115)),  # In
            (50, (102, 128, 128)),  # Sn
            (51, (158, 99, 181)),   # Sb
            (52, (212, 122, 0)),    # Te
            (53, (148, 0, 148)),    # I
            (54, (66, 158, 176)),   # Xe
            (55, (87, 23, 143)),    # Cs
            (56, (0, 201, 0)),      # Ba
            (57, (112, 212, 255)),  # La
            (58, (255, 255, 199)),  # Ce
            (59, (217, 255, 199)),  # Pr
            (60, (199, 255, 199)),  # Nd
            (61, (163, 255, 199)),  # Pm
            (62, (143, 255, 199)),  # Sm
            (63, (97, 255, 199)),   # Eu
            (64, (69, 255, 199)),   # Gd
            (65, (48, 255, 199)),   # Tb
            (66, (31, 255, 199)),   # Dy
            (67, (0, 255, 156)),    # Ho
            (68, (0, 230, 117)),    # Er
            (69, (0, 212, 82)),     # Tm
            (70, (0, 191, 56)),     # Yb
            (71, (0, 171, 36)),     # Lu
            (72, (77, 194, 255)),   # Hf
            (73, (77, 166, 255)),   # Ta
            (74, (33, 148, 214)),   # W
            (75, (38, 125, 171)),   # Re
            (76, (38, 102, 150)),   # Os
            (77, (23, 84, 135)),    # Ir
            (78, (208, 208, 224)),  # Pt
            (79, (255, 209, 35)),   # Au
            (80, (184, 184, 208)),  # Hg
            (81, (166, 84, 77)),    # Tl
            (82, (87, 89, 97)),     # Pb
            (83, (158, 79, 181)),   # Bi
            (84, (171, 92, 0)),     # Po
            (85, (117, 79, 69)),    # At
            (86, (66, 130, 150)),   # Rn
            (87, (66, 0, 102)),     # Fr
            (88, (0, 125, 0)),      # Ra
            (89, (112, 171, 250)),  # Ac
            (90, (0, 186, 255)),    # Th
            (91, (0, 161, 255)),    # Pa
            (92, (0, 143, 255)),    # U
            (93, (0, 128, 255)),    # Np
            (94, (0, 107, 255)),    # Pu
            (95, (84, 92, 242)),    # Am
            (96, (120, 92, 227)),   # Cm
            (97, (138, 79, 227)),   # Bk
            (98, (161, 54, 212)),   # Cf
            (99, (179, 31, 212)),   # Es
            (100, (179, 31, 186)),  # Fm
            (101, (179, 13, 166)),  # Md
            (102, (189, 13, 135)),  # No
            (103, (199, 0, 102)),   # Lr
            (104, (204, 0, 89)),    # Rf
            (105, (209, 0, 79)),    # Db
            (106, (217, 0, 69)),    # Sg
            (107, (224, 0, 56)),    # Bh
            (108, (230, 0, 46)),    # Hs
            (109, (235, 0, 38)),    # Mt
        )
        for e, rgb in colors:
            ec[e, :3] = rgb

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
            'a': (123, 104, 238, 255),
            'b': (240, 128, 128, 255),
            'c': (143, 188, 143, 255),
            'd': (222, 184, 135, 255),
            'e': (255, 127, 80, 255),
            'f': (128, 128, 128, 255),
            'g': (107, 142, 35, 255),
            'h': (100, 100, 100, 255),
            'i': (255, 255, 0, 255),
            'j': (55, 19, 112, 255),
            'k': (255, 255, 150, 255),
            'l': (202, 62, 94, 255),
            'm': (205, 145, 63, 255),
            'n': (12, 75, 100, 255),
            'o': (255, 0, 0, 255),
            'p': (175, 155, 50, 255),
            'q': (105, 205, 48, 255),
            'r': (37, 70, 25, 255),
            's': (121, 33, 135, 255),
            't': (83, 140, 208, 255),
            'u': (0, 154, 37, 255),
            'v': (178, 220, 205, 255),
            'w': (255, 152, 213, 255),
            'x': (200, 90, 174, 255),
            'y': (175, 200, 74, 255),
            'z': (63, 25, 12, 255),
            '1': (87, 87, 87, 255),
            '2': (173, 35, 35, 255),
            '3': (42, 75, 215, 255),
            '4': (29, 105, 20, 255),
            '5': (129, 74, 25, 255),
            '6': (129, 38, 192, 255),
            '7': (160, 160, 160, 255),
            '8': (129, 197, 122, 255),
            '9': (157, 175, 255, 255),
            '0': (41, 208, 208, 255),
        }

    for cid in set(cids):
        c = str(cid).lower()
        if c not in rgba_256:
            from random import randint, seed
            seed(c)
            rgba_256[c] = (randint(128, 255), randint(128, 255), randint(128, 255), 255)

    from numpy import array, uint8, empty
    if len(cids) == 0:
        c = empty((0, 4), uint8)
    else:
        c = array(tuple(rgba_256[cid.lower()] for cid in cids), uint8)
    return c


# -----------------------------------------------------------------------------
#
def chain_rgba(cid):
    return tuple(float(c / 255.0) for c in chain_colors([cid])[0])


# -----------------------------------------------------------------------------
#
def chain_rgba8(cid):
    return chain_colors([cid])[0]


_df_state = {}


def distinguish_from(rgbs, *, num_candidates=3, seed=None, save_state=True):
    """Best effort to return an RGB that perceptually differs from the given RGBs"""
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


def _init():
    for name in BuiltinColors:
        rgb = BuiltinColors[name]
        color = Color([x / 255 for x in rgb])
        BuiltinColors[name] = color
_init()
