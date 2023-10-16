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

import re
from ..colors import Color
from . import Annotation, AnnotationError, next_token, Or, TupleOf, FloatArg, EnumOf


class ColorArg(Annotation):
    """Support color names and CSS3 color specifications.

    The CSS3 color specifications supported are: rgb, rgba, hsl, hsla, and
    gray from CSS4.

    The following examples are all ``red``, except for the gray ones::

        red
        #f00
        #ff0000
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
        if not text:
            raise AnnotationError("Missing color name or specifier")
        if text[0] == '#':
            token, text, rest = next_token(text)
            c = Color(token)
            c.explicit_transparency = (len(token) in (5, 9, 17))
            return c, text, rest
        if text[0].isdigit():
            token, text, rest = next_token(text)
            c = _parse_rgba_values(token)
            return c, text, rest
        m = _color_func.match(text)
        if m is None:
            color = None
            if session is not None:
                name, color, rest = find_named_color(session.user_colors, text)
            else:
                from ..colors import BuiltinColors
                name, color, rest = find_named_color(BuiltinColors, text)
            if color is None:
                raise AnnotationError("Invalid color name or specifier")
            return color, name, rest
        color_space = m.group(1)
        numbers = _parse_numbers(m.group(2))
        rest = text[m.end():]
        if color_space == 'gray' and len(numbers) in (1, 2):
            # gray( number [%], [ number [%] ])
            try:
                x = _convert_number(numbers[0], 'gray scale')
                if len(numbers) == 2:
                    alpha = _convert_number(numbers[1], 'alpha', maximum=1)
                else:
                    alpha = 1
            except AnnotationError as err:
                err.offset += m.end(1)
                raise
            c = Color([x, x, x, alpha])
            c.explicit_transparency = (len(numbers) == 2)
            return c, m.group(), rest
        if color_space == 'rgb' and len(numbers) == 3:
            # rgb( number [%], number [%], number [%])
            try:
                red = _convert_number(numbers[0], 'red')
                green = _convert_number(numbers[1], 'green')
                blue = _convert_number(numbers[2], 'blue')
            except AnnotationError as err:
                err.offset += m.end(1)
                raise
            c = Color([red, green, blue, 1])
            c.explicit_transparency = False
            return c, m.group(), rest
        if color_space == 'rgba' and len(numbers) == 4:
            # rgba( number [%], number [%], number [%], number [%])
            try:
                red = _convert_number(numbers[0], 'red')
                green = _convert_number(numbers[1], 'green')
                blue = _convert_number(numbers[2], 'blue')
                alpha = _convert_number(numbers[3], 'alpha', maximum=1)
            except AnnotationError as err:
                err.offset += m.end(1)
                raise
            c = Color([red, green, blue, alpha])
            c.explicit_transparency = True
            return c, m.group(), rest
        if color_space == 'hsl' and len(numbers) == 3:
            # hsl( number [%], number [%], number [%])
            try:
                hue = _convert_angle(numbers[0], 'hue angle')
                sat = _convert_number(numbers[1], 'saturation', maximum=1)
                light = _convert_number(numbers[2], 'lightness', maximum=1)
            except AnnotationError as err:
                err.offset += m.end(1)
                raise
            import colorsys
            red, green, blue = colorsys.hls_to_rgb(hue, light, sat)
            c = Color([red, green, blue, 1])
            c.explicit_transparency = False
            return c, m.group(), rest
        if color_space == 'hsla' and len(numbers) == 4:
            # hsla( number [%], number [%], number [%], number [%])
            try:
                hue = _convert_angle(numbers[0], 'hue angle')
                sat = _convert_number(numbers[1], 'saturation', maximum=1)
                light = _convert_number(numbers[2], 'lightness', maximum=1)
                alpha = _convert_number(numbers[3], 'alpha', maximum=1)
            except AnnotationError as err:
                err.offset += m.end(1)
                raise
            import colorsys
            red, green, blue = colorsys.hls_to_rgb(hue, light, sat)
            c = Color([red, green, blue, alpha])
            c.explicit_transparency = True
            return c, m.group(), rest
        raise AnnotationError(
            "Wrong number of components for %s specifier" % color_space,
            offset=m.end())

    @staticmethod
    def unparse(color, session):
        for name, c in session.user_colors.items():
            if color is c:
                return name
        if color.opaque():
            return color.hex()
        return color.hex_with_alpha()


def _parse_rgba_values(text):
    values = text.split(',')
    if len(values) not in (3, 4):
        raise AnnotationError('Color must be 3 or 4 comma-separated numbers 0-100')
    try:
        rgba = tuple(float(v) / 100.0 for v in values)
    except Exception:
        raise AnnotationError('Color must be 3 or 4 comma-separated numbers 0-100')
    transparent = (len(rgba) == 4)
    if len(rgba) == 3:
        rgba += (1.0,)
    c = Color(rgba)
    c.explicit_transparency = transparent
    return c


class Color8Arg(ColorArg):
    @staticmethod
    def parse(text, session):
        c, text, rest = ColorArg.parse(text, session)
        return c.uint8x4(), text, rest


class Color8TupleArg(ColorArg):
    @staticmethod
    def parse(text, session):
        c, text, rest = ColorArg.parse(text, session)
        return tuple(c.uint8x4()), text, rest


ColormapRangeArg = Or(TupleOf(FloatArg, 2), EnumOf(['full']))


class ColormapArg(Annotation):
    """Support color map names and value-color pairs specifications.

    Accepts name of a standard color map::

        rainbow
        grayscale, gray
        red-white-blue, redblue,
        blue-white-red, bluered
        cyan-white-maroon, cyanmaroon

    Or a custom color map can be specified as colon-separated colors, or as colon-separated
    (value, color) pairs with values ranging from 0 to 1.

    Example colormap specifications::

        grayscale
        orange:tan:green:yellow
        0,purple:.49,khaki:.5,beige:1,blue

    """
    name = 'a colormap'

    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        if token[0] == "^":
            reversed = True
            token = token[1:]
        else:
            reversed = False
        parts = token.split(':')
        cmap = None
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
                    # More than one comma
                    # Handle RGB color spec with commas
                    try:
                        color, t, r = ColorArg.parse(p, session)
                    except Exception:
                        val, col = p.split(',', maxsplit=1)
                        try:
                            values.append(float(val))
                            color, t, r = ColorArg.parse(col, session)
                        except Exception:
                            raise ValueError("Could not parse colormap color %s" % p)
                if r:
                    raise ValueError("Bad color in colormap")
                colors.append(color)
            if len(values) != len(colors) and len(values) > 0:
                raise ValueError("Number of values and color must match in colormap")
            if len(values) == 0:
                values = None
            from ..colors import Colormap
            consumed = f'^{text}' if reversed else text
            cmap = Colormap(values, [c.rgba for c in colors])
        else:
            ci_token = token.casefold()
            if session is not None:
                name = _name_completion(session.user_colormaps, ci_token)
                if name:
                    consumed = f'^{name}' if reversed else name
                    cmap = session.user_colormaps[name]
            if cmap is None:
                from ..colors import BuiltinColormaps
                name = _name_completion(BuiltinColormaps, ci_token)
                if name:
                    consumed = f'^{name}' if reversed else name
                    cmap = BuiltinColormaps[name]
            if cmap is None and session is not None:
                consumed = f'^{text}' if reversed else text
                cmap = _fetch_colormap(session, palette_id=token)
        if cmap is None:
            from ..errors import UserError
            raise UserError("Cannot find palette named %r" % token)
        return cmap.reversed() if reversed else cmap, consumed, rest


def _name_completion(sorted_dict, prefix, all_possible=False):
    # find best match(es) for prefix
    # Use fact that Colors and ColorMaps are subclassed from SortedDict
    num_names = len(sorted_dict)
    i = sorted_dict.bisect_left(prefix)
    if i >= num_names:
        return [] if all_possible else None
    all_names = sorted_dict.keys()
    if not all_possible:
        name = all_names[i]
        if name.startswith(prefix):
            return name
        return None
    matches = []
    for i in range(i, num_names):
        name = all_names[i]
        if not name.startswith(prefix):
            break
        matches.append(name)
    return matches


def find_named_color(color_dict, name):
    # handle color names with spaces
    # returns key, value, part of name that was unused
    if len(color_dict) == 0:
        return None, None, name
    # extract up to 10 words from name
    from . import cli
    first = True
    if name[0] == '"':
        m = cli._double_quote.match(name)
        if not m:
            raise AnnotationError("incomplete quoted text")
        end = m.end()
        if name[end - 1].isspace():
            end -= 1
        text = name[1:end - 1]
        quoted = name[end:]
    elif name[0] == "'":
        quoted = True
        m = cli._single_quote.match(name)
        if not m:
            raise AnnotationError("incomplete quoted text")
        end = m.end()
        if name[end - 1].isspace():
            end -= 1
        text = name[1:end - 1]
        quoted = name[end:]
    else:
        quoted = None
        text = name
    words = []
    while len(words) < 10:
        m = cli._whitespace.match(text)
        text = text[m.end():]
        if not text:
            break
        word, _, rest = next_token(text)
        if not word or word == ';':
            break
        if first and ' ' in word:
            words = [(w, rest) for w in word.split()]
            break
        words.append((word, rest))
        text = rest
        first = False
    real_name = None
    last_real_name = None
    w = 0
    choices = []
    cur_name = ""
    while w < len(words):
        if cur_name:
            cur_name += ' '
        cur_name += words[w][0]
        choices = _name_completion(color_dict, cur_name, all_possible=True)
        if len(choices) == 0:
            break
        multiword_choices = [(c.split()[w], c) for c in choices if ' ' in c]
        if len(multiword_choices) == 0:
            last_real_name = None
            real_name = choices[0]
            break
        choices.sort(key=len)
        last_real_name = choices[0]
        cur_name = cur_name[:-len(words[w][0])] + multiword_choices[0][0]
        w += 1
    if last_real_name:
        w -= 1
        real_name = last_real_name
    if (first or quoted) and w + 1 != len(words):
        return None, None, name
    if real_name:
        if quoted is not None:
            return real_name, color_dict[real_name], quoted
        return real_name, color_dict[real_name], words[w][1]
    return None, None, name


_color_func = re.compile(r"^(rgb|rgba|hsl|hsla|gray)\s*\(([^)]*)\)")
_number = re.compile(r"\s*[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)")
_units = re.compile(r"\s*(%|deg|grad|rad|turn|)\s*")


def _fetch_colormap(session, palette_id):
    '''Fetch color map from colourlovers.com'''
    try:
        palette_id = int(palette_id)
    except Exception:
        pass
    if isinstance(palette_id, int):
        cmap = _colourlovers_fetch_by_id(session, palette_id)
    else:
        cmap = _colourlovers_fetch_by_name(session, palette_id)
    return cmap


def _colourlovers_fetch_by_id(session, palette_id):
    url = 'https://www.colourlovers.com/api/palette/%d?format=json' % palette_id
    from ..fetch import fetch_file
    filename = fetch_file(session, url, 'palette %d' % palette_id, '%d.json' % palette_id, 'COLOURlovers')
    f = open(filename, 'r')
    import json
    j = json.load(f)
    f.close()
    if len(j) == 0:
        from ..errors import UserError
        raise UserError('No palette %d at COLOURlovers.com' % palette_id)
    hex_colors = j[0]['colors']
    rgba = [tuple(int(r, base=16) / 255 for r in (c[0:2], c[2:4], c[4:6])) + (1.0,)
            for c in hex_colors]
    from ..colors import Colormap
    cmap = Colormap(None, rgba)
    return cmap


def _colourlovers_fetch_by_name(session, palette_name):
    bi = palette_name.find(' by ')
    if bi > 0:
        name, author = palette_name[:bi], palette_name[bi + 4:]
    else:
        name, author = palette_name, None
    from urllib.parse import quote
    url = 'https://www.colourlovers.com/api/palettes?keywords=%s&format=json&numResults=100' % quote(name)
    from ..fetch import fetch_file
    try:
        # fetch_file potentially raises OSError
        filename = fetch_file(session, url, 'palette %s' % name, '%s.json' % name, 'COLOURlovers')
        f = open(filename, 'r')
        import json
        j = json.load(f)
        f.close()
        pals = [p for p in j if (p['title'] == name and author is None or p['userName'] == author)]
        if len(pals) == 0:
            raise OSError("no match")
    except OSError:
        from ..errors import UserError
        raise UserError('Could not find palette %s at COLOURlovers.com using keyword search'
                        % (name if author is None else '%s author %s' % (name, author)))
    if len(pals) > 1:
        pals.sort(key=lambda p: p['numViews'], reverse=True)
        session.logger.info('Found %d ColourLover palettes with name "%s", '
                            'using palette id %d by author %s with most views (%d). '
                            'To choose a different one use "name by author" or id number.'
                            % (len(pals), name, pals[0]['id'], pals[0]['userName'], pals[0]['numViews']))
    p = pals[0]
    hex_colors = p['colors']
    rgba = [tuple(int(r, base=16) / 255 for r in (c[0:2], c[2:4], c[4:6])) + (1.0,)
            for c in hex_colors]
    from ..colors import Colormap
    cmap = Colormap(None, rgba)
    return cmap


def _parse_numbers(text):
    # parse comma separated list of number [units]
    result = []
    start = 0
    while True:
        m = _number.match(text, start)
        if not m:
            raise AnnotationError("Expected a number", start)
        n = m.group()
        n_pos = start
        start = m.end()
        m = _units.match(text, start)
        u = m.group(1)
        if not m:
            raise AnnotationError("Unknown units", start)
        u_pos = start
        start = m.end()
        result.append((n, n_pos, u, u_pos))
        if start == len(text):
            return result
        if text[start] != ',':
            raise AnnotationError("Expected a comma", start)
        start += 1


def _convert_number(number, name, *, maximum=255, clamp=True,
                    require_percent=False):
    """Return number scaled to 0 <= n <= 1"""
    n_str, n_pos, u, u_pos = number
    n = float(n_str)
    if require_percent and u != '%':
        raise AnnotationError("%s must be a percentage" % name, u_pos)
    if u == '%':
        n = n / 100
    elif '.' in n_str:
        pass
    elif u == '':
        n = n / maximum
    else:
        raise AnnotationError("Unexpected units for %s" % name, u_pos)
    if clamp:
        if n < 0:
            n = 0
        elif n > 1:
            n = 1
    return n


def _convert_angle(number, name):
    n_str, n_pos, u, u_pos = number
    n = float(n_str)
    if u in ('', 'deg'):
        return n / 360
    if u == 'rad':
        from math import pi
        return n / (2 * pi)
    if u == 'grad':
        return n / 400
    if u == 'turn':
        return n
    raise AnnotationError("'%s' doesn't make sense for %s" % (u, name),
                          offset=u_pos)


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
