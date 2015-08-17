# vi: set expandtab shiftwidth=4 softtabstop=4:

import re
from . import cli
from ..colors import Color

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
            from ..colors import BuiltinColors
            i = BuiltinColors.bisect_left(token)
            if i >= len(BuiltinColors):
                raise ValueError("Invalid color name")
            name = BuiltinColors.iloc[i]
            if not name.startswith(token):
                raise ValueError("Invalid color name")
            color = BuiltinColors[name]
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
            from .. import colors
            return colors.Colormap(values, colors), text, rest
        else:
            if session is not None:
                i = session.user_colormaps.bisect_left(token)
                if i < len(session.user_colormaps):
                    name = session.user_colormaps.iloc[i]
                    if name.startswith(token):
                        return session.user_colormaps[name], name, rest
            from ..colors import BuiltinColormaps
            i = BuiltinColormaps.bisect_left(token)
            if i >= len(BuiltinColormaps):
                raise ValueError("Invalid colormap name")
            name = BuiltinColormaps.iloc[i]
            if not name.startswith(token):
                raise ValueError("Invalid colormap name")
            return BuiltinColormaps[name], name, rest

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

