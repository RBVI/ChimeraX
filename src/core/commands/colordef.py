import re
ColorNames = re.compile(r'[a-z][-_a-z0-9 ]*')


def _find_named_color(color_dict, name):
    # handle color names with spaces
    # returns key, value, part of name that was unused
    num_colors = len(color_dict)
    words = name.split(maxsplit=10)
    real_name = None
    last_real_name = None
    w = 0
    choices = []
    cur_name = ""
    while w < len(words):
        if cur_name:
            cur_name += ' '
        cur_name += words[w]
        i = color_dict.bisect_left(cur_name)
        if i >= num_colors:
            break
        choices = []
        for i in range(i, num_colors):
            color_name = color_dict.iloc[i]
            if not color_name.startswith(cur_name):
                break
            choices.append(color_name)
        if len(choices) == 0:
            break
        multiword_choices = [(c.split()[w], c) for c in choices if ' ' in c]
        if len(multiword_choices) == 0:
            last_real_name = None
            real_name = choices[0]
            break
        choices.sort(key=len)
        last_real_name = choices[0]
        cur_name = cur_name[:-len(words[w])] + multiword_choices[0][0]
        w += 1
    if last_real_name:
        w -= 1
        real_name = last_real_name
    if real_name:
        start = 0
        for i in range(w + 1):
            start = name.find(words[i], start)
            start += len(words[i])
        unused = name[start:]
        return real_name, color_dict[real_name], unused
    return None, None, name


def colordef(session, name, color=None):
    """Create a user defined color."""
    if ColorNames.match(name) is None:
        from ..errors import UserError
        raise UserError('Illegal color name: "%s"' % name)
    if color is None:
        # TODO: need to merge the two color dictionaries to properly
        # resolve abbreviations
        if session is not None:
            real_name, color, rest = _find_named_color(session.user_colors, name)
            if rest:
                color = None
        else:
            from ..colors import _BuiltinColors
            real_name, color, rest = _find_named_color(_BuiltinColors, name)
            if rest:
                color = None
        if color is None:
            from ..errors import UserError
            raise UserError('Unknown color %r' % name)

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
        if not session.ui.is_gui:
            session.logger.info(msg)
        else:
            session.logger.status(msg)
            session.logger.info(
                msg +
                '<div style="width:1em; height:.5em;'
                ' display:inline-block;'
                ' border:1px solid #000; background-color:%s"></div>'
                % color.hex(), is_html=True)
        return
    name = ' '.join(name.split())   # canonicalize
    session.user_colors[name] = color


def uncolordef(session, name):
    """Remove a user defined color."""
    del session.user_colors[name]


# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, StringArg, ColorArg
    register(
        'colordef',
        CmdDesc(required=[('name', StringArg)],
                optional=[('color', ColorArg)],
                synopsis="define a custom color"),
        colordef
    )
    register(
        '~colordef',
        CmdDesc(required=[('name', StringArg)],
                synopsis="remove color definition"),
        uncolordef
    )
