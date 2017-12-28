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

import re
ColorNames = re.compile(r'[a-z][-_a-z0-9 ]*')


def _find_named_color(color_dict, name):
    # handle color names with spaces
    # returns key, value, part of name that was unused
    num_colors = len(color_dict)
    # extract up to 10 words from name
    from . import cli
    first = True
    text = name
    words = []
    while len(words) < 10:
        m = cli._whitespace.match(text)
        text = text[m.end():]
        if not text:
            break
        word, _, rest = cli.next_token(text, no_raise=True)
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
        cur_name = cur_name[:-len(words[w][0])] + multiword_choices[0][0]
        w += 1
    if last_real_name:
        w -= 1
        real_name = last_real_name
    if first and w + 1 != len(words):
        return None, None, name
    if real_name:
        return real_name, color_dict[real_name], words[w][1]
    return None, None, name


def html_color_swatch(color):
    return (
        '&nbsp;<div style="width:1em; height:.6em; display:inline-block;'
        ' border:1px solid #000; background-color:%s"></div>'
        % color.hex())


def name_color(session, name, color):
    """Create a custom color."""
    if ColorNames.match(name) is None:
        from ..errors import UserError
        raise UserError('Illegal color name: "%s"' % name)

    name = ' '.join(name.split())   # canonicalize
    session.user_colors.add(name, color)
    show_color(session, name)


def show_color(session, name):
    """Show color in log."""
    if ColorNames.match(name) is None:
        from ..errors import UserError
        raise UserError('Illegal color name: "%s"' % name)

    name = ' '.join(name.split())   # canonicalize

    if session is not None:
        real_name, color, rest = _find_named_color(session.user_colors, name)
        if rest:
            color = None
    else:
        from ..colors import BuiltinColors
        real_name, color, rest = _find_named_color(BuiltinColors, name)
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
        transmit = '100% transparent'
    else:
        transmit = '%.4g%% transparent' % percent(1 - alpha)

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
            msg + html_color_swatch(color), is_html=True)


def delete_color(session, name):
    """Remove a custom color."""
    if name == 'custom':
        color_names = session.user_colors.list()
        for name in color_names:
            session.user_colors.remove(name)
        return
    if name not in session.user_colors:
        from ..errors import UserError
        raise UserError('Unknown color %r' % name)
    try:
        session.user_colors.remove(name)
    except ValueError as v:
        from ..errors import UserError
        raise UserError(v)


def list_colors(session, which='all'):
    from . import cli
    if which == 'all' or which == 'custom':
        from sortedcontainers import SortedDict
        d = SortedDict([(name, session.user_colors[name])
                        for name in session.user_colors.list()])
        _list_colors(session, d, 'custom')
    if which == 'all' or which == 'builtin':
        from .. import colors
        _list_colors(session, colors.BuiltinColors, 'builtin',
                     "http://rbvi.ucsf.edu/chimerax/docs/user/commands/"
                     "colornames.html#builtin")


def _list_colors(session, colors_dict, kind, url=None):
    from . import cli
    import html
    logger = session.logger
    if not colors_dict:
        logger.info("No %s colors." % kind)
        return
    is_html = session.ui.is_gui
    colors = []
    for name, c in colors_dict.items():
        if is_html:
            colors.append(html.escape(name) + html_color_swatch(c))
        else:
            colors.append(name)
    noun = cli.plural_form(colors, 'color')
    if url is None or not is_html:
        label = "%d %s %s: " % (len(colors), kind, noun)
    else:
        label = "%d <a href=\"%s\">%s %s</a>: " % (len(colors), url, kind, noun)
    logger.info(label + cli.commas(colors, ' and') + '.', is_html=is_html)


# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, StringArg, ColorArg, NoArg, EnumOf, Or, RestOfLine, create_alias
    register(
        'color list',
        CmdDesc(
            optional=[('which', EnumOf(["all", "custom", "builtin"]))],
            synopsis='list colors'),
        list_colors, logger=session.logger
    )

    register(
        'color show',
        CmdDesc(required=[('name', RestOfLine)],
                synopsis="show color"),
        show_color, logger=session.logger
    )
    register(
        'color name',
        CmdDesc(required=[('name', StringArg), ('color', ColorArg)],
                synopsis="name a custom color"),
        name_color, logger=session.logger
    )
    register(
        'color delete',
        CmdDesc(required=[('name', Or(EnumOf(['custom']), StringArg))],
                synopsis="remove color definition"),
        delete_color, logger=session.logger
    )
    create_alias('colordef', 'color name $*', logger=session.logger)
    create_alias('~colordef', 'color delete $*', logger=session.logger)
