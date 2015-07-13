#!/bin/env python
# vi: set expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8:
# Copyright © 2010-2015 Regents of the University of California.
# All Rights Reserved.
#
# Make X11 desktop menu, icon, and mime types with xdg-utils
#
# Background:
#
#    XDG stands for X Desktop Group, the name formerly used by
#    freedesktop.org.  freedesktop.org is the organization which publishes
#    the Linux Desktop Entry specification,
#    http://www.freedesktop.org/wiki/Specifications/desktop-entry-spec/,
#    and a set of shell script that implement it, xdg-utils, at
#    http://portland.freedesktop.org/.
#
#   A .desktop file for the desktop icon and a .mime.types file for the
#   MIME types are needed.  The .mime.types file can be distributed with
#   Chimera2, but the .desktop has to be generated during/after
#   installation because it refers to acutal location Chimera2 is
#   installed.
#
# Protocol:
#
#   The main entry points are uninstall and install_if_needed.  The
#   latter looks to see if the .desktop and .mime.types files exists,
#   and if not, installs them.  The .desktop file on the Desktop can
#   be safely removed and will not reappear unless explicitly asked
#   to reinstall.

import os
import subprocess
import sys

verbose = False

# From Desktop Entry Specification 1.0:
#
# The escape sequences \s, \n, \t, \r, and \\ are supported for values
# of type string and localestring, meaning ASCII space, newline, tab,
# carriage return, and backslash, respectively.
#
# Some keys can have multiple values. In such a case, the value of the key
# is specified as a plural: for example, string(s). The multiple values
# should be separated by a semicolon. Those keys which have several values
# should have a semicolon as the trailing character. Semicolons in these
# values need to be escaped using \;.


def str_quote(text):
    result = ""
    for ch in text:
        if ch == '\n':
            result += '\\n'
        elif ch == '\t':
            result += '\\t'
        elif ch == '\r':
            result += '\\r'
        elif ch == '\\':
            result += '\\\\'
        elif ch == ';':
            result += '\\;'
        elif ord(ch) < 32:
            continue
        else:
            result += ch
    return result

# From Desktop Entry Specification 1.0:
#
# Arguments may be quoted in whole.  If an argument contains a reserved
# character the argument must be quoted.  The rules for quoting of arguments
# is also applicable to the executable name or path of the executable
# program as provided.
#
# Quoting must be done by enclosing the argument between double quotes and
# escaping the double quote character, backtick character ("`"), dollar
# sign ("$") and backslash character ("\") by preceding it with an
# additional backslash character.  Implementations must undo quoting before
# expanding field codes and before passing the argument to the executable
# program.  Reserved characters are space (" "), tab, newline, double quote,
# single quote ("'"), backslash character ("\"), greater-than sign (">"),
# less-than sign ("<"), tilde ("~"), vertical bar ("|"), ampersand ("&"),
# semicolon (";"), dollar sign ("$"), asterisk ("*"), question mark ("?"),
# hash mark ("#"), parenthesis ("(") and (")") and backtick character ("`").
reserved_char = """ \t\n"'\\><~|&;$*?#()`"""


def arg_quote(arg):
    has_reserved = any(True for ch in arg if ch in reserved_char)
    if not has_reserved:
        return arg
    result = '"'
    for ch in arg:
        if ch in '"`$\\':
            result += '\\'
        result += ch
    result += '"'
    return result

# <?xml version="1.0"?>
# <mime-info xmlns='http://www.freedesktop.org/standards/shared-mime-info'>
#   <mime-type type="text/x-shiny">
#     <comment>Shiny new file type</comment>
#     <glob pattern="*.shiny"/>
#     <glob pattern="*.shi"/>
#   </mime-type>
# </mime-info>


class MimeInfo:
    IDENT = '    '

    class Nested:

        def __init__(self, plist, tag, args=''):
            self.plist = plist
            self.tag = tag
            self.args = args

        def __enter__(self):
            p = self.plist
            p.output.write("%s<%s%s>\n" % (p.IDENT * p.level, self.tag, self.args))
            p.level += 1

        def __exit__(self, exc_type, exc_value, traceback):
            p = self.plist
            p.level -= 1
            p.output.write("%s</%s>\n" % (p.IDENT * p.level, self.tag))

    def __init__(self, output=sys.stdout):
        self.level = 0
        self.output = output

    def __enter__(self):
        self.output.write(
            """<?xml version="1.0" encoding="UTF-8"?>
            <mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
            """)
        self.level = 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.output.write("</mime-info>\n")
        self.output.close()

    def xml_comment(self, text):
        self.output.write("%s<!-- %s -->\n" % (self.IDENT * self.level, text))

    def comment(self, text):
        self.output.write("%s<comment>%s</comment>\n"
                          % (self.IDENT * self.level, text))

    def glob(self, pattern):
        self.output.write('%s<glob pattern="*%s"/>\n'
                          % (self.IDENT * self.level, pattern))

    def type(self, mimetype):
        # <mime-type type="text/x-shiny">
        return self.Nested(self, "mime-type", args=' type="%s"' % mimetype)


def desktop_comment(f, text):
    f.write("# %s\n" % text)


def desktop_group(f, name):
    # assert '[' not in name and ']' not in name
    f.write("[%s]\n" % name)


def desktop_boolean(f, tag, value):
    f.write("%s=%s\n" % (tag, "true" if value else "false"))


def desktop_numeric(f, tag, value, format="%f"):
    f.write(("%s=" + format + "\n") % (tag, value))


def desktop_string(f, tag, value):
    f.write("%s=%s\n" % (tag, str_quote(value)))


def desktop_stringlist(f, tag, values):
    f.write("%s=%s;\n" % (tag, ';'.join(str_quote(v) for v in values)))


def make_desktop(info, localized_app_name={}):
    if verbose:
        print("generating", info.desktop)
    mime_types = get_mime_types()
    with open(info.desktop, mode='wt', encoding='utf-8') as f:
        desktop_group(f, "Desktop Entry")
        from . import __copyright__ as copyright
        desktop_comment(f, copyright)
        desktop_string(f, "Type", "Application")
        desktop_string(f, "Version", info.version)
        desktop_string(f, "Encoding", "UTF-8")
        desktop_string(f, "Name", "%s %s %s" % (
                       info.app_author, info.app_name, info.version))
        locales = list(localized_app_name.keys())
        locales.sort()
        for lo in locales:
            desktop_string(f, "Name[%s]" % lo, "%s %s %s" % (
                           info.app_author, localized_app_name[lo], info.version))
        desktop_string(f, "GenericName", "Molecular Visualization")
        desktop_string(f, "Comment",
                       "A extensible molecular modeling system, "
                       "http://www.cgl.ucsf.edu/chimera/")
        desktop_string(f, "Icon", info.name)
        desktop_stringlist(f, "Categories", [
                           "Education", "Science", "Biology", "Chemistry",
                           "Graphics", "2DGraphics", "DataVisualization"])
        desktop_stringlist(f, "MimeType", mime_types)
        if '=' in sys.executable:
            print("warning: '=' found in path to chimera", file=sys.stderr)
        else:
            desktop_string(f, "Exec",
                           "%s -- %%F" % arg_quote(sys.executable))
    s = os.stat(info.desktop)
    os.chmod(info.desktop, s.st_mode | 0o555)  # make executable


def make_mime_file(name):
    if verbose:
        print("generating", name)
    from . import io
    mi = MimeInfo(open(name, mode='wt', encoding='utf-8'))
    with mi:
        from . import __copyright__ as copyright
        mi.xml_comment(copyright)

        names = io.format_names()
        names.sort()
        for fn in names:
            extensions = io.extensions(fn)
            mime_types = io.mime_types(fn)
            if not extensions or not mime_types:
                continue
            for m in mime_types:
                with mi.type(m):
                    mi.comment(io.category(fn))
                    for e in extensions:
                        mi.glob(e)


def add_xdg_utils_to_path(data_dir):
    if verbose:
        print("adding xdg scripts to end of path")
    old_path = os.getenv("PATH")
    path = old_path + ":%s/xdg-utils" % data_dir
    os.environ["PATH"] = path
    return old_path


def install_icons(info, data_dir):
    if verbose:
        print("installing icons")

    # install application icon
    # image_dir = "%s/images" % data_dir
    image_dir = data_dir
    sizes = (16, 32, 64, 128)
    for size in sizes:
        path = '%s/%s-icon%d.png' % (image_dir, info.app_name, size)
        if not os.path.exists(path):
            continue
        cmd = [
            'xdg-icon-resource', 'install',
            '--context', 'apps',
            '--size', str(size),
            '--mode', 'user',
            path, info.name
        ]
        if size != sizes[-1]:
            cmd[2:2] = ['--noupdate']
        subprocess.call(cmd)
    # scalable application icon
    if os.path.exists('/usr/share/icons/hicolor/scalable'):
        path = '%s/%s-icon.svg' % (image_dir, info.app_name)
        p2 = os.path.expanduser("~/.local/share/icons/hicolor/scalable/apps")
        os.mkdirs(p2, exist_ok=True)

    # install icons for file formats
    from . import io
    from PIL import Image
    for fn in io.format_names(open=True, export=True):
        icon = io.icon(fn)
        if icon is None:
            continue
        try:
            im = Image.open(icon)
        except IOError as e:
            # logger.warning('unable to load icon: %s' % icon)
            continue
        if im.width != im.height:
            # logger.warning('skipping non-square icon: %s' % icon)
            continue
        mime_types = io.mime_types(fn)
        if not mime_types:
            continue
        for mt in mime_types:
            cmd = [
                'xdg-icon-resource', 'install',
                '--context', 'mimetypes',
                '--size', '%d' % im.width,
                '--mode', 'user',
                icon, mt
            ]
            subprocess.call(cmd)


def install_desktop_menu(desktop):
    if verbose:
        print("installing desktop menu")
    cmd = ['xdg-desktop-menu', 'install', desktop]
    subprocess.call(cmd)


def install_desktop_icon(desktop):
    if verbose:
        print("installing desktop icon")
    cmd = ['xdg-desktop-icon', 'install', desktop]
    subprocess.call(cmd)


def uninstall_desktop_menu(desktop):
    if verbose:
        print("uninstalling desktop menu")
    cmd = ['xdg-desktop-menu', 'uninstall', desktop]
    subprocess.call(cmd)


def uninstall_desktop_icon(desktop):
    if verbose:
        print("uninstalling desktop icon")
    cmd = ['xdg-desktop-icon', 'uninstall', desktop]
    subprocess.call(cmd)


def install_mime_file(mimetypes):
    if verbose:
        print("installing MIME info")
    cmd = ['xdg-mime', 'install', mimetypes]
    subprocess.call(cmd)


def uninstall_mime_file(mimetypes):
    if verbose:
        print("uninstalling MIME info")
    cmd = ['xdg-mime', 'uninstall', mimetypes]
    subprocess.call(cmd)


def generate(session, localized_app_name):
    info = get_info(session)
    if info.already_generated:
        if verbose:
            print("already generated")
    else:
        make_desktop(info, localized_app_name)
        make_mime_file(info.mime_file)


def install(session, localized_app_name, reinstall=False, info=None):
    if info is None:
        info = get_info(session)
    if not info.already_generated or reinstall:
        make_desktop(info, localized_app_name)
        make_mime_file(info.mime_file)
    old_path = add_xdg_utils_to_path(session.app_data_dir)
    if not info.already_generated or reinstall:
        install_mime_file(info.mime_file)
        install_icons(info, session.app_data_dir)
        install_desktop_menu(info.desktop)
        install_desktop_icon(info.desktop)
    os.environ["PATH"] = old_path


def uninstall(session):
    info = get_info(session)
    old_path = add_xdg_utils_to_path(session.app_data_dir)
    if info.already_generated:
        uninstall_desktop_icon(info.desktop)
        uninstall_desktop_menu(info.desktop)
        uninstall_mime_file(info.mime_file)
        os.remove(info.desktop)
        os.remove(info.mime_file)
    os.environ["PATH"] = old_path


def install_if_needed(session, localized_app_name={}, reinstall=False):
    info = get_info(session)
    reinstall = False
    if info.already_generated and not reinstall:
        # TODO: check if we should reinstall
        return
    install(session, localized_app_name, reinstall=reinstall, info=info)


def get_mime_types():
    from . import io
    mime_types = []
    for fn in io.format_names():
        mt = io.mime_types(fn)
        if isinstance(mt, (list, tuple)):
            mime_types.extend(mt)
        elif mt:
            mime_types.append(mt)
    mime_types.sort()
    return mime_types


def get_info(session, command=None):
    class Info:
        pass
    info = Info()
    info.app_name = session.app_dirs.appname
    info.app_author = session.app_dirs.appauthor
    info.name = '%s-%s' % (info.app_author, info.app_name)
    version = None
    import pip
    dists = pip.get_installed_distributions(local_only=True)
    for d in dists:
        if d.key == 'chimera.core':
            version = d.version
            break
    if version is None:
        version = 'unknown'
    info.version = version
    info.desktop = '%s/%s-%s.desktop' % (
        session.app_dirs.user_config_dir, info.name, info.version)
    info.mime_file = '%s/%s-%s.mime.types' % (
        session.app_dirs.user_config_dir, info.name, info.version)
    info.already_generated = (os.path.exists(info.desktop) and
                              os.path.exists(info.mime_file))
    return info
