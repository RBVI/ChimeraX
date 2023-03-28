# vim: set expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8:

# === UCSF ChimeraX Copyright ===
# Copyright 2016-2023 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

#
# Make X11 desktop menu, icon, and mime types with xdg-utils
#
# Background:
#
#    XDG stands for X Desktop Group, the name formerly used by
#    freedesktop.org.  freedesktop.org is the organization which publishes
#    the Linux Desktop Entry specification,
#    https://www.freedesktop.org/wiki/Specifications/desktop-entry-spec/,
#    and a set of shell script that implement it, xdg-utils, at
#    https://www.freedesktop.org/wiki/Software/xdg-utils/.
#
#   A .desktop file for the desktop icon and a .mime.types file for the
#   MIME types are needed.  The .mime.types file can be distributed with
#   ChimeraX, but the .desktop has to be generated during/after
#   installation because it refers to actual location ChimeraX is
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

CATEGORIES = [
    "Education", "Science",
    "Biology", "Chemistry", "DataVisualization",
    "Graphics", "3DGraphics"
]

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


def make_desktop(session, info, localized_app_name={}, verbose=False):
    if verbose:
        print("generating", info.desktop)
    mime_types = get_mime_types(session)
    with open(info.desktop, mode='wt', encoding='utf-8') as f:
        desktop_group(f, "Desktop Entry")
        desktop_string(f, "Type", "Application")
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
                       "An extensible molecular modeling system, "
                       "https://www.rbvi.ucsf.edu/chimerax/")
        desktop_string(f, "Icon", info.name)
        desktop_stringlist(f, "Categories", CATEGORIES)
        desktop_stringlist(f, "MimeType", mime_types)
        # Don't set StartupWMClass because is shared with all releases
        # and Gnome picks the last .desktop for showing the version of
        # the running program, even if a different release.
        # desktop_string(f, "StartupWMClass", info.app_name)
        if '=' in sys.executable:
            raise RuntimeError("warning: '=' found in path to ChimeraX")
        else:
            desktop_string(f, "Exec",
                           "%s -- %%F" % arg_quote(sys.executable))
        desktop_boolean(f, "PrefersNonDefaultGPU", True)
    s = os.stat(info.desktop)
    os.chmod(info.desktop, s.st_mode | 0o555)  # make executable


def make_mime_file(session, name, verbose=False):
    if verbose:
        print("generating", name)
    mi = MimeInfo(open(name, mode='wt', encoding='utf-8'))
    with mi:
        fmts = session.data_formats.formats
        fmts.sort(key=lambda f: f.name)
        for f in fmts:
            extensions = f.suffixes
            mime_types = f.mime_types
            if not extensions or not mime_types:
                continue
            for m in mime_types:
                with mi.type(m):
                    mi.comment(f.category)
                    for e in extensions:
                        mi.glob(e)


def install_icons(session, info, verbose=False):
    if verbose:
        print("installing icons")

    # install application icon
    # image_dir = "%s/images" % info.icon_dir
    image_dir = info.icon_dir
    sizes = (16, 32, 64, 128)
    for size in sizes:
        path = '%s/%s-icon%d.png' % (image_dir, info.app_name, size)
        if not os.path.exists(path):
            continue
        cmd = [
            'xdg-icon-resource', 'install',
            '--context', 'apps',
            '--size', str(size),
            '--mode', 'system' if info.system else 'user',
            path, info.name
        ]
        if size != sizes[-1]:
            cmd[2:2] = ['--noupdate']
        try:
            subprocess.call(cmd)
        except OSError as e:
            print("Unable to install %sx%s icon: %s" % (size, size, e), file=sys.stderr)
    # scalable application icon
    is_root = os.getuid() == 0
    if info.system == is_root and os.path.exists('/usr/share/icons/hicolor/scalable'):
        path = '%s/%s-icon.svg' % (image_dir, info.app_name)
        if info.system:
            p2 = '/usr/share/icons/hicolor/scalable'
        else:
            p2 = os.path.expanduser("~/.local/share/icons/hicolor/scalable/apps")
            os.makedirs(p2, exist_ok=True)
        import shutil
        shutil.copyfile(path, os.path.join(p2, '%s.svg' % info.name))
        cmd = [
            'xdg-icon-resource', 'forceupdate',
            '--mode', 'system' if info.system else 'user'
        ]
        try:
            subprocess.call(cmd)
        except OSError as e:
            print("Unable to install SVG icon: %s" % e, file=sys.stderr)

    # No format actually provides an icon, and therefore session.data_formats
    # doesn't currently support it; the below code could be revived if that
    # situation changes
    """
    # install icons for file formats
    from PIL import Image
    for f in session.data_formats.formats:
        icon = f.icon
        if icon is None:
            continue
        if not os.path.exists(icon):
            continue
        try:
            im = Image.open(icon)
        except IOError:
            # logger.warning('unable to load icon: %s' % icon)
            continue
        if im.width != im.height:
            # logger.warning('skipping non-square icon: %s' % icon)
            continue
        mime_types = f.mime_types
        if not mime_types:
            continue
        for mt in mime_types:
            cmd = [
                'xdg-icon-resource', 'install',
                '--context', 'mimetypes',
                '--size', '%d' % im.width,
                '--mode', 'system' if info.system else 'user',
                icon, mt
            ]
            try:
                subprocess.call(cmd)
            except OSError as e:
                print("Unable to install %s icon: %s" % (f.name, e), file=sys.stderr)
    """


def install_desktop_menu(desktop, system, verbose=False):
    if verbose:
        print("installing desktop menu")
    cmd = [
        'xdg-desktop-menu',
        'install',
        '--mode', 'system' if system else 'user',
        desktop
    ]
    try:
        subprocess.call(cmd)
    except OSError as e:
        print("Unable to install desktop menu: %s" % e, file=sys.stderr)


def uninstall_desktop_menu(desktop, system, verbose=False):
    if verbose:
        print("uninstalling desktop menu")
    cmd = [
        'xdg-desktop-menu',
        'uninstall',
        '--mode', 'system' if system else 'user',
        desktop
    ]
    try:
        subprocess.call(cmd)
    except OSError as e:
        print("Unable to uninstall desktop menu: %s" % e, file=sys.stderr)


def install_desktop_icon(desktop, verbose=False):
    # only works for current user
    if verbose:
        print("installing desktop icon")
    cmd = ['xdg-desktop-icon', 'install', desktop]
    try:
        subprocess.call(cmd)
    except OSError as e:
        print("Unable to install desktop icon: %s" % e, file=sys.stderr)


def uninstall_desktop_icon(desktop, verbose=False):
    # only works for current user
    if verbose:
        print("uninstalling desktop icon")
    cmd = ['xdg-desktop-icon', 'uninstall', desktop]
    try:
        subprocess.call(cmd)
    except OSError as e:
        print("Unable to uninstall desktop icon: %s" % e, file=sys.stderr)


def install_mime_file(mimetypes, system, verbose=False):
    if verbose:
        print("installing MIME info")
    cmd = [
        'xdg-mime',
        'install',
        '--mode', 'system' if system else 'user',
        mimetypes
    ]
    try:
        subprocess.call(cmd)
    except OSError as e:
        print("Unable to install mime types: %s" % e, file=sys.stderr)


def uninstall_mime_file(mimetypes, system, verbose=False):
    if verbose:
        print("uninstalling MIME info")
    cmd = [
        'xdg-mime',
        'uninstall',
        '--mode', 'system' if system else 'user',
        mimetypes
    ]
    try:
        subprocess.call(cmd)
    except OSError as e:
        print("Unable to uninstall mime types: %s" % e, file=sys.stderr)


def generate(session, info=None, system=False, verbose=False):
    if not info:
        info = get_info(session, system, verbose=verbose)
    from chimerax.core.__main__ import localized_app_name
    make_desktop(session, info, localized_app_name, verbose=verbose)
    make_mime_file(session, info.mime_file, verbose)


def install(session, system=False, verbose=False):
    info = get_info(session, system, verbose=verbose)
    generate(session, info, system, verbose)
    install_mime_file(info.mime_file, system, verbose)
    install_icons(session, info, verbose)
    install_desktop_menu(info.desktop, info.system, verbose)
    if not system:
        install_desktop_icon(info.desktop, verbose=verbose)


def uninstall(session, system=False, verbose=False):
    info = get_info(session, system, verbose=verbose)
    if os.path.exists(info.desktop):
        if not system:
            uninstall_desktop_icon(info.desktop, verbose=verbose)
        uninstall_desktop_menu(info.desktop, info.system, verbose)
        try:
            os.remove(info.desktop)
        except FileNotFoundError:
            pass
    # Don't uninstall icons because they might be
    # shared with other packages
    if os.path.exists(info.mime_file):
        uninstall_mime_file(info.mime_file, system, verbose)
        try:
            os.remove(info.mime_file)
        except FileNotFoundError:
            pass


def get_mime_types(session):
    mime_types = []
    for f in session.data_formats.formats:
        mt = f.mime_types
        if isinstance(mt, (list, tuple)):
            mime_types.extend(mt)
        elif mt:
            mime_types.append(mt)
    mime_types.sort()
    return mime_types


def get_info(session, system, create=False, verbose=False):
    class Info:
        pass
    info = Info()
    info.system = system
    from chimerax import app_dirs, app_data_dir
    if not system:
        info.save_dir = app_dirs.user_config_dir
    else:
        info.save_dir = app_data_dir
    info.icon_dir = app_data_dir
    info.app_name = app_dirs.appname
    info.app_author = app_dirs.appauthor
    info.name = '%s-%s' % (info.app_author, info.app_name)
    version = None
    from chimerax.core import BUNDLE_NAME as CORE_BUNDLE_NAME
    import pkg_resources
    for d in pkg_resources.working_set:
        if d.project_name == CORE_BUNDLE_NAME:
            version = d.version
            break
    if version is None:
        version = 'unknown'
    info.version = version
    info.desktop = '%s/%s-%s.desktop' % (
        info.save_dir, info.name, info.version)
    info.mime_file = '%s/%s-%s.mime.types' % (
        info.save_dir, info.name, info.version)
    return info
