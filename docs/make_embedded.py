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

from html import escape
import pkg_resources
import copy

def get_packages_info():
    # based on http://stackoverflow.com/questions/19086030/can-pip-or-setuptools-distribute-etc-list-the-license-used-by-each-install
    # updated for Python3 and modified for use in ChimeraX

    KEY_MAP = {
        "Name": 'name',
        "Version": 'version',
        "License": 'license',
        "Summary": 'summary',
        "Home-page": 'homepage',
    }
    empty_info = {}
    for key, name in KEY_MAP.items():
        empty_info[name] = ""

    packages = pkg_resources.working_set.by_key
    infos = []
    for pkg_name, pkg in packages.items():
        if pkg_name.startswith('chimerax.'):
            continue
        info = copy.deepcopy(empty_info)
        try:
            lines = pkg.get_metadata_lines('METADATA')
        except (KeyError, IOError):
            lines = pkg.get_metadata_lines('PKG-INFO')

        for line in lines:
            try:
                key, value = line.split(': ', 1)
            except ValueError:
                pass
            if key in KEY_MAP:
                info[KEY_MAP[key]] = value
            elif key == 'Classifier':
                what = [x.strip() for x in value.split('::')]
                if len(what) < 2:
                    continue
                if what[0] != 'License':
                    continue
                # TODO: handle multiple licenses, like pyzmq
                # Classifier: License :: OSI Approved :: BSD License
                #if info['license'].startswith('http'):
                #    continue
                if what[1] == 'OSI Approved':
                    if len(what) > 2:
                        info['license'] = what[2]
                    continue
                info['license'] = what[1]

        infos += [info]

    return infos

def html4_id(name):
    # must match [A-Za-z][-A-Za-z0-9_:.]*
    # Python package names must start with letter, so don't check
    def cvt(c):
        if c.isalnum() or c in '-_:.':
            return c
        return '_'
    return ''.join(cvt(c) for c in name)

def print_pkgs(out):
    infos = get_packages_info()
    infos.sort(key=(lambda item: item['name'].casefold()))

    ids = [html4_id(info['name']) for info in infos]
    print('<p>Packages:', file=out)
    sep = ' '
    for info, id in zip(infos, ids):
        print('%s<a href="#%s">%s</a>' % (sep, id, info['name']), end='', file=out)
        sep = ', '
    print('.', file=out)
    print('<dl>', file=out)
    for info, id in zip(infos, ids):
        print('<dt><a href="%s" id="%s">%s</a> version %s' % (
            escape(info['homepage']),
            id,
            escape(info['name']),
            escape(info['version'])), file=out)
        print('<dd>%s<br>' % escape(info['summary']), file=out)
        license = info['license']
        if license:
            print('License type: %s' % escape(license), file=out)
    print('</dl>', file=out)

with open('embedded.html.in') as src:
    with open('embedded.html', 'w') as out:
        for line in src.readlines():
            if line == 'PYTHON_PKGS\n':
                print_pkgs(out)
            else:
                print(line, end='', file=out)

raise SystemExit(0)
