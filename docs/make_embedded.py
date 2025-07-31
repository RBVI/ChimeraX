# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

import sys
import copy
from html import escape
import os
import pathlib
import platform
import importlib.metadata
import shutil

ffmpeg_doc = """
<p>
<dt><a href="https://ffmpeg.org/" target="_blank"> FFmpeg </a> version VERSION
<dd>&ldquo;A complete, cross-platform solution to record, convert and stream audio and video.&rdquo;
<br>
License: <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank">GNU General Public License v3</a><br>
Embedded licenses: <a href="licenses/ffmpeg-LICENSE.html" target="_blank">FFmpeg embedded licences</a>
<p>
FFmpeg is bundled as a convenience for users of ChimeraX.
It is a separate product,
and thus, its license does not affect ChimeraX's license.
FFmpeg can be freely redistributed separately from ChimeraX.
See the <a href="https://www.gnu.org/licenses/gpl-faq.html" target="_blank">GPL FAQ</a> for more details.
It can be found in the <code>bin</code> directory.
"""

openmm_doc = """
<p>
<dt><a href="https://openmm.org/" target="_blank">OpenMM</a> version 7.3.0
<dd>
A high performance toolkit for molecular simulation.
<br>
License type: MIT and LGPL.
<br>
All licenses: <a href="licenses/openmm.txt" target="_blank">OpenMM licences</a>
<p>
Any work that uses OpenMM should cite the following publication:
<blockquote>
P. Eastman, J. Swails, J. D. Chodera, R. T. McGibbon, Y. Zhao, K. A. Beauchamp, L.-P. Wang, A. C. Simmonett, M. P. Harrigan, C. D. Stern, R. P. Wiewiora, B. R. Brooks, and V. S. Pande.
<i>OpenMM 7: Rapid development of high performance algorithms for molecular dynamics.</i> PLOS Comp. Biol. 13(7): e1005659. (2017)
</blockquote>
"""


def get_packages_info():
    """Get information about installed python packages"""
    # based on https://stackoverflow.com/questions/19086030/can-pip-or-setuptools-distribute-etc-list-the-license-used-by-each-install
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

    infos = []
    for pkg in importlib.metadata.distributions():
        if pkg.name.startswith('ChimeraX-'):
            continue
        info = copy.deepcopy(empty_info)
        classifier_licenses = []
        for key, value in pkg.metadata.items():
            value = value.strip()
            if key in KEY_MAP:
                info[KEY_MAP[key]] = value
            elif key == 'Classifier':
                category, text = value.split('::', 1)
                category = category.strip()
                if category != 'License':
                    continue
                text = text.strip()
                # Classifier: License :: OSI Approved :: BSD License
                if text.startswith('OSI Approved'):
                    extract = text.split('::', 1)
                    if len(extract) > 1:
                        text = extract[1].strip()
                classifier_licenses.append(text)
        if classifier_licenses:
            # TODO? if not info['license'].startswith('http'):
            info['license'] = ', '.join(classifier_licenses)

        license_file = find_license_file(pkg)
        info['license-file'] = license_file

        infos += [info]

    return infos


def find_license_file(pkg):
    # return absolute path to a file with the license for a package if found
    import os
    import re

    license_re = re.compile('^(LICENSE|COPYING)')
    license_files = []
    top_levels = []

    for path in pkg.files:
        if not path.parts[0].endswith("-info"):
            continue
        if path.name == "top_level.txt":
            with open(pkg.locate_file(path), 'rt') as f:
                top_levels = [x.strip() for x in f.readlines()]
        elif path.name == 'licenses':
            entry = pkg.locate_file(path)
            if entry.is_dir():
                files = list(entry.iterdir())
                if len(files) == 1:
                    license_files.append(str(files[0]))
                else:
                    for filename in files:
                        if license_re.match(filename.name):
                            license_files.append(str(filename))
        elif license_re.match(path.name):
            entry = pkg.locate_file(path)
            if entry.is_file():
                license_files.append(str(entry))

    # scan modules provided by egg/wheel for a license
    if not license_files and top_levels:
        import importlib
        for t in top_levels:
            try:
                m = importlib.import_module(t)
            except Exception:
                continue
            if not hasattr(m, '__path__'):
                continue
            for dirname in m.__path__:
                for filename in os.listdir(dirname):
                    if license_re.match(filename):
                        license_files.append(os.path.join(dirname, filename))
    if not license_files:
        return None
    return sorted(license_files, key=len)[0]


def html4_id(name):
    # must match [A-Za-z][-A-Za-z0-9_:.]*
    # Python package names must start with letter, so don't check
    def cvt(c):
        if c.isalnum() or c in '-_:.':
            return c
        return '_'
    return ''.join(cvt(c) for c in name)


def print_pkgs(infos, out):
    ids = [html4_id(info['name']) for info in infos]
    print('<p>Packages:', file=out)
    sep = ' '
    for info, id in zip(infos, ids):
        print('%s<a href="#%s">%s</a>' % (sep, id, info['name']), end='', file=out)
        sep = ', '
    print('.', file=out)
    print('<dl>', file=out)
    for info, id in zip(infos, ids):
        print('<dt><a href="%s" id="%s" target="_blank">%s</a> version %s' % (
            escape(info['homepage']),
            id,
            escape(info['name']),
            escape(info['version'])), file=out)
        print('<dd>%s<br>' % escape(info['summary']), file=out)
        license = info['license']
        if license:
            lf = info['license-file']
            if not lf:
                print('License type: %s' % escape(license), file=out)
            else:
                fn = 'licenses/%s-%s' % (id, os.path.basename(lf))
                if not os.path.splitext(fn)[1]:
                    fn += '.txt'
                shutil.copyfile(lf, fn)
                print('License type: <a href="%s">%s</a>' % (fn, escape(license)), file=out)
    print('</dl>', file=out)


def extract_version(srcdir, var_name):
    makefile = os.path.join(srcdir, 'Makefile')
    version_str = f"{var_name} ="
    with open(makefile) as f:
        # var_name = 7.3.0
        for line in f.readlines():
            if line.startswith(version_str):
                version = line.split()[2]
                break
        else:
            return None
        return version


def ffmpeg_version():
    ffmpeg_srcdir = pathlib.Path('..', 'prereqs', 'ffmpeg')
    version = extract_version(ffmpeg_srcdir, "FFMPEG_VERSION")
    if version is None:
        print('Unable to find ffmpeg version')
        return ''
    return version


def ffmpeg_licenses():
    print("GOT HERE!!!")
    import markdown
    license_file = pathlib.Path('..', 'prereqs', 'ffmpeg', 'LICENSE.md')
    with open(license_file, 'r', encoding='utf-8') as f:
        license = f.read()
    html = markdown.markdown(license)
    with open('licenses/ffmpeg-LICENSE.html', 'w') as f:
        f.write(html)
    return
    ffmpeg_srcdir = pathlib.Path('..', 'prereqs', 'ffmpeg')
    import zipfile
    windist_dir = 'ffmpeg-%s-win64-static' % version
    zip_filename = os.path.join(ffmpeg_srcdir, '%s.zip' % windist_dir)
    with zipfile.ZipFile(zip_filename) as zf:
        out_dir = os.path.join('licenses', 'ffmpeg')
        out_filename = os.path.join(out_dir, 'index.html')
        license_prefix = '%s/licenses/' % windist_dir
        licenses = [n for n in zf.namelist() if n.startswith(license_prefix)]
        licenses.sort()
        zf.extractall(path=out_dir, members=licenses)
        with open(out_filename, 'w') as f:
            print("""<html>
 <head>
  <title> FFmpeg Embedded Licenses </title>
 </head
 <body>
 The FFmpeg binary incorporates some or all of the following libraries and their licenses:
  <ul>
""", file=f)
            for n in licenses:
                bn = os.path.basename(n)
                if not bn:
                    continue
                library = os.path.splitext(bn)[0]
                print('<li> <a href="%s">%s</a>' % (n, library), file=f)
            print("""
  </ul>
 </body>
</html>
""", file=f)


def openmm_licenses():
    openmm_srcdir = pathlib.Path('..', 'prereqs', 'openmm')
    version = extract_version(openmm_srcdir, "VERSION")
    if version is None:
        print('Unable to find openmm version')
        return
    import tarfile
    # openmm-7.3.0-linux-py37_cuda92_rc_1.tar.bz2
    tarfiles = openmm_srcdir.glob(f'openmm-{version}-*.tar*')
    for filename in tarfiles:
        # TODO: select tarfile for our platform
        with tarfile.open(filename) as f:
            try:
                member = f.getmember("licenses/Licenses.txt")
            except KeyError:
                continue
            licenses = f.extractfile(member)
            content = licenses.read()
            if not content:
                continue
            out_filename = pathlib.Path('licenses', 'openmm.txt')
            with open(out_filename, 'wb') as out:
                out.write(content)
            break


infos = get_packages_info()
infos.sort(key=(lambda info: info['name'].casefold()))

include_ffmpeg = 'ffmpeg' in sys.argv
# don't include openmm if it will be automatically included
include_openmm = not any(info['name'].casefold() == 'openmm' for info in infos)

os.makedirs('licenses', exist_ok=True)

with open('embedded.html.in') as src:
    with open('embedded.html', 'w') as out:
        for line in src.readlines():
            if line == 'PYTHON_PKGS\n':
                print_pkgs(infos, out)
            elif line == 'PYTHON_VERSION\n':
                print(platform.python_version(), file=out)
            elif line == 'GENERATED\n':
                from chimerax.core import buildinfo
                system = platform.system()
                chver = buildinfo.version
                date = buildinfo.date
                msg = f"This information was generatered for ChimeraX {chver} on {system} from {date}."
                print(msg, file=out)
            elif line == '<!--ffmpeg-->\n':
                if include_ffmpeg:
                    version = ffmpeg_version()
                    print(ffmpeg_doc.replace("VERSION", version), end='', file=out)
            elif line == '<!--openmm-->\n':
                if include_openmm:
                    print(openmm_doc, end='', file=out)
            else:
                print(line, end='', file=out)

try:
    if include_ffmpeg:
        ffmpeg_licenses()
except Exception:
    pass
try:
    if include_openmm:
        openmm_licenses()
except Exception:
    pass

raise SystemExit(0)
