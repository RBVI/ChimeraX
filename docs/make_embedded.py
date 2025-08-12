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
"""
Generate license informatin for embedded packages
"""

import copy
from html import escape
import os
import pathlib
import platform
import importlib
import importlib.metadata
import re
import shutil
import subprocess

FFMPEG_DOC = """
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

OPENMM_DOC = """
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

    key_map = {
        "Name": 'name',
        "Version": 'version',
        "License": 'license',
        "Summary": 'summary',
        "Home-page": 'homepage',
    }
    empty_info = {}
    for key, name in key_map.items():
        empty_info[name] = ""

    infos = []
    for pkg in importlib.metadata.distributions():
        if pkg.name.startswith('ChimeraX-'):
            continue
        info = copy.deepcopy(empty_info)
        classifier_licenses = []
        for key, value in pkg.metadata.items():
            value = value.strip()
            if key in key_map:
                info[key_map[key]] = value
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
    """Get absolute path to a file with the license for a package if found"""

    license_re = re.compile('^(LICENSE|COPYING)')
    license_files = []
    top_levels = []

    for path in pkg.files:
        if not path.parts[0].endswith("-info"):
            continue
        if path.name == "top_level.txt":
            with open(pkg.locate_file(path), 'rt', encoding='utf-8') as f:
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
        for t in top_levels:
            try:
                m = importlib.import_module(t)
            except (ImportError, ValueError):
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
    """"Return HMTL4 version of package name"""
    # must match [A-Za-z][-A-Za-z0-9_:.]*
    # Python package names must start with letter, so don't check
    def cvt(c):
        if c.isalnum() or c in '-_:.':
            return c
        return '_'
    return ''.join(cvt(c) for c in name)


def print_pkgs(infos, out):
    """Output package information"""
    ids = [html4_id(info['name']) for info in infos]
    print('<p>Packages:', file=out)
    sep = ' '
    for info, id in zip(infos, ids):
        print(f'{sep}<a href="#{id}">{info["name"]}</a>', end='', file=out)
        sep = ', '
    print('.', file=out)
    print('<dl>', file=out)
    for info, id in zip(infos, ids):
        print(f'<dt><a href="{escape(info["homepage"])}" id="{id}" target="_blank">{escape(info["name"])}</a> version {escape(info["version"])}', file=out)
        print(f'<dd>{escape(info["summary"])}<br>', file=out)
        license = info['license']
        if license:
            lf = info['license-file']
            if not lf:
                print(f'License type: {escape(license)}', file=out)
            else:
                fn = f'licenses/{id}-{os.path.basename(lf)}'
                if not os.path.splitext(fn)[1]:
                    fn += '.txt'
                shutil.copyfile(lf, fn)
                print(f'License type: <a href="{fn}">{escape(license)}</a>', file=out)
    print('</dl>', file=out)


def get_ffmpeg_version():
    """Get installed ffpeg's version"""
    try:
        output = subprocess.check_output(
            ["../ChimeraX.app/bin/ffmpeg", "-version"],
            encoding='utf-8',
            env={'LANG': 'en_US.UTF-8'}
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    first_line = output.split('\n', maxsplit=1)[0].split()
    if first_line[0:2] != ['ffmpeg', 'version']:
        return None
    return first_line[2]


def ffmpeg_licenses():
    """Get ffpmeg licences"""
    import markdown
    license_file = pathlib.Path('..', 'prereqs', 'ffmpeg', 'LICENSE.md')
    try:
        with open(license_file, 'r', encoding='utf-8') as f:
            license = f.read()
        html = markdown.markdown(license)
    except FileNotFoundError:
        html = "<html><body>ffmpeg licenses not found</body</html>"
    with open('licenses/ffmpeg-LICENSE.html', 'w', encoding='utf-8') as f:
        f.write(html)


def write_embedded():
    """Output embedded.html"""
    infos = get_packages_info()
    infos.sort(key=lambda info: info['name'].casefold())

    ffmpeg_version = get_ffmpeg_version()
    include_ffmpeg = ffmpeg_version is not None
    # don't include openmm if it will be automatically included
    include_openmm = not any(info['name'].casefold() == 'openmm' for info in infos)

    os.makedirs('licenses', exist_ok=True)

    with open('embedded.html.in', encoding='utf-8') as src:
        with open('embedded.html', 'w', encoding='utf-8') as out:
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
                    msg = f"This information was generated from ChimeraX {chver} for {system} on {date}."
                    print(msg, file=out)
                elif line == '<!--ffmpeg-->\n':
                    if include_ffmpeg:
                        print(FFMPEG_DOC.replace("VERSION", ffmpeg_version), end='', file=out)
                elif line == '<!--openmm-->\n':
                    if include_openmm:
                        print(OPENMM_DOC, end='', file=out)
                else:
                    print(line, end='', file=out)

    if include_ffmpeg:
        ffmpeg_licenses()


if __name__ == '__main__':
    write_embedded()
