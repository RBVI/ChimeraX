# vim: set et ts=4 sts=4:
# === UCSF ChimeraX Copyright ====
# Copyright 2022 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ====
import sys
import toml

from packaging.requirements import Requirement
from typing import Optional

package_toolshed_name_map: dict[str, str] = {
    'chimerax.alphafold': 'ChimeraX-AlphaFold'
    , 'chimerax.alignments': 'ChimeraX-Alignments'
    , 'chimerax.blastprotein': 'ChimeraX-Blastprotein'
    , 'chimerax.core': 'ChimeraX-Core'
    , 'chimerax.ui': 'ChimeraX-UI'
    , 'cxwebservices': 'ChimeraX-WebServices'
}

def main(bundle_toml: dict):
    project_info = bundle_toml['project']
    authors = project_info['authors']
    bundle_name = package_toolshed_name_map[project_info['name']]
    chimerax_info = bundle_toml['tool']['chimerax']
    b_info = '<BundleInfo name="%s" version="%s" package="%s" minSessionVersion="%s" maxSessionVersion="%s">\n' \
             % (bundle_name, project_info['version'], project_info['name']
                , chimerax_info['min-session-version'], chimerax_info['max-session-version'])
    # Take only the first author name and email for now
    b_info += '\t<Author>%s</Author>\n' % authors[0]['name']
    b_info += '\t<Email>%s</Email>\n' % authors[0]['email']
    # Take only the first URL for now
    b_info += '\t<URL>%s</URL>\n' % list(project_info['urls'].values())[0]
    b_info += '\t<Synopsis>%s</Synopsis>\n' % project_info['synopsis']
    b_info += '\t<Description>\n%s\n\t</Description>\n' % project_info['description'].rstrip()
    b_info += '\t<Categories>\n'
    for category in chimerax_info['categories']:
        b_info += '\t\t<Category name="%s"/>\n' % category
    b_info += '\t</Categories>\n'
    b_info += '\t<Dependencies>\n'
    for dep in project_info['dependencies']:
        pkg = Requirement(dep)
        expected_pkg_name = package_toolshed_name_map[pkg.name]
        b_info += '\t\t<Dependency name="%s" version="%s"/>\n' % (expected_pkg_name, pkg.specifier)
    b_info += '\t</Dependencies>\n'
    b_info += '\t<Classifiers>\n'
    for classifier in project_info['classifiers']:
        b_info += '\t\t<PythonClassifier>%s</PythonClassifier>\n' % classifier
    for classifier in chimerax_info['chimerax-classifiers']:
        b_info += '\t\t<ChimeraXClassifier>%s</ChimeraXClassifier>\n' % classifier
    b_info += '\t</Classifiers>\n'
    b_info += '</BundleInfo>\n'
    with open('bundle_info.xml', 'w') as f:
        f.write(b_info)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        ... # print help and exit
    info: Optional[dict] = None
    try:
        with open(sys.argv[1]) as f:
            info = toml.loads(f.read())
    except toml.TomlDecodeError as e:
        print(str(e))
        sys.exit(1)
    main(info)
