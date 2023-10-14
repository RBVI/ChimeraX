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

sel_info = {}

from chimerax.atomic import Element, Atom
# Since IDATM has types in conflict with element symbols (e.g. 'H'),
# put the types in first so that the elements 'win'
for idatm, info in Atom.idatm_info_map.items():
    sel_info[idatm] = info.description
for i in range(1,Element.NUM_SUPPORTED_ELEMENTS):
    name = Element.get_element(i).name
    sel_info[name] = "%s (element)" % name

# classifiers
selectors = ["    <ChimeraXClassifier>"
             "ChimeraX :: Selector :: %s :: %s"
             "</ChimeraXClassifier>\n" %
             (name, description) for name, description in sel_info.items()]

with open("bundle_info.xml.in") as f:
    content = f.read()
with open("bundle_info.xml", "w") as f:
    f.write(content.replace("ELEMENT_AND_IDATM_SELECTOR_CLASSIFIERS", "".join(selectors)))
raise SystemExit(0)
