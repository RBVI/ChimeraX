# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

group_info = {
    "acyl halide": "R(C=O)X",
    "adenine": "6-aminopurine",
    "aldehyde": "R(C=O)H",
    "amide": "R(C=O)NR2",
    "amine": "RxNHy",
    "aliphatic amine": "RxNHy",
    "aliphatic primary amine": "RNH2",
    "aliphatic secondary amine": "R2NH",
    "aliphatic tertiary amine": "R3N",
    "aliphatic quaternary amine": "R4N+",
    "aromatic amine": "RxNHy",
    "aromatic primary amine": "RNH2",
    "aromatic secondary amine": "R2NH",
    "aromatic tertiary amine": "R3N",
    "aromatic ring": "aromatic",
    "carbonyl": "R2C=O",
    "carboxylate": "RCOO-",
    "cytosine": "2-oxy-4-aminopyrimidine",
    "disulfide": "RSSR",
    "ester": "R(C=O)OR",
    "ether O": "ROR",
    "guanine": "2-amino-6-oxypurine",
    "halide": "RX",
    "hydroxyl": "COH or NOH",
    "imine": "R2C=NR",
    "ketone": "R2C=O",
    "methyl": "RCH3",
    "nitrile": "RC*N",
    "nitro": "RNO2",
    "phosphate": "PO4",
    "phosphinyl": "R2PO2-",
    "phosphonate": "RPO3-",
    "purines": "purine-like rings",
    "pyrimidines": "pyrimidine-like rings",
    "sulfate": "SO4",
    "sulfonamide": "RSO2NR2",
    "sulfonate": "RSO3-",
    "sulfone": "R2SO2",
    "sulfonyl": "R2SO2",
    "thiocarbonyl": "C=S",
    "thioether": "RSR",
    "thiol": "RSH",
    "thymine": "5-methyl-2,4-dioxypyrimidine",
    "uracil": "2,4-dioxypyrimidine",
}
group_names = list(group_info.keys())

# synonyms
for group_name in list(group_info.keys()):
    if group_name.startswith("sulf"):
        group_info["sulph" + group_name[4:]] = group_info[group_name]
group_info["aromatic"] = group_info["aromatic ring"]

# classifiers
selectors = ["<ChimeraXClassifier>"
             "Selector :: %s :: %s"
             "</ChimeraXClassifier>\n" %
             (group_name.replace(' ', '-'), alt_name)
             for group_name, alt_name in group_info.items()]

with open("bundle_info.xml.in") as f:
    content = f.read()
with open("bundle_info.xml", "w") as f:
    f.write(content.replace("SELECTOR_CLASSIFIERS", "".join(selectors)))

with open("src/__init__.py.in") as f:
    content = f.read()
amine_endings = ['', ' primary', ' secondary', ' tertiary', ' quaternary']
amines = ["all"] + ['aliphatic' + ending for ending in amine_endings] + ['aromatic' + ending
    for ending in amine_endings[:-1]]
bases = []
base_names = set(['adenine', 'cytosine', 'guanine', 'thymine', 'uracil'])
# ribose is from atomic, but is more natural in this menu
menu_info = [('&amine', amines), ('&nucleoside base', bases), ('ribose', None)]
for name in group_names:
    if name.endswith('amine'):
        continue
    elif name in base_names:
        bases.append(name)
    else:
        menu_info.append((name, None))
bases.sort()
menu_info.sort(key=lambda n: n[0][1:] if n[0][0] == '&' else n[0])
with open("src/__init__.py", "w") as f:
    f.write(content.replace("SELECTOR_NAMES",
        ", ".join([repr(s) for s in menu_info])))
raise SystemExit(0)
