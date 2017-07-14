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
    "purine": "purine",
    "pyrimidine": "pyrimidine",
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

# synonyms
for group_name in list(group_info.keys()):
    if group_name.startswith("sulf"):
        group_info["sulph" + group_name[4:]] = group_info[group_name]
group_info["aromatic"] = group_info["aromatic ring"]

# classifiers
selectors = ["<ChimeraXClassifier>"
             "ChimeraX :: Selector :: %s :: %s"
             "</ChimeraXClassifier>\n" %
             (group_name.replace(' ', '-'), alt_name)
             for group_name, alt_name in group_info.items()]

with open("bundle_info.xml.in") as f:
    content = f.read()
with open("bundle_info.xml", "w") as f:
    f.write(content.replace("SELECTOR_CLASSIFIERS", "".join(selectors)))
raise SystemExit(0)
