# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California. All rights reserved.
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

"""
SNFG (Symbol Nomenclature for Glycans) definitions.

Based on the SNFG standard:
  Varki A, Cummings RD, Aebi M, et al. Symbol Nomenclature for Graphical
  Representations of Glycans. Glycobiology. 2015;25(12):1323-1324.

  Updates: https://www.ncbi.nlm.nih.gov/glycans/snfg.html

3D-SNFG implementation based on:
  Thieker DF, Hadden JA, Schulten K, Woods RJ. 3D Implementation of the Symbol
  Nomenclature for Graphical Representation of Glycans. Glycobiology. 2016;26(8):786-787.
"""

# SNFG Colors - RGB values from the official SNFG specification
# These are the recommended RGB conversions from CMYK
COLORS = {
    'white':      (255, 255, 255),
    'blue':       (0, 114, 188),
    'green':      (0, 166, 81),
    'yellow':     (255, 212, 0),
    'light_blue': (143, 204, 233),
    'pink':       (246, 158, 161),
    'purple':     (165, 67, 153),
    'brown':      (161, 122, 77),
    'orange':     (244, 121, 32),
    'red':        (237, 28, 36),
}

# Shape types used in SNFG
SPHERE = 'sphere'           # Hexoses
CUBE = 'cube'               # N-acetylhexosamines (HexNAc)
CROSSED_CUBE = 'crossed_cube'  # Hexosamines (HexN)
DIAMOND = 'diamond'         # Hexuronic acids (HexA) and Sialic acids
CONE = 'cone'               # Deoxyhexoses
DIVIDED_CONE = 'divided_cone'  # N-acetyl deoxyhexosamines
RECTANGLE = 'rectangle'     # Di-deoxyhexoses
STAR = 'star'               # Pentoses
HEXAGON = 'hexagon'         # Heptoses and specialized sugars
PENTAGON = 'pentagon'       # Ketoses
FLAT_DIAMOND = 'flat_diamond'  # Nonulosonic acids (9-carbon)

# Monosaccharide types to (shape, color) mapping
# Based on SNFG Table 1: https://www.ncbi.nlm.nih.gov/glycans/snfg.html
MONOSACCHARIDE_SYMBOLS = {
    # Hexoses (Sphere)
    'Glc': (SPHERE, 'blue'),
    'Man': (SPHERE, 'green'),
    'Gal': (SPHERE, 'yellow'),
    'Gul': (SPHERE, 'orange'),
    'Alt': (SPHERE, 'pink'),
    'All': (SPHERE, 'purple'),
    'Tal': (SPHERE, 'light_blue'),
    'Ido': (SPHERE, 'brown'),

    # N-Acetylhexosamines (Cube) - HexNAc
    'GlcNAc': (CUBE, 'blue'),
    'ManNAc': (CUBE, 'green'),
    'GalNAc': (CUBE, 'yellow'),
    'GulNAc': (CUBE, 'orange'),
    'AltNAc': (CUBE, 'pink'),
    'AllNAc': (CUBE, 'purple'),
    'TalNAc': (CUBE, 'light_blue'),
    'IdoNAc': (CUBE, 'brown'),

    # Hexosamines (Crossed Cube) - HexN
    'GlcN': (CROSSED_CUBE, 'blue'),
    'ManN': (CROSSED_CUBE, 'green'),
    'GalN': (CROSSED_CUBE, 'yellow'),
    'GulN': (CROSSED_CUBE, 'orange'),
    'AltN': (CROSSED_CUBE, 'pink'),
    'AllN': (CROSSED_CUBE, 'purple'),
    'TalN': (CROSSED_CUBE, 'light_blue'),
    'IdoN': (CROSSED_CUBE, 'brown'),

    # Hexuronic acids (Diamond) - HexA
    'GlcA': (DIAMOND, 'blue'),
    'ManA': (DIAMOND, 'green'),
    'GalA': (DIAMOND, 'yellow'),
    'GulA': (DIAMOND, 'orange'),
    'AltA': (DIAMOND, 'pink'),
    'AllA': (DIAMOND, 'purple'),
    'TalA': (DIAMOND, 'light_blue'),
    'IdoA': (DIAMOND, 'brown'),

    # Deoxyhexoses (Cone)
    'Qui': (CONE, 'blue'),       # 6-deoxy-Glc (Quinovose)
    'Rha': (CONE, 'green'),      # 6-deoxy-Man (Rhamnose)
    '6dGul': (CONE, 'orange'),   # 6-deoxy-Gul
    '6dAlt': (CONE, 'pink'),     # 6-deoxy-Alt
    '6dTal': (CONE, 'light_blue'),  # 6-deoxy-Tal
    'Fuc': (CONE, 'red'),        # 6-deoxy-Gal (Fucose)

    # N-acetyl deoxyhexosamines (Divided Cone)
    'QuiNAc': (DIVIDED_CONE, 'blue'),
    'RhaNAc': (DIVIDED_CONE, 'green'),
    'FucNAc': (DIVIDED_CONE, 'red'),
    '6dAltNAc': (DIVIDED_CONE, 'pink'),
    '6dTalNAc': (DIVIDED_CONE, 'light_blue'),

    # Di-deoxyhexoses (Rectangle)
    'Oli': (RECTANGLE, 'blue'),      # Olivose
    'Tyv': (RECTANGLE, 'green'),     # Tyvelose
    'Abe': (RECTANGLE, 'orange'),    # Abequose
    'Par': (RECTANGLE, 'pink'),      # Paratose
    'Dig': (RECTANGLE, 'purple'),    # Digitoxose
    'Col': (RECTANGLE, 'light_blue'), # Colitose

    # Pentoses (Star)
    'Ara': (STAR, 'green'),      # Arabinose (L)
    'Lyx': (STAR, 'yellow'),     # Lyxose
    'Xyl': (STAR, 'orange'),     # Xylose
    'Rib': (STAR, 'pink'),       # Ribose

    # Sialic acids and nonulosonic acids (Diamond)
    'Kdn': (DIAMOND, 'green'),       # 2-keto-3-deoxy-D-glycero-D-galacto-nononic acid
    'Neu5Ac': (DIAMOND, 'purple'),   # N-acetylneuraminic acid
    'Neu5Gc': (DIAMOND, 'light_blue'), # N-glycolylneuraminic acid
    'Neu': (DIAMOND, 'brown'),       # Neuraminic acid
    'Sia': (DIAMOND, 'red'),         # Generic sialic acid

    # Nonulosonic acids - 3,9-dideoxy (Flat Diamond)
    'Pse': (FLAT_DIAMOND, 'green'),  # Pseudaminic acid
    'Leg': (FLAT_DIAMOND, 'yellow'), # Legionaminic acid
    'Aci': (FLAT_DIAMOND, 'pink'),   # Acinetaminic acid
    '4eLeg': (FLAT_DIAMOND, 'light_blue'),

    # Heptoses and specialized sugars (Hexagon)
    'Bac': (HEXAGON, 'blue'),        # Bacillosamine
    'LDManHep': (HEXAGON, 'green'),  # L-glycero-D-manno-heptose
    'Kdo': (HEXAGON, 'yellow'),      # 3-deoxy-D-manno-oct-2-ulosonic acid
    'Dha': (HEXAGON, 'orange'),      # 3-deoxy-D-lyxo-hept-2-ulosaric acid
    'DDManHep': (HEXAGON, 'pink'),   # D-glycero-D-manno-heptose
    'MurNAc': (HEXAGON, 'purple'),   # N-acetylmuramic acid
    'MurNGc': (HEXAGON, 'light_blue'), # N-glycolylmuramic acid
    'Mur': (HEXAGON, 'brown'),       # Muramic acid

    # Ketoses (Pentagon)
    'Api': (PENTAGON, 'blue'),       # Apiose
    'Fru': (PENTAGON, 'green'),      # Fructose
    'Tag': (PENTAGON, 'yellow'),     # Tagatose
    'Sor': (PENTAGON, 'orange'),     # Sorbose
    'Psi': (PENTAGON, 'pink'),       # Psicose
}

# PDB residue code to SNFG monosaccharide type mapping
# Based on wwPDB Chemical Component Dictionary carbohydrate standardization
# See: https://www.wwpdb.org/documentation/carbohydrate-remediation
PDB_TO_SNFG = {
    # Glucose variants
    'GLC': 'Glc',   # alpha-D-glucose
    'BGC': 'Glc',   # beta-D-glucose
    'AGC': 'Glc',   # alpha-D-glucose (newer code)

    # Mannose variants
    'MAN': 'Man',   # alpha-D-mannose
    'BMA': 'Man',   # beta-D-mannose

    # Galactose variants
    'GAL': 'Gal',   # beta-D-galactose
    'GLA': 'Gal',   # alpha-D-galactose

    # N-acetylglucosamine variants
    'NAG': 'GlcNAc',  # N-acetyl-beta-D-glucosamine
    'NDG': 'GlcNAc',  # N-acetyl-alpha-D-glucosamine

    # N-acetylgalactosamine
    'NGA': 'GalNAc',  # N-acetyl-D-galactosamine
    'A2G': 'GalNAc',  # N-acetyl-alpha-D-galactosamine

    # N-acetylmannosamine
    'BM3': 'ManNAc',  # N-acetyl-beta-D-mannosamine

    # Fucose variants
    'FUC': 'Fuc',   # alpha-L-fucose
    'FUL': 'Fuc',   # beta-L-fucose
    'FCB': 'Fuc',   # beta-D-fucose

    # Xylose
    'XYS': 'Xyl',   # alpha-D-xylose
    'XYP': 'Xyl',   # beta-D-xylopyranose
    'LXC': 'Xyl',   # L-xylopyranose

    # Ribose
    'RIB': 'Rib',   # D-ribose
    'BDR': 'Rib',   # beta-D-ribofuranose

    # Arabinose
    'ARA': 'Ara',   # alpha-L-arabinose
    'ARB': 'Ara',   # beta-L-arabinose

    # Glucosamine
    'GCS': 'GlcN',  # D-glucosamine
    'PA1': 'GlcN',  # alpha-D-glucosamine

    # Glucuronic acid
    'GCU': 'GlcA',  # alpha-D-glucuronic acid
    'BDP': 'GlcA',  # beta-D-glucuronic acid

    # Galacturonic acid
    'GTR': 'GalA',  # alpha-D-galacturonic acid

    # Iduronic acid
    'IDS': 'IdoA',  # alpha-L-iduronic acid
    'IDR': 'IdoA',  # L-iduronic acid

    # Sialic acids
    'SIA': 'Sia',   # N-acetylneuraminic acid
    'SLB': 'Neu5Ac', # sialyl-lactose (contains Neu5Ac)

    # Rhamnose
    'RAM': 'Rha',   # alpha-L-rhamnose
    'RM4': 'Rha',   # methyl alpha-L-rhamnopyranoside

    # Fructose
    'FRU': 'Fru',   # D-fructose
    'BDF': 'Fru',   # beta-D-fructofuranose

    # KDO
    'KDO': 'Kdo',   # 3-deoxy-D-manno-oct-2-ulosonic acid

    # Muramic acid derivatives
    'MUB': 'MurNAc', # N-acetylmuramic acid
    'AMU': 'MurNAc', # N-acetylmuramic acid

    # Heptoses
    'GMH': 'LDManHep', # L-glycero-D-manno-heptose

    # ==========================================================================
    # GLYCAM force field residue codes
    # The GLYCAM naming uses 3-letter codes encoding: sugar type, ring form,
    # anomeric configuration, enantiomeric form, and linkage positions.
    # See: https://glycam.org/docs/forcefield/glycam-naming-2/
    # ==========================================================================

    # GLYCAM Sialic acids (Neu5Ac)
    '0SA': 'Neu5Ac',  # alpha-Neu5Ac
    '0SB': 'Neu5Ac',  # beta-Neu5Ac
    '4SA': 'Neu5Ac',  # 4-linked alpha-Neu5Ac
    '4SB': 'Neu5Ac',  # 4-linked beta-Neu5Ac
    '7SA': 'Neu5Ac',  # 7-linked alpha-Neu5Ac
    '7SB': 'Neu5Ac',  # 7-linked beta-Neu5Ac
    '8SA': 'Neu5Ac',  # 8-linked alpha-Neu5Ac
    '8SB': 'Neu5Ac',  # 8-linked beta-Neu5Ac
    '9SA': 'Neu5Ac',  # 9-linked alpha-Neu5Ac
    '9SB': 'Neu5Ac',  # 9-linked beta-Neu5Ac

    # GLYCAM N-acetylgalactosamine (GalNAc) - "V" codes
    '0VA': 'GalNAc',  # terminal alpha-GalNAc
    '0VB': 'GalNAc',  # terminal beta-GalNAc
    '3VA': 'GalNAc',  # 3-linked alpha-GalNAc
    '3VB': 'GalNAc',  # 3-linked beta-GalNAc
    '4VA': 'GalNAc',  # 4-linked alpha-GalNAc
    '4VB': 'GalNAc',  # 4-linked beta-GalNAc
    '6VA': 'GalNAc',  # 6-linked alpha-GalNAc
    '6VB': 'GalNAc',  # 6-linked beta-GalNAc

    # GLYCAM Galactose - "L" codes
    '0LA': 'Gal',     # terminal alpha-Gal
    '0LB': 'Gal',     # terminal beta-Gal
    '2LA': 'Gal',     # 2-linked alpha-Gal
    '2LB': 'Gal',     # 2-linked beta-Gal
    '3LA': 'Gal',     # 3-linked alpha-Gal
    '3LB': 'Gal',     # 3-linked beta-Gal
    '4LA': 'Gal',     # 4-linked alpha-Gal
    '4LB': 'Gal',     # 4-linked beta-Gal
    '6LA': 'Gal',     # 6-linked alpha-Gal
    '6LB': 'Gal',     # 6-linked beta-Gal
    'ZLA': 'Gal',     # 2,3-linked alpha-Gal
    'ZLB': 'Gal',     # 2,3-linked beta-Gal
    'YLA': 'Gal',     # 3,4-linked alpha-Gal
    'YLB': 'Gal',     # 3,4-linked beta-Gal
    'XLA': 'Gal',     # 2,6-linked alpha-Gal
    'XLB': 'Gal',     # 2,6-linked beta-Gal
    'WLA': 'Gal',     # 3,6-linked alpha-Gal
    'WLB': 'Gal',     # 3,6-linked beta-Gal
    'VLA': 'Gal',     # 4,6-linked alpha-Gal
    'VLB': 'Gal',     # 4,6-linked beta-Gal

    # GLYCAM N-acetylglucosamine (GlcNAc) - "Y" codes
    # Note: "Y" is the GLYCAM one-letter code for GlcNAc
    '0YA': 'GlcNAc',  # terminal alpha-GlcNAc
    '0YB': 'GlcNAc',  # terminal beta-GlcNAc
    '1YA': 'GlcNAc',  # 1-linked alpha-GlcNAc
    '1YB': 'GlcNAc',  # 1-linked beta-GlcNAc
    '3YA': 'GlcNAc',  # 3-linked alpha-GlcNAc
    '3YB': 'GlcNAc',  # 3-linked beta-GlcNAc
    '4YA': 'GlcNAc',  # 4-linked alpha-GlcNAc
    '4YB': 'GlcNAc',  # 4-linked beta-GlcNAc
    '6YA': 'GlcNAc',  # 6-linked alpha-GlcNAc
    '6YB': 'GlcNAc',  # 6-linked beta-GlcNAc
    'WYA': 'GlcNAc',  # 3,6-linked alpha-GlcNAc
    'WYB': 'GlcNAc',  # 3,6-linked beta-GlcNAc
    'VYA': 'GlcNAc',  # 4,6-linked alpha-GlcNAc
    'VYB': 'GlcNAc',  # 4,6-linked beta-GlcNAc
    'UYA': 'GlcNAc',  # 3,4-linked alpha-GlcNAc
    'UYB': 'GlcNAc',  # 3,4-linked beta-GlcNAc
    'QYA': 'GlcNAc',  # 3,4,6-linked alpha-GlcNAc
    'QYB': 'GlcNAc',  # 3,4,6-linked beta-GlcNAc
    'YYA': 'GlcNAc',  # 2,4-linked alpha-GlcNAc (or other branching)
    'YYB': 'GlcNAc',  # 2,4-linked beta-GlcNAc

    # GLYCAM Glucose - "G" codes
    '0GA': 'Glc',     # terminal alpha-Glc
    '0GB': 'Glc',     # terminal beta-Glc
    '1GA': 'Glc',     # 1-linked alpha-Glc
    '1GB': 'Glc',     # 1-linked beta-Glc
    '2GA': 'Glc',     # 2-linked alpha-Glc
    '2GB': 'Glc',     # 2-linked beta-Glc
    '3GA': 'Glc',     # 3-linked alpha-Glc
    '3GB': 'Glc',     # 3-linked beta-Glc
    '4GA': 'Glc',     # 4-linked alpha-Glc
    '4GB': 'Glc',     # 4-linked beta-Glc
    '6GA': 'Glc',     # 6-linked alpha-Glc
    '6GB': 'Glc',     # 6-linked beta-Glc
    'WGA': 'Glc',     # 3,6-linked alpha-Glc
    'WGB': 'Glc',     # 3,6-linked beta-Glc
    'VGA': 'Glc',     # 4,6-linked alpha-Glc
    'VGB': 'Glc',     # 4,6-linked beta-Glc

    # GLYCAM Mannose - "M" codes
    '0MA': 'Man',     # terminal alpha-Man
    '0MB': 'Man',     # terminal beta-Man
    '2MA': 'Man',     # 2-linked alpha-Man
    '2MB': 'Man',     # 2-linked beta-Man
    '3MA': 'Man',     # 3-linked alpha-Man
    '3MB': 'Man',     # 3-linked beta-Man
    '4MA': 'Man',     # 4-linked alpha-Man
    '4MB': 'Man',     # 4-linked beta-Man
    '6MA': 'Man',     # 6-linked alpha-Man
    '6MB': 'Man',     # 6-linked beta-Man
    'ZMA': 'Man',     # 2,3-linked alpha-Man
    'ZMB': 'Man',     # 2,3-linked beta-Man
    'YMA': 'Man',     # 3,6-linked alpha-Man
    'YMB': 'Man',     # 3,6-linked beta-Man
    'XMA': 'Man',     # 2,6-linked alpha-Man
    'XMB': 'Man',     # 2,6-linked beta-Man
    'WMA': 'Man',     # 3,6-linked alpha-Man (alternative)
    'WMB': 'Man',     # 3,6-linked beta-Man
    'VMA': 'Man',     # 4,6-linked alpha-Man (branched)
    'VMB': 'Man',     # 4,6-linked beta-Man (branched)
    'UMA': 'Man',     # 3,4-linked alpha-Man
    'UMB': 'Man',     # 3,4-linked beta-Man
    'QMA': 'Man',     # 3,4,6-linked alpha-Man
    'QMB': 'Man',     # 3,4,6-linked beta-Man

    # GLYCAM Fucose - "F" codes
    '0FA': 'Fuc',     # terminal alpha-Fuc
    '0FB': 'Fuc',     # terminal beta-Fuc
    '2FA': 'Fuc',     # 2-linked alpha-Fuc
    '2FB': 'Fuc',     # 2-linked beta-Fuc
    '3FA': 'Fuc',     # 3-linked alpha-Fuc
    '3FB': 'Fuc',     # 3-linked beta-Fuc
    '4FA': 'Fuc',     # 4-linked alpha-Fuc
    '4FB': 'Fuc',     # 4-linked beta-Fuc

    # GLYCAM Xylose - "X" codes
    '0XA': 'Xyl',     # terminal alpha-Xyl
    '0XB': 'Xyl',     # terminal beta-Xyl
    '2XA': 'Xyl',     # 2-linked alpha-Xyl
    '2XB': 'Xyl',     # 2-linked beta-Xyl
    '3XA': 'Xyl',     # 3-linked alpha-Xyl
    '3XB': 'Xyl',     # 3-linked beta-Xyl
    '4XA': 'Xyl',     # 4-linked alpha-Xyl
    '4XB': 'Xyl',     # 4-linked beta-Xyl

    # GLYCAM Glucuronic acid - "Z" codes (note: different from branching codes)
    '0ZA': 'GlcA',    # terminal alpha-GlcA
    '0ZB': 'GlcA',    # terminal beta-GlcA
    '4ZA': 'GlcA',    # 4-linked alpha-GlcA
    '4ZB': 'GlcA',    # 4-linked beta-GlcA

    # GLYCAM Iduronic acid - "U" codes
    '0UA': 'IdoA',    # terminal alpha-IdoA
    '0UB': 'IdoA',    # terminal beta-IdoA
    '4UA': 'IdoA',    # 4-linked alpha-IdoA
    '4UB': 'IdoA',    # 4-linked beta-IdoA
}

# Ring atom names commonly found in sugar rings
# Used to identify pyranose (6-membered) and furanose (5-membered) rings
PYRANOSE_RING_ATOMS = {'C1', 'C2', 'C3', 'C4', 'C5', 'O5'}  # or O4 for furanoses
FURANOSE_RING_ATOMS = {'C1', 'C2', 'C3', 'C4', 'O4'}

# Default shape size in Angstroms (sphere diameter)
# Per 3D-SNFG: "4 Å diameter in the case of a sphere"
DEFAULT_SIZE = 4.0
