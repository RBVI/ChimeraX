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

# $Id: __init__.py 41155 2016-06-30 23:18:29Z pett $

"""Find chemical groups in a structure"""

from chimerax.atomic.idatm import type_info, tetrahedral, planar, linear, single

# R is a shorthand for alkyl group
# X is a shorthand for 'any halide'
# None matches anything (and nothing)
from chimerax.atomic import Element
N = Element.get_element('N').number
C = Element.get_element('C').number
O = Element.get_element('O').number
H = Element.get_element('H').number
R = (C, H)
X = ('F', 'Cl', 'Br', 'I')
single_bond = (H, {'geometry':tetrahedral}, {'geometry':single})
heavy =    {'not type': ['H', 'HC', 'D', 'DC']}
non_oxygen_single_bond = (H, {'geometry':tetrahedral, 'not type': ['O', 'O3', 'O2', 'O3-', 'O2-', 'Oar', 'Oar+']}, {'geometry':single})

# 'R'/None and H must be found only in a list of descendents (not as the key
# element of a group or subgroup) and they must be arranged to be trailing
# components of a group description
#
# H can match nothing (and therefore R can match nothing)
#
# Note that 'R' in the textual group description is interpreted to mean:
# "C or H, unless H would produce a different functional group (then just C)"
#
#---
#
# For algorithmic group descriptions that aren't themselves functions,
# the following applies:
#
#    Atoms are indicated as either strings [idatm types] or numbers/
#    symbolic constants [element number], or None [any atom].  Tuples indicate
#    the atom can be any one of the alternatives in the tuple.
#
#    A list where an atom is expected is always two elements:  an atom,
#    and a list of what is bonded to that atom.
#
#    Dictionaries indicate facets of the idatm type (as per the
#    chimera.idatm.type_info dictionaries).  The keys of the dictionary
#    correspond to type_info dictionaries' keys, and the values are what
#    attribute value the atom must have to match.  The dictionary can
#    have a special key named 'default'.  The value of True for 'default'
#    means that atom types not in the type_info dictionary will match,
#    False means they won't.  The default for 'default' [urg] is False.  The
#    dictionary can also have a special 'not type' key, whose value is a
#    list of idatm types that aren't allowed to match.
#
#    Lastly, an "atom" can be an instance of RingAtom, which is initialized
#    with two arguments, the first of which is any of the preceding
#    forms of atom specification, and the second is the number of
#    intraresidue rings that the atom participates in 
#
#---
#
# The last component of the group description indicates which atoms are
# considered "principal" and are returned as part of the atom group.  This
# typically prunes the 'R's from a group.  A '1' indicates the atom is
# principal, a '0' that it is not.  Due to the algorithm for pruning, 
# 'optional' atoms (R and H) must appear at the end of the group description,
# with all R's before any H's.
#
# This last component is also used to indicate ring closures.  A value that
# is not '0' or '1' is assumed to the first of several such values.  Each
# atom with that value must be identical or the group will be pruned.  All
# atoms but the first with that value will be pruned from the returned group.
#

def find_ali_amines(structures, return_collection):
    results = [find_group("aliphatic " + order + " amine", structures, return_collection)
        for order in ("primary", "secondary", "tertiary", "quaternary")]
    from .support import collate_results
    return collate_results(results, return_collection)

def find_aro_amines(structures, return_collection, order=0):
    # if order == 0, return all aromatic amines
    from ._chem_group import find_aro_amines as faa
    from .support import call_c_plus_plus
    return call_c_plus_plus(faa, structures, return_collection, order)

def find_aromatics(structures, return_collection):
    from ._chem_group import find_aromatics as fa
    from .support import call_c_plus_plus
    return call_c_plus_plus(fa, structures, return_collection)

# specialized groups for hbond finding...
def find_ring_planar_NHR2(structures, return_collection, ring_size, aromatic_only=False):
    from ._chem_group import find_ring_planar_NHR2 as fr_pnhr2
    from .support import call_c_plus_plus
    return call_c_plus_plus(fr_pnhr2, structures, return_collection, ring_size, aromatic_only)

def find_5ring_planar_NR2(structures, return_collection, symmetric=False):
    from ._chem_group import find_5ring_planar_NR2 as f5r_pnr2
    from .support import call_c_plus_plus
    return call_c_plus_plus(f5r_pnr2, structures, return_collection, symmetric)

def find_6ring_planar_NR2(structures, return_collection, symmetric=False):
    from ._chem_group import find_6ring_planar_NR2 as f6r_pnr2
    from .support import call_c_plus_plus
    return call_c_plus_plus(f6r_pnr2, structures, return_collection, symmetric)

def find_5ring_OR2(structures, return_collection):
    from ._chem_group import find_5ring_OR2 as f5r_or2
    from .support import call_c_plus_plus
    return call_c_plus_plus(f5r_or2, structures, return_collection)

def find_nonring_NR2(structures, return_collection):
    from ._chem_group import find_nonring_NR2 as fnr_nr2
    from .support import call_c_plus_plus
    return call_c_plus_plus(fnr_nr2, structures, return_collection)

def find_nonring_ether(structures, return_collection):
    from ._chem_group import find_nonring_ether as fnr_ether
    from .support import call_c_plus_plus
    return call_c_plus_plus(fnr_ether, structures, return_collection)

class RingAtom:
    def __init__(self, atom_desc, num_rings):
        self.atom_desc = atom_desc
        self.num_rings = num_rings

# if a group is added here, or the synonyms just below group_info, it also has to be added
# to setup.py.in
from .support import collate_results
group_info = {
    "acyl halide":    ("R(C=O)X",    ['C2', [X, 'O2', single_bond]], [1,1,1,0]),
    "adenine":    ("6-aminopurine",
            ['Npl', [['Car', ['Car', ['N2', [['Car', [['N2', [['Car', ['Car', ['Npl', ['C3', ['Car', [['N2', ['Car']], H]]]]]]]], H]]]]]], H, H]],
                    [1,1,2,1,1,1,1,2,1,0,1,1,2,1,1,1,1]),
    "aldehyde":    ("R(C=O)H",    ['C2', ['O2', single_bond, H]], [1,1,0,1]),
    "amide":    ("R(C=O)NR2",    ['C2', ['O2', 'Npl', None]], [1,1,1,0]),
    "amine":    ("RxNHy",    lambda s, rc: collate_results((find_ali_amines(s, rc), find_aro_amines(s, rc)), rc), None),
    "aliphatic amine": ("RxNHy",    find_ali_amines, None),
    "aliphatic primary amine": ("RNH2",    [('N3','N3+'), ['C3', H, H, H]], [1,0,1,1,1]),
    "aliphatic secondary amine": ("R2NH",    [('N3','N3+'), ['C3', 'C3', H, H]], [1,0,0,1,1]),
    "aliphatic tertiary amine": ("R3N",        [('N3','N3+'), ['C3', 'C3', 'C3', H]], [1,0,0,0,1]),
    "aliphatic quaternary amine": ("R4N+",    ['N3+', ['C3', 'C3', 'C3', 'C3']], [1,0,0,0,0]),
    "aromatic amine": ("RxNHy",    find_aro_amines, None),
    "aromatic primary amine": ("RNH2",    lambda s, rc: find_aro_amines(s, rc, 1), None),
    "aromatic secondary amine": ("R2NH",    lambda s, rc: find_aro_amines(s, rc, 2), None),
    "aromatic tertiary amine": ("R3N",        lambda s, rc: find_aro_amines(s, rc, 3), None),
    "aromatic ring":("aromatic",    find_aromatics, None),
    "carbonyl":    ("R2C=O",        ['O2', [C]], [1,1]),
    "carboxylate":    ("RCOO-",    ['Cac', [['O2-', []], ['O2-', []], single_bond]], [1,1,1,0]),
    "cytosine":    ("2-oxy-4-aminopyrimidine",
                    ['Npl', [['C2', ['N2', ['C2', [['C2', [['Npl', ['C3', ['C2', ['N2', 'O2']]]], H]], H]]]], H, H]],
                    [1,1,2,1,1,1,0,1,2,1,1,1,1,1]),
    "disulfide":    ("RSSR",    ['S3', [['S3', [single_bond]], single_bond]], [1,1,0,0]),
    "ester":    ("R(C=O)OR",    ['C2', ['O2', [O, [C]], single_bond]], [1,1,1,0,0]),
    "ether O":    ("ROR",        ['O3', [C, C]], [1,0,0]),
    "guanine":    ("2-amino-6-oxypurine",
        ['Npl', [['C2', [['N2', ['Car']], ['Npl', [['C2', ['O2', ['Car', ['Car', ['N2', [['Car', [['Npl', ['C3', 'Car']], H]]]]]]]], H]]]], H, H]],
                    [1,1,1,2,1,1,1,1,2,1,1,1,0,2,1,1,1,1]),
    "halide":    ("RX",        [X, [heavy]], [1,0]),
    "hydroxyl":    ("COH or NOH",    ['O3', [(C, N), H]], [1,0,1]),
    "imine":    ("R2C=NR",    ['C2', [['N2', [single_bond]], single_bond, single_bond]], [1,1,0,0,0]),
    "ketone":    ("R2C=O",    ['C2', ['O2', C, C]], [1,1,0,0]),
    "methyl":    ("RCH3",    ['C3', [heavy, H, H, H]], [1,0,1,1,1]),
    "nitrile":    ("RC*N",    ['C1', ['N1', single_bond]], [1,1,0]),
    "nitro":    ("RNO2",    ['Ntr', ['O2-', 'O2-', non_oxygen_single_bond]], [1,1,1,0]),
    "phosphate":    ("PO4",        ['Pac', [O, O, O, O]], [1,1,1,1,1]),
    "phosphinyl":    ("R2PO2-",    ['Pac', [['O3-', []], ['O3-', []], R,R]], [1,1,1,0,0]),
    "phosphonate":    ("RPO3-",    ['Pac', ['O3-', 'O3-', 'O3-', C]], [1,1,1,1,0]),
    "purines":    ("purine-like rings",    [C, [C, RingAtom(N,1), [RingAtom(N,1), [[RingAtom(C,1), [[RingAtom(N,1), [[RingAtom(C,2), [RingAtom(C,2), [RingAtom(C,1), [[RingAtom(N,1), [[RingAtom(C,1), [RingAtom(N,1), None]], None]], None]]]], None]], None]], None]]]],
                    [3,2,4,1,1,1,2,3,1,1,1,4,0,0,0,0,0,0]),
    "pyrimidines":    ("pyrimidine-like rings",    [RingAtom(N,1), [RingAtom(C,1), [RingAtom(C,1), [[RingAtom(N,1), [[RingAtom(C,1), [[RingAtom(C,1), [RingAtom(C,1), None]], None]], None]], None]], None]],
                    [1,2,1,1,1,1,2,0,0,0,0,0]),
    "sulfate":    ("SO4",        ['Sac', [O, O, O, O]], [1,1,1,1,1]),
    "sulfonamide":    ("RSO2NR2",    ['Son', ['O2', 'O2', 'Npl', non_oxygen_single_bond]], [1,1,1,0,0]),
    "sulfonate":    ("RSO3-",    ['Sac', ['O3-', 'O3-', 'O3-', C]], [1,1,1,1,0]),
    "sulfone":    ("R2SO2",    ['Son', ['O2', 'O2', C, C]], [1,1,1,0,0]),
    "sulfonyl":    ("R2SO2",    ['Son', ['O2', 'O2', non_oxygen_single_bond, non_oxygen_single_bond]],
                    [1,1,1,0,0]),
    "thiocarbonyl":    ("C=S",        ['S2', [C]], [1,1]),
    "thioether":    ("RSR",        ['S3', [C, C]], [1,0,0]),
    "thiol":    ("RSH",        ['S3', [C, H]], [1,0,1]),
    "thymine":    ("5-methyl-2,4-dioxypyrimidine",
                ['C3', [['C2', ['C2', ['C2', ['O2', ['Npl', [['C2', ['O2', ['Npl', ['C3', ['C2', ['C2', H]]]]]], H]]]]]], H, H, H]],
                    [1,2,3,1,1,1,1,1,1,0,3,2,1,1,1,1,1]),
    "uracil":    ("2,4-dioxypyrimidine",
                    ['O2', [['C2', ['Npl', ['Npl', [['C2', ['O2', ['C2', [['C2', [['Npl', ['C2', 'C3']], H]], H]]]], H]]]]]],
                    [1,2,3,1,1,1,1,1,3,2,0,1,1,1])
}

# synonyms
for group_name in list(group_info.keys()):
    if group_name.startswith("sulf"):
        group_info["sulph" + group_name[4:]] = group_info[group_name]
group_info["aromatic"] = group_info["aromatic ring"]

def find_group(group_desc, structures, return_collection=False):
    
    if isinstance(group_desc, str):
        try:
            group_formula, group_rep, group_principals = group_info[group_desc]
        except Exception:
            raise KeyError("No known chemical group named '%s'" % group_desc)
    else:
        group_rep, group_principals = group_desc
    
    if callable(group_rep):
        return group_rep(structures, return_collection)
    
    from ._chem_group import find_group as fg
    from .support import call_c_plus_plus
    return call_c_plus_plus(fg, structures, return_collection,
        group_rep, group_principals, RingAtom)

def register_selectors(logger):
    def select(results, models, group):
        from .chem_group import find_group
        atoms = find_group(group, models, return_collection=True)
        for m in atoms.unique_structures:
            results.add_model(m)
        results.add_atoms(atoms)
    from chimerax.core.commands import register_selector
    for group_name in group_info.keys():
        register_selector(group_name.replace(' ', '-'),
            lambda ses, models, results, gn=group_name: select(results, models, gn),
            logger)

