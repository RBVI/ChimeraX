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

from chimerax.atomic import Residue

N_H = 1.01
def bond_with_H_length(heavy, geom):
    element = heavy.element.name
    if element == "C":
        if geom == 4:
            return 1.09
        if geom == 3:
            return 1.08
        if geom == 2:
            return 1.056
    elif element == "N":
        return N_H
    elif element == "O":
        # can't rely on water being in chain "water" anymore...
        if heavy.num_bonds == 0 or heavy.num_bonds == 2 \
        and len([nb for nb in heavy.neighbors if nb.element.number > 1]) == 0:
            return 0.9572
        return 0.96
    elif element == "S":
        return 1.336
    from chimerax.atomic import Element
    return Element.bond_length(heavy.element, Element.get_element(1))

def complete_terminal_carboxylate(session, cter):
    from chimerax.atomic.bond_geom import bond_positions
    from chimerax.atomic.struct_edit import add_atom
    from chimerax.atomic import Element
    c = cter.find_atom("C")
    if not c:
        return
    missing_O = missing_OXT = True
    for nb in c.neighbors:
        if nb.name == "O":
            missing_O = False
        elif nb.name == "OXT":
            missing_OXT = False
        else:
            continue
        if len([nnb for nnb in nb.neighbors if nb.element.number > 1]) > 1:
            session.logger.warning("C-terminal %s does not look like carboxylate oxygen;"
                " cannot complete teminus" % str(nb))
            return
    if missing_O and missing_OXT:
        session.logger.warning("Both O and OXT missing from C-terminal residue %s; cannot complete teminus"
            % str(cter))
        return
    if not missing_O and not missing_OXT:
        return
    missing_name = "OXT" if missing_OXT else "O"
    if c.num_bonds != 2:
        return
    loc = bond_positions(c.coord, 3, 1.229, [n.coord for n in c.neighbors])[0]
    oxt = add_atom(missing_name, Element.get_element("O"), cter, loc, bonded_to=c)
    from chimerax.atomic.colors import element_color
    if c.color == element_color(c.element.number):
        oxt.color = element_color(oxt.element.number)
    else:
        oxt.color = c.color
    session.logger.info("Missing %s added to C-terminal residue %s" % (missing_name, str(cter)))

def determine_termini(session, structs):
    real_N = []
    real_C = []
    fake_N = []
    fake_C = []
    fake_5p = []
    fake_3p = []
    logger = session.logger
    for s in structs:
        sr_res = set()
        for chain in s.chains:
            if chain.from_seqres:
                sr_res.update(chain.residues)
                rn, rc, fn, fc, f5p, f3p = termini_from_seqres(chain)
                logger.info("Termini for %s determined from SEQRES records" % chain.full_name)
            else:
                rn, rc, fn, fc, f5p, f3p = guess_termini(chain)
                logger.info("No usable SEQRES records for %s; guessing termini instead"
                    % chain.full_name)
            real_N.extend(rn)
            real_C.extend(rc)
            fake_N.extend(fn)
            fake_C.extend(fc)
            fake_5p.extend(f5p)
            fake_3p.extend(f3p)
        if sr_res:
            # Look for peptide termini not in SEQRES records
            from chimerax.atomic import Sequence
            protein3to1 = Sequence.protein3to1
            for r in s.residues:
                if r in sr_res:
                    continue
                if protein3to1(r.name) == 'X':
                    continue
                ca = r.find_atom("CA")
                o = r.find_atom("O")
                n = r.find_atom("N")
                c = r.find_atom("C")
                if ca and o and n and c:
                    for atom_name, termini in [('N', real_N), ('C', real_C)]:
                        for nb in r.find_atom(atom_name).neighbors:
                            if nb.residue != r:
                                break
                        else:
                            termini.append(r)

    return real_N, real_C, fake_N, fake_C, fake_5p, fake_3p

def termini_from_seqres(chain):
    real_N = []
    real_C = []
    fake_N = []
    fake_C = []
    fake_5p = []
    fake_3p = []
    if chain.polymer_type == Residue.PT_AMINO:
        if chain.residues[0]:
            real_N.append(chain.residues[0])
        if chain.residues[-1]:
            real_C.append(chain.residues[-1])

        last = chain.residues[0]
        for res in chain.residues[1:]:
            if res:
                if not last:
                    fake_N.append(res)
            else:
                if last:
                    fake_C.append(last)
            last = res
    else:
        for res in chain.residues:
            if not res:
                continue
            if res != chain.residues[0]:
                fake_5p.append(res)
            break
        for res in reversed(chain.residues):
            if not res:
                continue
            if res != chain.residues[-1]:
                fake_3p.append(res)
            break
    return real_N, real_C, fake_N, fake_C, fake_5p, fake_3p

def guess_termini(seq):
    real_N = []
    real_C = []
    fake_N = []
    fake_C = []
    fake_5p = []
    fake_3p = []
    if seq.polymer_type == Residue.PT_AMINO:
        residues = seq.residues
        existing_residues = seq.existing_residues
        n_term = residues[0]
        if not n_term:
            fake_N.append(existing_residues[0])
        elif cross_residue(n_term, 'N'):
            fake_N.append(n_term)
        else:
            n = n_term.find_atom('N')
            if n:
                if n.num_bonds == 2 and 'H' in [nb.name for nb in n.neighbors]:
                    fake_N.append(n_term)
                else:
                    real_N.append(n_term)
            else:
                real_N.append(n_term)
        c_term = residues[-1]
        if not c_term:
            fake_C.append(existing_residues[-1])
        elif cross_residue(c_term, 'C'):
            fake_C.append(c_term)
        else:
            c = c_term.find_atom('C')
            if c:
                if c.num_bonds == 2 and 'O' in [nb.name for nb in c.neighbors]:
                    fake_C.append(c_term)
                else:
                    real_C.append(c_term)
            else:
                real_C.append(c_term)
        for i, res in enumerate(existing_residues[:-1]):
            next_res = existing_residues[i+1]
            if res.connects_to(next_res):
                continue
            if res.number + 1 < next_res.number:
                fake_C.append(res)
                fake_N.append(next_res)
    return real_N, real_C, fake_N, fake_C, fake_5p, fake_3p

def cross_residue(res, at_name):
    a = res.find_atom(at_name)
    if a:
        for nb in a.neighbors:
            if a.residue != nb.residue:
                return True
    return False

naming_exceptions = {
    'ATP': {
        "N6": ["HN61", "HN62"],
    },
    'ADP': {
        "N6": ["HN61", "HN62"],
    },
    'GTP': {
        "N1": ["HN1"],
        "N2": ["HN21", "HN22"]
    },
    'GDP': {
        "N1": ["HN1"],
        "N2": ["HN21", "HN22"]
    }
}

def determine_naming_schemas(structure, type_info):
    """Determine for each residue, method for naming hydrogens

    The possible schemas are:
        1) 'prepend' -- put 'H' in front of atom element name
        2) a set of hetero atoms that should be prepended (others will have the element symbol replaced
            with 'H'); in this case the prepend will be in front of the entire atom name
        3) 'simple' -- first hydrogen is H1, second is H2, etc.  Used if there are 4-character heavy
            atom names in the residue.
    
    In the first two cases, a number will be appended if more than one hydrogen is to be added.
    (Unless the base atom name ends in a prime ['] character, in which case additional primes
    will be added as long as the resulting name is 4 characters or less)
    
    The "set" is the preferred scheme and is used when the heavy atoms have been given reasonably
    distinctive names.  'Prepend' is used mostly in small molecules where the atoms have names such as 'C1',
    'C2', 'C3', 'N1', 'N2', etc. and 'replace' would not work."""

    schemas = {structure: 3} # default to PDB version 3 naming
    for residue in structure.residues:
        if schemas[structure] == 3 and residue.name == "T" and residue.find_atom("O1P"):
            # DNA thymine is "DT" in version 3
            # (but RNA still has A/C/G so can't check those)
            schemas[structure] = 2 # PDB version 2

        heavy_names = [len(a.name) for a in residue.atoms if a.element.number > 1]
        if heavy_names and max(heavy_names) >= 4:
            schemas[residue] = 'simple'
            continue
        carbons = set()
        hets = set()
        identifiers = set()
        for atom in residue.atoms:
            # skip pre-existing hydrogens
            if atom.element.number == 1:
                # use :1 instead of 0 in case atom name is empty
                if not atom.name[:1].isalpha():
                    schemas[structure] = 2 # PDB version 2
                if atom.num_bonds == 1:
                    nb = atom.neighbors[0]
                    if nb.name == 'N' and atom.name == 'H':
                        # this 'exception' occurs in standard amino
                        # acids so, don't allow it to force prepend...
                        continue
                    if nb in type_info and type_info[nb].substituents - nb.num_bonds >= 1:
                        # the neighbor will have more protons added to it,
                        # so skip its identifier...
                        continue
                    if not nb.name.startswith(nb.element.name):
                        schemas[residue] = 'prepend'
                        break
                    if atom.name[:1].isdigit():
                        atom_test_name = atom.name[2:] + atom.name[:1]
                    else:
                        atom_test_name = atom.name[1:]
                    if atom_test_name.startswith(nb.name):
                        identifiers.add(nb.name[len(nb.element.name):])
                    elif atom_test_name.startswith(nb.name[len(nb.element.name):]):
                        identifiers.add(nb.name[len(nb.element.name):])
                    else:
                        schemas[residue] = 'prepend'
                        break
                continue

            # make sure leading characters are atomic symbol (otherwise use 'prepend')
            symbol = atom.element.name
            if len(atom.name) < len(symbol) or atom.name[0:len(symbol)].upper() != symbol.upper():
                schemas[residue] = 'prepend'
                break

            # if this atom won't have hydrogens added, we don't care
            if atom not in type_info:
                continue
            num_to_add = type_info[atom].substituents - atom.num_bonds
            if num_to_add < 1:
                continue

            # if this atom has explicit naming given, don't enter in identifiers dict
            res_name = atom.residue.name
            if res_name in naming_exceptions and atom.name in naming_exceptions[res_name]:
                continue
            if atom.element.name == "C":
                carbons.add(atom)
            else:
                hets.add(atom)
        else:
            dups = set()
            for c in carbons:
                identifier = c.name[1:]
                if identifier in identifiers:
                    schemas[residue] = 'prepend'
                    break
                identifiers.add(identifier)

            else:
                het_identifiers = set()
                for het in hets:
                    identifier = het.name[len(het.element.name):]
                    if identifier in identifiers:
                        if identifier in het_identifiers:
                            schemas[residue] = 'prepend'
                            break
                        else:
                            dups.add(het)
                            het_identifiers.add(identifier)
                    else:
                        identifiers.add(identifier)
                else:
                    schemas[residue] = dups
    return schemas
