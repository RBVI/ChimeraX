# vim: set expandtab ts=4 sw=4:

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

from chimerax.core.state import State  # For session saving

class Variant(State):
    def __init__(self, hgvs_protein, hgvs_nucleotide = None):
        self.hgvs_protein = hgvs_protein
        self.hgvs_nucleotide = hgvs_nucleotide	# Used to distinguish alternate codons

        self.residue_number, self.from_aa, self.to_aa, self.change = _parse_hgvs(hgvs_protein)

    def __hash__(self):
        return hash((self.hgvs_protein, self.hgvs_nucleotide))

    def __eq__(self, v):
        return v.hgvs_protein == self.hgvs_protein and v.hgvs_nucleotide == self.hgvs_nucleotide

    @property
    def is_synonymous(self):
        return self.to_aa == self.from_aa and self.to_aa is not None

    def residue_variant(self):
        '''For handling per-residue computed scores we use a synonymous mutation variant.'''
        if self.residue_number:
            hgvs_pro = f'p.{self.from_aa}{self.residue_number}{self.from_aa}'
            return Variant(hgvs_pro)
        return None
    
    def take_snapshot(self, session, flags):
        data = {'hgvs_protein': self.hgvs_protein, 'version': 1}
        if self.hgvs_nucleotide:
            data['hgvs_nucleotide'] = self.hgvs_nucleotide
        return data
    
    @classmethod
    def restore_snapshot(cls, session, data):
        return cls(data['hgvs_protein'], data.get('hgvs_nucleotide'))


# ------------------------------------------------------------------------------------
#
aa_3_to_1 = {'Cys':'C', 'Asp':'D', 'Ser':'S', 'Gln':'Q', 'Lys':'K',
             'Ile':'I', 'Pro':'P', 'Thr':'T', 'Phe':'F', 'Asn':'N', 
             'Gly':'G', 'His':'H', 'Leu':'L', 'Arg':'R', 'Trp':'W', 
             'Ala':'A', 'Val':'V', 'Glu':'E', 'Tyr':'Y', 'Met':'M'}
aa_1 = set(aa_3_to_1.values())

# ------------------------------------------------------------------------------------
# Example human genome variant society (hgvs) notation.
#
# p.(T396N)
# p.Val27Asp
# p.Arg2Ter	stop codon
# p.Arg2*	stop codon (non-compliant)
# p.Arg2=	synonymous
# p.(S357S)	synonymous
# p.(A104del)
# p.(A104_N106del)
# p.(Ala104_Arg106del)
# p.(S309del1)	(non-compliant, from Willow)
# p.(S308del2)	(non-compliant, from Willow)
# p.(T14_D15insG)
# p.(E3_L4insGSG)
# p.(Glu3_Lys4insGSG)
# p.(E3insGSG)	(non-compliant, from Willow)
# p.[Lys48Arg;Lys101Glu]
# p.[Leu9Phe;Met62Val;Val86Ala;Ile107Asn;Lys146Asn]
# 
def _parse_hgvs(hgvs):

    res_num = from_aa = to_aa = change = None

    if not hgvs.startswith('p.'):
        # Not a protein variant
        return res_num, from_aa, to_aa, change

    if ';' in hgvs:
        # Multi-position variant
        return res_num, from_aa, to_aa, change

    # Strip p. and parentheses.
    var = hgvs[2:]
    if var.startswith('(') and var.endswith(')'):
        var = var[1:-1]

    # One-letter codes
    from_aa, rest = _prefix_amino_acid(var)
    if from_aa is not None:
        res_num, rest = _prefix_integer(rest)
        if res_num is None:
            from_aa = None
        else:
            to_aa, rest = _prefix_amino_acid(rest)
            if to_aa is None:
                if rest == '=':
                    to_aa = from_aa
                elif rest == '*' or rest == 'Ter':
                    change = 'stop'
                elif rest.startswith('del') or rest.startswith('ins'):
                    change = rest
                elif rest.startswith('_'):
                    from_aa2, rest = _prefix_amino_acid(rest[1:])
                    if from_aa2 is None:
                        from_aa = res_num = None
                    else:
                        res_num2, rest = _prefix_integer(rest)
                        if res_num2 is None:
                            from_aa = res_num = None
                        else:
                            if rest.startswith('del'):
                                change = f'del{res_num2-res_num+1}'
                            elif rest.startswith('ins'):
                                change = rest
                            else:
                                from_aa = res_num = None
                else:
                    from_aa = res_num = None

    return res_num, from_aa, to_aa, change

def _prefix_amino_acid(string):
    if string[:3] in aa_3_to_1:
        return aa_3_to_1[string[:3]], string[3:]
    elif string[:1] in aa_1:
        return string[:1], string[1:]
    return None, string

def _prefix_integer(string):
    digits = []
    for c in string:
        if c.isdigit():
            digits.append(c)
        else:
            break
    digits = ''.join(digits)
    return (int(digits), string[len(digits):]) if digits else (None, string)

def _is_multiresidue_mutation(hgvs_nt):
    if hgvs_nt.startswith('c.[') and hgvs_nt.endswith(']'):
        mutations = hgvs_nt[3:-1].split(';')
        resnums = set()
        for mut in mutations:
            if 'delins' in mut:
                # Example mut = "241_243delinsCTG"
                rnum0, rnum1 = [1+(int(i)-1)//3 for i in mut[:mut.index('delins')].split('_')]
                for rnum in range(rnum0, rnum1+1):
                    resnums.add(rnum)
            else:
                # Example mut = "371A>G"
                rnum = 1+(int(mut[:-3])-1)//3
                resnums.add(rnum)
        return len(resnums) > 1
    return False

def variant_parsing_problems(mutation_set, path, max_problems = 3):
    # Find HGVS parsing errors.
    hgvs_parse_errors = []
    for ms in mutation_set.mutation_scores:
        v = ms.variant
        if ';' not in v.hgvs_protein and v.residue_number is None:
            hgvs_parse_errors.append(v)

    if hgvs_parse_errors:
        hgvs_problems = [f'{v.hgvs_protein} line {v.line_number}' for v in hgvs_parse_errors[:max_problems]]
        nprob = len(hgvs_parse_errors)
        if nprob > max_problems:
            hgvs_problems.append('...')
        from os.path import basename
        filename = basename(path)
        warnings = f'Failed to parse {nprob} variants in {filename}: {", ".join(hgvs_problems)}'
    else:
        warnings = ''

    return warnings
