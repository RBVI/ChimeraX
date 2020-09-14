#!/usr/bin/env python3
# vi: set shiftwidth=4 expandtab:
#
# Extract standard residue "templates" from aa-variants and components files
#
# The aa-variants file is used to get the C- and N-terminal versions
#
import gzip

standard_residues = set([
    # Amino acids
    "ALA", "ASX", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS",
    "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP",
    "TYR", "GLX",
    "UNK",      # unknown amino acid
    # Ribonucleotides
    "A", "G", "C", "U", "I",
    "N",        # unknown ribonucleotide
    # Deoxyribonucleotides
    "DA", "DG", "DC", "DT", "DI",
    "DN",       # unknown deoxyribonucleotide
    # Other
    "HOH",      # water
    "UNL",      # unknown ligand
    # other common residues (> 1000 in current PDB)
    'MSE', 'MLY', 'ACE', 'SEP', 'PSU', 'NH2', 'HYP', 'TPO',
])

found = set()

with open("stdresidues.cif", "w") as output:
    # Prefer components.cif entries because they are actively maintained
    # with gzip.open("components.cif.gz", "rt", encoding='utf-8') as f:
    with open("components.cif", "rt", encoding='utf-8') as f:
        save_lines = False
        for line in f.readlines():
            if line.startswith('data_'):
                ccd = line[5:].strip()
                if ccd in found:
                    continue
                save_lines = ccd in standard_residues
                if save_lines:
                    found.add(ccd)
            if save_lines:
                output.write(line)

    with open("aa-variants-v1.cif", "rt", encoding='utf-8') as f:
        save_lines = False
        for line in f.readlines():
            if line.startswith('data_'):
                ccd = line[5:].strip()
                if ccd in found:
                    continue
                save_lines = (ccd in standard_residues
                        or ccd.endswith("_LEO2H")
                        or ccd.endswith("_LSN3"))
                if save_lines:
                    found.add(ccd)
            if save_lines:
                output.write(line)

if found == standard_residues:
    raise SystemExit(0)

missing = standard_residues - found
if missing:
    print("Missing CCD entries for:", " ".join(missing))
else:
    print("All residues found")
raise SystemExit(1)
