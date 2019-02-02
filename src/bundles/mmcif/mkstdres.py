#!/usr/bin/env python
#
# Extract standard residue "templates" from components file
#
from __future__ import print_function
import gzip

standard_residues = {
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
}

found = set()

with open("stdresidues.cif", "w") as output:
    with gzip.open("components.cif.gz", "rU") as f:
        save_lines = False
        for line in f.readlines():
            if line.startswith('data_'):
                ccd = line[5:].strip()
                save_lines = ccd in standard_residues
                if save_lines:
                    found.add(ccd)
            if save_lines:
                output.write(line)

if found == standard_residues:
    raise SystemExit(0)

print("Missing CCD entries for:", " ".join(standard_residues - found))
raise SystemExit(1)
