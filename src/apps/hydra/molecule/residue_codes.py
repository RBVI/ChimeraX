regex3to1 = {
	'A':'A',		# Nucleic bases
	'+A':'A',		# Nucleic bases
	'ADE':'A',
	'C':'C',
	'+C':'C',
	'CYT':'C',
	'DA': 'A',
	'DC': 'C',
	'DG': 'G',
	'DT': 'T',
	'G':'G',
	'+G':'G',
	'GUA':'G',
	'T':'T',
	'+T':'T',
	'THY':'T',
	'U':'U',
	'+U':'U',
	'URA':'U',

	'ALA':'A',		# Amino acids
	'ARG':'R',
	'ASH':'[DB]',		# Amber (protonated ASP)
	'ASN':'[NB]',		# ASN or ASX
	'ASP':'[DB]',		# ASP or ASX
	'CYS':'C',
	'CYX':'C',		# Amber (disulfide)
	'GLH':'[EZ]',		# Amber (protonated GLU)
	'GLU':'[EZ]',		# GLU or GLX
	'GLN':'[QZ]',		# GLN or GLX
	'GLY':'G',
	'HID':'H',		# Amber (delta protonated)
	'HIE':'H',		# Amber (epsilon protonated)
	'HIP':'H',		# Amber (doubly protonated)
	'HIS':'H',
	'HYP':'P',
	'ILE':'I',
	'LEU':'L',
	'LYS':'K',
	'MET':'M',
	'MSE':'M',		# Selenomethianine
	'PHE':'F',
	'PRO':'P',
	'SER':'S',
	'THR':'T',
	'TRP':'W',
	'TYR':'Y',
	'VAL':'V',
	'ASX':'[BND]',		# ASX or ASN or ASP
	'GLX':'[ZEQ]'		# GLX or GLN or GLU
}

nucleic3to1 = {			# Nucleic bases
	'A':'A',		# adenine
	'+A':'A',
	'ADE':'A',
	'C':'C',		# cytosine
	'+C':'C',
	'CYT':'C',
	'DA': 'A',		# deoxy form of A
	'DC': 'C',		# deoxy form of C
	'DG': 'G',		# deoxy form of G
	'DT': 'T',		# deoxy form of T
	'G':'G',		# guanine
	'+G':'G',
	'GUA':'G',
	'T':'T',		# thymine
	'+T':'T',
	'THY':'T',
	'U':'U',		# uracil
	'+U':'U',
	'URA':'U'
}

protein3to1 = {
	'ALA':'A',		# Amino acids
	'ARG':'R',
	'ASH':'D',		# Amber (protonated ASP)
	'ASN':'N',
	'ASP':'D',
	'CYS':'C',
	'CYX':'C',		# Amber (disulfide)
	'GLH':'E',		# Amber (protonated GLU)
	'GLU':'E',
	'GLN':'Q',
	'GLY':'G',
	'HID':'H',		# Amber (delta protonated)
	'HIE':'H',		# Amber (epsilon protonated)
	'HIP':'H',		# Amber (doubly protonated)
	'HIS':'H',
	'HYP':'P',		# using 'O' proved problematic with
				# similarity matricies/conservation scores
	'ILE':'I',
	'LEU':'L',
	'LYS':'K',
	'MET':'M',
	'MSE':'M',		# Selenomethianine
	'PHE':'F',
	'PRO':'P',
	'SER':'S',
	'THR':'T',
	'TRP':'W',
	'TYR':'Y',
	'VAL':'V',
	'ASX':'B',
	'GLX':'Z'
}

protein1to3 = {
	'A':'ALA',
	'B':'ASX',
	'C':'CYS',
	'D':'ASP',
	'E':'GLU',
	'F':'PHE',
	'G':'GLY',
	'H':'HIS',
	'I':'ILE',
	'K':'LYS',
	'L':'LEU',
	'M':'MET',
	'N':'ASN',
	'O':'HYP',
	'P':'PRO',
	'Q':'GLN',
	'R':'ARG',
	'S':'SER',
	'T':'THR',
	'V':'VAL',
	'W':'TRP',
	'Y':'TYR',
	'Z':'GLX'
}

class StdResDict(dict):
        pass

standard3to1 = StdResDict()
standard3to1.update(nucleic3to1)
standard3to1.update(protein3to1)

def res3to1(code3):
        return standard3to1.get(code3, 'X')
