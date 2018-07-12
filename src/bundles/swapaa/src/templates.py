#
# Create mmcif file of templates for 20 standard amino acids for swapping amino acids
# using coordinates from PDB chemical components dictionary.
#
# For swapping amino acids the output is edited to delete 3 hydrogens, one on aspartate and
# one on glutamate that are deprotonated at neutral pH, and the amide hydrogen on proline.
#

cc_path = '/Users/goddard/Downloads/Components-pub.cif'
templ_path = 'templates.cif'

def amino_acid_template_mmcif(chem_components_path):
    resnames = set(['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                    'SER', 'THR', 'TRP', 'TYR', 'VAL'])
    f = open(cc_path, 'r')
    lines = chem_comp_atom_lines(f, resnames)
    f.close()

    templ_mmcif = atom_site_table(lines)
    return templ_mmcif

def chem_comp_atom_lines(file, resnames):
    in_cca = False
    lines = []
    while True:
        line = file.readline()
        if not line:
            break
        if line.startswith('_chem_comp_atom.pdbx_ordinal'):
            in_cca = True
        if in_cca:
            if line.startswith('#'):
                in_cca = False
            else:
                if line[:3] in resnames:
                    lines.append(line)
    return lines

def atom_site_table(lines, exclude_atoms = ['OXT', 'H2', 'HXT'], asym_id = 'A', entity_id = '1'):
    alines = []
    anum = 1
    snum = 0
    last_rname = None
    for line in lines:
        fields = line.split()
        rname, aname, atype, x, y, z = (fields[0], fields[1], fields[3],
                                        float(fields[12]), float(fields[13]), float(fields[14]))
        if aname not in exclude_atoms:
            if rname != last_rname:
                snum += 1
                last_rname = rname
            alines.append('ATOM %3d %1s %4s %3s %1s %1s %d %8.3f %8.3f %8.3f'
                          % (anum, atype, aname, rname, entity_id, asym_id, snum, x, y, z))
            anum += 1

    header = \
'''
loop_
_atom_site.group_PDB 
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_entity_id 
_atom_site.label_seq_id 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z 
'''

    templ_mmcif = '#%s%s\n#\n' % (header, '\n'.join(alines))
    return templ_mmcif

def write_templates(cc_path, templ_path):
    templ_mmcif = amino_acid_template_mmcif(cc_path)
    f = open(templ_path, 'w')
    f.write(templ_mmcif)
    f.close()

#write_templates(cc_path, templ_path)
