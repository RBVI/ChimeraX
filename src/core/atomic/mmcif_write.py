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

"""
Write mmCIF files.

Currently only writes a minimal atom_site table.
No sequence, no secondary structure, no assemblies, no alt locs.
"""

def write_mmcif(session, path, format, models=None):
    from . import Structure
    if models is None:
        models = session.models.list(type = Structure)

    models = [m for m in models if isinstance(m, Structure)]
    nm = len(models)
    if nm == 0:
        from ..errors import UserError
        raise UserError('No structures specified')

    if nm > 1:
        session.logger.info('Writing %d models to mmCIF file by appending model number to chain ids' % nm)

    name = ''.join(models[0].name.split())				# Remove all whitespace from name
    name = name.encode('ascii', errors='ignore').decode('ascii')	# Drop non-ascii characters
    data_header = 'data_%s' % name
    
    alines = []
    entities = {}
    sclines = []
    srlines = []
    for mi, m in enumerate(models):
        cid_suffix = '' if nm == 1 else '%d' % (mi+1)
        alines.extend(atom_site_lines(m, len(alines), cid_suffix, entities))
        res = m.residues
        sclines.extend(struct_conf_lines(res, len(sclines), cid_suffix))
        srlines.extend(struct_sheet_range_lines(res, len(sclines), cid_suffix))

    lines = [data_header + '\n#\n']
    if alines:
        lines.append(atom_site_header + '\n'.join(alines) + '\n#\n')
    if sclines:
        lines.append(struct_conf_header + '\n'.join(sclines) + '\n#\n')
    if srlines:
        lines.append(struct_sheet_range_header + '\n'.join(srlines) + '\n#\n')
    text = ''.join(lines)
    
    f = open(path, 'w')
    f.write(text)
    f.close()

atom_site_header = '''loop_
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_alt_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_entity_id 
_atom_site.label_seq_id 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z 
_atom_site.occupancy 
_atom_site.B_iso_or_equiv 
_atom_site.pdbx_PDB_model_num
'''

def atom_site_lines(m, anum_offset, cid_suffix, entities):

    # Atoms of a residue need to be contiguous for the ChimeraX mmCIF reader to recognize
    # them as one residue.  Also the Phenix mmCIF reader seems to require residue atoms to be contiguous.
    # So output atoms in order of m.residues.
    atoms = m.residues.atoms
    xyz = atoms.scene_coords
    n = len(xyz)
    elem = atoms.element_names
    aname = atoms.names
    aloc = atoms.alt_locs
    occ = atoms.occupancy
    bfact = atoms.bfactors
    res = atoms.residues
    rname = res.names
    cid = res.chain_ids  # string fields need "." if blank.
    rnum = res.numbers
    eid = entity_ids(atoms, res, rname, entities)
    model_num = 1
    
    lines = [('%s %s %s %s %s %s %d %d %.3f %.3f %.3f %.2f %.2f %d' %
             (a+1+anum_offset, elem[a], aname[a], alt_loc_text(aloc[a]), rname[a], cid[a] + cid_suffix, eid[a], rnum[a],
              xyz[a,0], xyz[a,1], xyz[a,2], occ[a], bfact[a], model_num))
             for a in range(n)]
        
    return lines

def alt_loc_text(i):
    return '.' if i == 32 else chr(i)

#
# To determine entity id, use residue name if in_chain is false, use sequence if in_chain is true.
# Dictionary entities maps sequence or residue name to entity id.
#
def entity_ids(atoms, residues, rname, entities):
    n = len(residues)
    from numpy import empty, int32
    eids = empty((n,), int32)

    inch = atoms.in_chains
    ch = residues.filter(inch).chains  # chain.characters is sequence.

    ci = 0		# Index of atoms in chains
    seqs = {}		# Map chain to sequence to avoid creating sequence string for each atom
    for i in range(n):
        if inch[i]:
            chain = ch[ci]
            ci += 1
            seq = seqs.get(chain)
            if seq is None:
                seqs[chain] = seq = chain.characters
            eid = entities.get(seq)
            if eid is None:
                entities[seq] = eid = len(entities) + 1
        else:
            r = ('residue', rname[i])
            eid = entities.get(r)
            if eid is None:
                entities[r] = eid = len(entities) + 1
        eids[i] = eid

    return eids

struct_conf_header = '''loop_
_struct_conf.conf_type_id 
_struct_conf.id 
_struct_conf.beg_label_comp_id 
_struct_conf.beg_label_asym_id 
_struct_conf.beg_label_seq_id 
_struct_conf.end_label_comp_id 
_struct_conf.end_label_asym_id 
_struct_conf.end_label_seq_id 
'''

def struct_conf_lines(residues, hnum_offset, cid_suffix):
    h = residues.is_helix
    hse = intervals(h)	# List of (start,end) indices for helices
    rname = residues.names
    cid = residues.chain_ids
    rnum = residues.numbers
    lines = ['HELX_P %d %s %s %d %s %s %d' % (i+1+hnum_offset, rname[s], cid[s]+cid_suffix, rnum[s], rname[e], cid[e]+cid_suffix, rnum[e])
             for i, (s,e) in enumerate(hse)]
    return lines

def intervals(mask):
    '''Return list of intervals (start index, end index) of true values in mask array.'''
    indices = []
    if mask[0]:
        indices.append(-1)
    n = len(mask)
    d = (mask[:n-1] != mask[1:])
    indices.extend(d.nonzero()[0])
    if mask[n-1]:
        indices.append(n-1)
    iv = [(a+1,b) for a,b in zip(indices[::2], indices[1::2])]
    return iv

struct_sheet_range_header = '''loop_
_struct_sheet_range.sheet_id 
_struct_sheet_range.id 
_struct_sheet_range.beg_label_comp_id 
_struct_sheet_range.beg_label_asym_id 
_struct_sheet_range.beg_label_seq_id 
_struct_sheet_range.end_label_comp_id 
_struct_sheet_range.end_label_asym_id 
_struct_sheet_range.end_label_seq_id 
'''

def struct_sheet_range_lines(residues, snum_offset, cid_suffix):
    s = residues.is_strand
    sse = intervals(s)	# List of (start,end) indices for helices
    rname = residues.names
    cid = residues.chain_ids
    rnum = residues.numbers
    lines = ['. %d %s %s %d %s %s %d' % (i+1+snum_offset, rname[s], cid[s]+cid_suffix, rnum[s], rname[e], cid[e]+cid_suffix, rnum[e])
             for i, (s,e) in enumerate(sse)]
    return lines
