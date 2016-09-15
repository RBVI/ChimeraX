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

def write_mmcif(session, path, models, **kw):
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

    lines = []
    entities = {}
    for mi, m in enumerate(models):
        cid_suffix = '' if nm == 1 else '%d' % (mi+1)
        atom_site_lines(m, cid_suffix, lines, entities)
             
    lines.append('#')

    atom_site_header = '''loop_
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
_atom_site.occupancy 
_atom_site.B_iso_or_equiv 
'''

    text = atom_site_header + '\n'.join(lines) + '\n'
    
    f = open(path, 'w')
    f.write(text)
    f.close()

def atom_site_lines(m, cid_suffix, lines, entities):
    
    atoms = m.atoms
    xyz = atoms.scene_coords
    n = len(xyz)
    elem = atoms.element_names
    aname = atoms.names
    occ = atoms.occupancy
    bfact = atoms.bfactors
    res = atoms.residues
    rname = res.names
    cid = res.chain_ids  # string fields need "." if blank.
    rnum = res.numbers
    eid = entity_ids(atoms, res, rname, entities)

    for a in range(n):
        line = ('%s %s %s %s %s %d %d %.3f %.3f %.3f %.2f %.2f' %
                (a+1, elem[a], aname[a], rname[a], cid[a] + cid_suffix, eid[a], rnum[a],
                 xyz[a,0], xyz[a,1], xyz[a,2], occ[a], bfact[a]))
        lines.append(line)

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
