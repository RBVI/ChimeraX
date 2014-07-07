def open_mmcif_file(path, session):
  '''
  Open an mmCIF file.
  '''
#  from time import time
#  t0 = time()
  from .pdb import use_pdbio
  if use_pdbio:
    mols = open_mmcif_file_with_pdbio(path, session)
  else:
    mols = open_mmcif_file_with_image3d(path, session)
#  t1 = time()
#  print('opened in %.2f sec, %s' % (t1-t0,path))
  return mols

def open_mmcif_file_with_image3d(path, session):
  from time import time
  ft0 = time()
  f = open(path, 'r')
  text = f.read()
  f.close()
  ft1 = time()
  from .. import _image3d
  t0 = time()
  matoms = _image3d.parse_mmcif_file(text, sort_residues = True)
  t1 = time()
  from . import pdb
  from . import Molecule, connect
  mols = []
  for a in matoms:
    atoms = pdb.atom_array(a)
    m = Molecule(path, atoms)
    m.color_by_chain()
    bonds, missing = connect.molecule_bonds(m, session)
    m.bonds = bonds
    mols.append(m)

#  from os.path import basename
#  session.show_info('Read %s %d atoms at %d per second\n' %
#                    (basename(path), len(xyz), int(len(xyz)/(ft1-ft0))))
#  session.show_info('Parsed %s %d atoms at %d per second\n' %
#                    (basename(path), len(xyz), int(len(xyz)/(t1-t0))))
#  session.show_info('Read+Parsed %s %d atoms at %d per second\n' %
#                    (basename(path), len(xyz), int(len(xyz)/((t1-t0)+(ft1-ft0)))))
#  session.show_info('Read %s %d atoms\n' % (basename(path), len(xyz)))

  return mols

def open_mmcif_file_with_pdbio(path, session):
  from time import time
  ft0 = time()
  f = open(path, 'rb')
  text = f.read()
  f.close()
  ft1 = time()
  from .. import pdbio
  t0 = time()
  sblob = pdbio.parse_mmCIF_file(text)
  t1 = time()

  from . import pdb
  mols = []
  from . import Molecule
  for sb in sblob.structures:
    atoms, bonds = pdb.structblob_atoms_and_bonds(sblob)
    m = Molecule(path, atoms)
# TODO: pdbio.parse_mmCIF_file() is not creating bonds
#  from . import connect
#  bonds, missing = connect.molecule_bonds(m, session)
    m.bonds = bonds
    m.color_by_chain()
    mols.append(m)

  return mols

def load_mmcif_local(id, session, mmcif_dir):
  '''Load an mmCIF file given its id from a local copy of the "divided" database.'''
  from os.path import join, exists
  p = join(mmcif_dir, id[1:3].lower(), '%s.cif' % id.lower())
  if not exists(p):
    return None
  mols = open_mmcif_file(p, session)
  return mols

def mmcif_sequences(mmcif_path):
  '''
  Read an mmcif file to find how residue numbers map to sequence positions.
  This is not available in PDB format.
  '''
  eps, sa, en = read_mmcif_tables(mmcif_path, ('_entity_poly_seq', '_struct_asym', '_entity'))
  if sa is None or eps is None:
    print('Missing sequence info in mmCIF file %s (_entity_poly_seq and _struct_asym tables)' % mmcif_path)
    return {}
  ce = sa.mapping('id', 'entity_id')
  es = eps.mapping('num', 'mon_id', foreach = 'entity_id')
  ed = en.mapping('id', 'pdbx_description')

  eseq = {}
  from .residue_codes import res3to1
  for eid, seq in es.items():
    rnums = [int(i) for i in seq.keys()]
    rnums.sort()
    r0,r1 = rnums[0], rnums[-1]
    if rnums != list(range(r0,r1+1)):
      from os.path import basename
      print(basename(mmcif_path), 'non-contiguous sequence for entity', eid, 'residue numbers', rnums)
      continue
    desc = ed.get(eid,'')
    eseq[eid] = (r0, ''.join(res3to1(seq[str(i)]) for i in rnums), desc)

  cseq = {}
  for cid, eid in ce.items():
    if eid in eseq:
      cseq[cid] = eseq[eid]
  
  return cseq

def sequence_residue_numbers(mmcif_path):
  '''
  Read an mmcif file to find how residue numbers map to sequence positions.
  This is not available in PDB format.
  '''
  pseq = '_pdbx_poly_seq_scheme.'
  f = open(mmcif_path)
  c = 0
  ccid = csnum = crnum = None
  while True:
    line = f.readline()
    if line.startswith(pseq):
      if line.startswith(pseq + 'asym_id'):
        ccid = c
      elif line.startswith(pseq + 'seq_id'):
        csnum = c
      elif line.startswith(pseq + 'pdb_seq_num'):
        crnum = c
      c += 1
    elif not ccid is None or line == '':
      break
  if ccid is None or csnum is None or crnum is None:
    f.close()
    return {}
  cr2s = {}
  while True:
    fields = line.split()
    cid = fields[ccid]
    snum = fields[csnum]
    rnum = fields[crnum]
    if rnum == '?':
      continue
    if not cid in cr2s:
      cr2s[cid] = {}
    r2s = cr2s[cid]
    r2s[int(rnum)] = int(snum)
    line = f.readline()
    if line.startswith('#') or line == '':
      break
  f.close()
  return cr2s

def read_mmcif_tables(mmcif_path, table_names):
  f = open(mmcif_path)
  tables = {}
  tname = None
  vcontinue = False
  semicolon_quote = False
  while True:
    line = f.readline()
    if tname is None:
      if line == '':
        break
      for tn in table_names:
        if line.startswith(tn + '.'):
          tname = tn
          tags = []
          values = []
          break
      if tname is None:
        continue
    if line.startswith(tname + '.'):
      tvalue = line.split('.', maxsplit=1)[1]
      tfields = tvalue.split(maxsplit = 1)
      tag = tfields[0]
      tags.append(tag)
      if len(tfields) == 2:
        value = remove_quotes(tfields[1])
        if values:            # Tags have values on same line without loop.
          values[0].append(value)
        else:
          values.append([value])
      elif values:
        # Other tags have values, so this one must have value on next line, e.g. 1afi _entity table.
        # Should really be looking for loop_.
        vcontinue = True
    elif line.startswith('#') or line == '':
      if [v for v in values if len(v) != len(tags)]:
        # Error: Number of values doesn't match number of tags.
        print (mmcif_path, tags, values)
      tables[tname] = mmCIF_Table(tname, tags, values)
      tname = None
    else:
      if line.startswith(';'):
        # Fields can extend onto next line if that line is preceded by a semicolon.
        # The whole line is treated as a single values as if quoted.
        lval = semicolon_quote = line[1:].rstrip()
        if lval:
          values[-1].append(lval)
      elif semicolon_quote:
        # Line that starts with semicolon continues on following lines until a line with only a semicolon.
        values[-1][-1] += line.rstrip()
      elif vcontinue:
        # Values simply continue on next line sometimes (e.g. 207l.cif _entity table).
        values[-1].extend(combine_quoted_values(line.split()))
      else:
        # New line of values
        values.append(combine_quoted_values(line.split()))
      vcontinue = (len(values[-1]) < len(tags))
        
  f.close()
  tlist = [tables.get(tn, None) for tn in table_names]
  return tlist

def combine_quoted_values(values):
  qvalues = []
  in_quote = False
  for e in values:
    if in_quote:
      if e.endswith(in_quote):
        qv.append(e[:-1])
        qvalues.append(' '.join(qv))
        in_quote = False
      else:
        qv.append(e)
    elif e.startswith("'") or e.startswith('"'):
      q = e[0]
      if e.endswith(q):
        qvalues.append(e[1:-1])
      else:
        in_quote = q
        qv = [e[1:]]
    else:
      qvalues.append(e)
  return qvalues

def remove_quotes(s):
  t = s.strip()
  if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
    return t[1:-1]
  return t

class mmCIF_Table:
  def __init__(self, table_name, tags, values):
    self.table_name = table_name
    self.tags = tags
    self.values = values
  def mapping(self, key_name, value_name, foreach = None):
    t = self.tags
    for n in (key_name, value_name, foreach):
      if n and not n in t:
        raise ValueError('Field "%s" not in table "%s", have fields %s' %
                         (n, self.table_name, ', '.join(t)))
    ki,vi = t.index(key_name), t.index(value_name)
    if foreach:
      fi = t.index(foreach)
      m = {}
      for f in set(v[fi] for v in self.values):
        m[f] = dict((v[ki],v[vi]) for v in self.values if v[fi] == f)
    else:
      m = dict((v[ki],v[vi]) for v in self.values)
    return m
