_GapChars = "-. "

class Blast_Output_Parser:
  """Parser for XML output from blastp (tested against version 2.2.29+)."""

  def __init__(self, name, xmlText):

    self.name = name

    # Bookkeeping data
    self.matches = []

    # Data from results
    self.database = None
    self.query = None
    self.queryLength = None
    self.reference = None
    self.version = None

    self.gapExistence = None
    self.gapExtension = None
    self.matrix = None

    self.dbSizeSequences = None
    self.dbSizeLetters = None

    # Extract information from results
    import xml.etree.ElementTree as ET
    tree = ET.fromstring(xmlText)
    if tree.tag != "BlastOutput":
      raise ValueError("Text is not BLAST XML output")
    self._extractRoot(tree)
    e = tree.find("./BlastOutput_param/Parameters")
    if e is not None:
      self._extractParams(e)
    el = tree.findall("BlastOutput_iterations/Iteration")
    if len(el) > 1:
      raise ValueError("Multi-iteration BLAST output unsupported")
    elif len(el) == 0:
      raise ValueError("No iteration data in BLAST OUTPUT")
    iteration = el[0]
    for he in iteration.findall("./Iteration_hits/Hit"):
      self._extractHit(he)
    self._extractStats(iteration.find("./Iteration_stat/Statistics"))

    self.matches.sort(key = lambda ma: ma.seq_match.evalue)
#    self.matches.sort(key = lambda ma: ma.seq_match.score, reverse = True)

  def _text(self, parent, tag):
    e = parent.find(tag)
    return e is not None and e.text.strip() or None

  def _extractRoot(self, oe):
    self.database = self._text(oe, "BlastOutput_db")
    self.query = self._text(oe, "BlastOutput_query-ID")
    self.queryLength = int(self._text(oe, "BlastOutput_query-len"))
    self.reference = self._text(oe, "BlastOutput_reference")
    self.version = self._text(oe, "BlastOutput_version")

  def _extractParams(self, pe):
    self.gapExistence = self._text(pe, "Parameters_gap-open")
    self.gapExtension = self._text(pe, "Parameters_gap-extend")
    self.matrix = self._text(pe, "Parameters_matrix")

  def _extractStats(self, se):
    self.dbSizeSequences = self._text(se, "Statistics_db-num")
    self.dbSizeLetters = self._text(se, "Statistics_db-len")

  def _extractHit(self, he):
    hid = self._text(he, "Hit_id")
    hdef = self._text(he, "Hit_def")
#    desc = '>%s %s' % (hid, hdef)
    desc = hdef
    for hspe in he.findall("./Hit_hsps/Hsp"):
      self._extractHSP(hspe, desc)

  def _extractHSP(self, hspe, desc):
    score = int(float(self._text(hspe, "Hsp_bit-score"))) #SH
    evalue = float(self._text(hspe, "Hsp_evalue"))
    qSeq = self._text(hspe, "Hsp_qseq")
    qStart = int(self._text(hspe, "Hsp_query-from"))
    qEnd = int(self._text(hspe, "Hsp_query-to"))
    hSeq = self._text(hspe, "Hsp_hseq")
    hStart = int(self._text(hspe, "Hsp_hit-from"))
    hEnd = int(self._text(hspe, "Hsp_hit-to"))
    sm = Sequence_Match(score, evalue, qStart, qEnd, qSeq, self.queryLength, hStart, hEnd, hSeq)
    matches = [Match(pdb_id, chain_id, hdesc, sm) for pdb_id, chain_id, hdesc in parse_blast_hit_chains(desc)]
    self.matches.extend(matches)

def parse_blast_hit_chains(blast_desc):
  '''Parse blast hit description and make  a list of 3-tuples,
  containing pdb id, list of chain ids, and pdb description.'''
  pcd = []
  for pc in blast_desc.split('|'):
    pdb_id, chains, desc = pc.split(maxsplit = 2)
    cids = chains.split(',')
    pcd.extend([(pdb_id,cid,desc) for cid in cids])
  return pcd

class Sequence_Match:
  """Sequence alignment from a single BLAST hit."""

  def __init__(self, score, evalue, qStart, qEnd, qSeq, qLen, hStart, hEnd, hSeq):
    self.score = score
    self.evalue = evalue
    self.qStart = qStart - 1        # switch to 0-base indexing
    self.qEnd = qEnd - 1
    self.qSeq = qSeq
    self.qLen = qLen                # Full query sequence length
    self.hStart = hStart - 1        # switch to 0-base indexing
    self.hEnd = hEnd - 1
    self.hSeq = hSeq
    if len(qSeq) != len(hSeq):
      raise ValueError("sequence alignment length mismatch")

  def residue_number_pairing(self):
    '''
    Returns two arrays of matching hit and query residue numbers.
    Sequence position 1 is residue number 1.
    '''
    if hasattr(self, 'rnum_pairs'):
      return self.rnum_pairs
    hs, qs = self.hSeq, self.qSeq
    h, q = self.hStart+1, self.qStart+1
    n = min(len(hs), len(qs))
    from numpy import empty, int32
    hp,qp = empty((n,), int32), empty((n,), int32)
    p = 0
    eqrnum = []
    for i in range(n):
      hstep = 0 if hs[i] in _GapChars else 1
      qstep = 0 if qs[i] in _GapChars else 1
      if hstep and qstep:
        hp[p] = h
        qp[p] = q
        p += 1
        if hs[i] == qs[i]:
          eqrnum.append(h)
      h += hstep
      q += qstep
    from numpy import resize, array, int32
    self.rnum_pairs = hqp = resize(hp, (p,)), resize(qp, (p,))
    self.rnum_equal = array(eqrnum, int32)
    return hqp

  def identical_residue_numbers(self):
    if hasattr(self, 'rnum_equal'):
      return self.rnum_equal
    residue_number_pairing()
    return self.rnum_equal

  def paired_residues_count(self):
    hrnum, qrnum = self.residue_number_pairing()
    return len(hrnum)

  def identical_residue_count(self):
    if not hasattr(self, 'nequal'):
      from numpy import frombuffer, byte
      self.nequal = sum(frombuffer(self.hSeq.encode('utf-8'), byte) == frombuffer(self.qSeq.encode('utf-8'), byte))
    return self.nequal

  def identity(self):
    return self.identical_residue_count() / self.paired_residues_count()

  def coverage(self):
    return self.paired_residues_count() / self.qLen

class Match:
  """Single PDB chain match to a query sequence from BLAST."""

  extra_chains = {}

  def __init__(self, pdb_id, chain_id, desc, seq_match):
    self.pdb_id = pdb_id
    self.chain_id = chain_id
    self.description = desc
    self.seq_match = seq_match
    self.mol = None
    self.rmsd = None

  def name(self):
    return '%s %s' % (self.pdb_id, self.chain_id)

  def load_structure(self, session):
    m = self.mol
    if m:
      return m

    # See if chain of already opened molecule can be used.
    k = (self.pdb_id, self.chain_id)
    ec = self.extra_chains
    if k in ec:
      m = self.mol = ec[k]
      del ec[k]
      return m
      
    from . import mmcif
    from .opensave import open_data
    m = open_data(self.pdb_id, session, from_database = 'PDBmmCIF', history = False)[0]
    self.mol = m
    for cid in m.chain_identifiers():
      cid = cid.decode('utf-8')
      if cid != self.chain_id:
        ec[(self.pdb_id, cid)] = m
    return m

def check_hit_sequences_match_mmcif_sequences(matches):

  for ma in matches:

    m,cid = ma.mol, ma.chain_id

    # Compute gapless hit sequence from match
    sm = ma.seq_match
    hseq = sm.hSeq
    for c in _GapChars:
      hseq = hseq.replace(c,'')
    hseq = '.'*sm.hStart + hseq

    # Using mmcif files the residue number (label_seq_id) is the index into the sequence.
    # This is not true of PDB files.
    from ..molecule.molecule import chain_sequence
    cseq = chain_sequence(m, cid)
    # Check that hit sequence matches PDB sequence
    if not sequences_match(hseq,cseq):
      print ('%s %s\n%s\n%s' % (m.name, cid, cseq, hseq))

# Find groups of overlapping structures and choose pairwise alignments to align structures within each group.
# A structure is aligned to the best scoring sequence hit it overlaps.
def connected_hit_groups(matches):
  if len(matches) == 0:
    return []
  qlen = matches[0].seq_match.qLen
  from numpy import empty, int32, minimum, in1d, unique
  min_m = empty((qlen+1,), int32)
  min_c = empty((qlen+1,), int32)       # Min covering hit.
  n = len(matches)
  min_m[:] = n
  min_c[:] = n
  joins = []
  mshort = []
  for mi,ma in enumerate(matches):
    sm = ma.seq_match
    rnum, qrnum = sm.residue_number_pairing()
    cid = ma.chain_id
    m = ma.mol
    rmask = m.residue_number_mask(cid, rnum.max())
    # Restrict to rnum with CA atoms.
    p = rmask[rnum].nonzero()[0]
#    if len(p) == 0:
    if len(p) < 3:
      # Blast match does not have enough CA atom coordinates for a unique alignment.
      mshort.append((ma, len(p)))
      continue
    qrnum_ca = qrnum[p]
    olap = min_m[qrnum_ca]
    minimum(olap, mi, olap)
    clap = minimum(min_c[qrnum_ca], mi)
    # Loop over disconnected overlapped subsets.
    joined = [mi]
    for j in unique(olap):
      if j < mi:
        clapj = clap[olap == j]
# Align longest overlap.  Really this is getting the highest overlap counting only
# residues that are the best scoring match at their residue position.  That is weird and hard to explain.
#        nol, jalign = max(((clapj == ja).sum(), ja) for ja in unique(clapj))
        # Align highest score chain of island j that overlaps mi
        jalign = clapj.min()
        nol = (clapj == jalign).sum()
        if nol < 3:
          # If highest score overlap is less than 3 residues choose the highest
          # that has at least 3 residues overlap.
          for ja in unique(clapj):
            nol = (clapj == ja).sum()
            if nol >= 3:
              jalign = ja
              break
        if nol < 3:
          # No overlap has at least 3 residues.
          maj = matches[j]
          mj,cj = maj.mol, maj.chain_id
          smj = maj.seq_match
          print('Alignment of %s %s %d-%d with best match %s %s %d-%d has only %d CA overlaps.' %
                (m.name, cid, sm.qStart+1, sm.qEnd+1, mj.name, cj, smj.qStart+1, smj.qEnd+1, nol))
#           print('Overlap sizes ', ','.join(('%d=%d' % (ja,(clapj == ja).sum()) for ja in unique(clapj))))
        else:
          joins.append((mi, jalign))
          joined.append(j)
    min_c[qrnum_ca] = clap
    min_m[qrnum_ca] = olap
    min_m[in1d(min_m,joined)] = min(joined)

  if mshort:
    print('%d matches had less than 3 matched CA atoms, preventing a unique alignment\n' % len(mshort) + 
          ', '.join('%d-%d %s %s %d' % (ma.seq_match.qStart+1, ma.seq_match.qEnd+1,
                                        ma.pdb_id, ma.chain_id, nca) for ma,nca in mshort))

  um = unique(min_m)[:-1]       # Remove n, always present at index 0.
  unhit = (min_m == n).sum()-1
  gra = {}
  for s,e in runs(min_m[1:]):
    if min_m[s+1] != n:
      gra.setdefault(min_m[s+1],[]).append('%d-%d' % (s+1,e+1))
  granges = [('(%s)' % ','.join(ra)) for ra in gra.values()]
  print('%d alignment groups %s' % (len(um), ', '.join(granges)))
#  print('%d joins' % len(joins))
  show_covering_ribbons(min_c, matches)
  ctrees = graph_trees(joins, matches)
  return ctrees

def graph_trees(joins, matches):
  t = {}
  for j1, j2 in joins:
    t.setdefault(j1,[]).append(j2)
    t.setdefault(j2,[]).append(j1)

  trees = []
  while t:
    root = min(t.keys())
    trees.append(dict_tree(root, t, matches))

  return trees

def dict_tree(root, d, replace):
  children = d.pop(root)
  return (replace[root], tuple(dict_tree(c, d, replace) for c in children if c in d))

def show_covering_ribbons(min_c, matches):
  mcset = set()
  rclist = []
  from numpy import unique
  for mi in unique(min_c):
    if mi >= len(matches):
      continue
    ma = matches[mi]
    m,cid = ma.mol, ma.chain_id
    rnum, qrnum = ma.seq_match.residue_number_pairing()
    r = m.atom_subset('CA', cid, residue_numbers = rnum[(min_c == mi)[qrnum]])
    rclist.append((m,cid,r))
    mcset.add(m)
  mols = set(ma.mol for ma in matches)
  for m in mols:
    m.display = (m in mcset)
    m.set_ribbon_display(False)
    m.atoms().hide_atoms()
  from random import randint as rint
  for m,cid,r in rclist:
    r.show_ribbon()
#    cr = m.atom_subset(chain_id = cid)
#    cr.show_ribbon()
#    cr.color_ribbon((128,128,128,255))
#    print('Covering %s, chain %s, %d residues' % (m.name, cid, r.count()))
    color = (rint(100,255),rint(100,255),rint(100,255),255)
    r.color_ribbon(color)

  # Show best e-value coverage per residue.
  csegs = []
  g = gr = 0
  segs = runs(min_c[1:])
  for s,e in segs:
    mi = min_c[s+1]
    ra = '%d-%d' % (s+1, e+1)
    if mi < len(matches):
      ma = matches[mi]
      m,c = ma.mol, ma.chain_id
      ev = ma.seq_match.evalue
      from os import path
      mname = path.splitext(m.name)[0]
      mc = '#%d.%s' % (m.id, c)
      cseg = '%10s %5d %6s %8s %8.0e' % (ra, e-s+1, mname, mc, ev)
    else:
      cseg = '%10s %5d %6s' % (ra, e-s+1, 'gap')
      g += 1
      gr += e-s+1
    csegs.append(cseg)
  from numpy import unique
  nc = len(unique(min_c))-1
  print('Best E-value coverage, %d segments with %d gaps (%d of %d residues), using %d chains'
        % (len(segs)-g, g, gr, len(min_c)-1, nc))
  print('\n'.join(csegs))

#  print('min_c', ' '.join('%d' % i for i in min_c))

def runs(a):
  r = []
  s = e = 0
  while e < len(a):
    if a[e] != a[s]:
      r.append((s,e-1))
      s = e
    e += 1
  if s < len(a):
    r.append((s,e-1))
  return r

def align_connected(ctrees):
  for ctree in ctrees:
    ma_ref, children = ctree
    if children:
      ma_align = [ma for ma,cc in children]
      align_matches_to_match(ma_align, ma_ref)
      align_connected(children)

def align_matches_to_chain(matches, qmol, qchain):
  qrmask = qmol.residue_number_mask(qchain, len(qmol.sequence))
  qtoref = None
  for ma in matches:
    align_match(ma, qmol, qchain, qrmask, qtoref)

def align_matches_to_match(matches, ref_match):
  rsm = ref_match.seq_match
  hrnum, qrnum = rsm.residue_number_pairing()
  qtoh = integer_array_map(qrnum, hrnum, rsm.qLen+1)
  m,c = ref_match.mol, ref_match.chain_id
  hrmask = m.residue_number_mask(c, qtoh.max())
  for match in matches:
    align_match(match, m, c, hrmask, qtoh)

def align_match(match, ref_mol, ref_chain, ref_rmask, qtoref):
  rnum, qrnum = match.seq_match.residue_number_pairing()
  ref_rnum = qrnum if qtoref is None else qtoref[qrnum]
  m, cid = match.mol, match.chain_id
  if m is ref_mol and cid == ref_chain:
    rmsd = 0
  else:
    rmsd = align_chain(m, cid, rnum, ref_mol, ref_chain, ref_rnum, ref_rmask)
  if not rmsd is None:
    match.rmsd = rmsd

def align_chain(mol, chain, rnum, ref_mol, ref_chain, ref_rnum, ref_rmask):
  # Restrict paired residues to those with CA atoms.
  rmask = mol.residue_number_mask(chain, rnum.max())
  from numpy import logical_and
  p = logical_and(rmask[rnum],ref_rmask[ref_rnum]).nonzero()[0]
  atoms = mol.atom_subset('CA', chain, residue_numbers = rnum[p])
  ref_atoms = ref_mol.atom_subset('CA', ref_chain, residue_numbers = ref_rnum[p])
  if atoms.count() == 0:
    return None

  # Compute RMSD of aligned hit and query.
  from ..molecule import align
  tf, rmsd = align.align_points(atoms.coordinates(), ref_atoms.coordinates())

  # Align hit chain to query chain
  mol.atom_subset(chain_id = chain).move_atoms(tf)

#  if atoms.count() < 5:
#    ma = mol.blast_match
#    ref_ma = ref_mol.blast_match
#    print('aligned %d residues between %d %s %s %d-%d (max %d) %d (%d CA) and %d %s %s %d-%d (%d CA)' %
#          (atoms.count(), mol.id, mol.name, chain, ma.qStart+1, ma.qEnd+1, rnum.max(), rmask.sum(), rmask[rnum].sum(),
#           ref_mol.id, ref_mol.name, ref_chain, ref_ma.qStart+1, ref_ma.qEnd+1, ref_rmask[ref_rnum].sum()))
#    print(ma.hSeq)
#    print(ma.qSeq)
#    rn, qrn = ma.residue_number_pairing()

  return rmsd

def integer_array_map(key, value, max_key):
  from numpy import zeros
  m = zeros((max_key+1,), value.dtype)
  m[key] = value
  return m

def match_metrics_table(matches):
  lines = [' PDB Chain  RMSD  Coverage(#,%) Identity(#,%) E-value Description']
  sms = set()
  for ma in matches:
    sm = ma.seq_match
    nqres = sm.qLen
    npair = sm.paired_residues_count()
    neq = sm.identical_residue_count()
    # Create table output line showing how well hit matches query.
    m, cid = ma.mol, ma.chain_id
    name = m.name[:-4] if m.name.endswith('.cif') else m.name
    desc = ma.description if not sm in sms else ''
    sms.add(sm)
    rmsd = ('%7.2f' % ma.rmsd) if not ma.rmsd is None else '.'
    lines.append('%4s %3s %7s %5d %5.0f   %5d %5.0f  %8.0e  %s'
                 % (name, cid, rmsd, npair, 100*npair/nqres,
                    neq, 100.0*neq/npair, sm.evalue, desc))

  return '\n'.join(lines)

def show_matches_as_ribbons(matches, ref_mol, ref_chain,
                            rescolor = (225,150,150,255), eqcolor = (225,100,100,255),
                            unaligned_rescolor = (225,225,150,255), unaligned_eqcolor = (225,225,100,255)):
  mset = set()
  for ma in matches:
    m = ma.mol
    m.single_color()
    aligned = getattr(m,'blast_match_rmsds', {})
    cid = ma.chain_id
    c1, c2 = (rescolor, eqcolor) if not ma.rmsd is None else (unaligned_rescolor, unaligned_eqcolor)
    sm = ma.seq_match
    hrnum, qrnum = sm.residue_number_pairing()
    r = m.atom_subset(chain_id = cid, residue_numbers = hrnum)
    r.color_ribbon(c1)
    req = m.atom_subset(chain_id = cid, residue_numbers = sm.identical_residue_numbers())
    req.color_ribbon(c2)
    if not m in mset:
      hide_atoms_and_ribbons(m)
      m.set_ribbon_radius(0.25)
      mset.add(m)
    m.atom_subset(chain_id = cid).show_ribbon()
  if ref_mol:
    ref_mol.set_ribbon_radius(0.25)
    hide_atoms_and_ribbons(ref_mol)
    ref_mol.atom_subset(chain_id = ref_chain).show_ribbon()

def hide_atoms_and_ribbons(m):
    atoms = m.atoms()
    atoms.hide_atoms()
    atoms.hide_ribbon()

def color_by_coverage(matches, mol, chain,
                      c0 = (200,200,200,255), c100 = (0,255,0,255)):
  rmax = max(ma.qEnd for ma in matches) + 1     # qEnd uses zero-base indexing, need 1-base
  from numpy import zeros, float32, outer, uint8
  qrc = zeros((rmax+1,), float32)
  for ma in matches:
    hrnum, qrnum = ma.seq_match.residue_number_pairing()
    qrc[qrnum] += 1

  qrc /= len(matches)
  rcolors = (outer((1-qrc),c0) + outer(qrc,c100)).astype(uint8)
  mol.color_ribbon(chain, rcolors)
  return qrc[1:].min(), qrc[1:].max()

def blast_color_by_coverage(session):
  if not hasattr(session, 'blast_results'):
    return 
  mol, chain, results = session.blast_results
  for ma in results.matches:
    ma.mol.display = False
  cmin, cmax = color_by_coverage(results.matches, mol, chain)
  session.show_status('Residues colored by number of sequence hits (%.0f-%.0f%%)' % (100*cmin, 100*cmax))

def show_only_matched_residues(matches):
  mset = set()
  for ma in matches:
    hrnum, qrnum = ma.seq_match.residue_number_pairing()
    m = ma.mol
    if not m in mset:
      m.display = True
      hide_atoms_and_ribbons(m)
    m.atom_subset(chain_id = ma.chain_id, residue_numbers = hrnum).show_ribbon()

def blast_show_matched_residues(session):
  if not hasattr(session, 'blast_results'):
    return 
  mol, chain, results = session.blast_results
  show_only_matched_residues(results.matches)

def sequences_match(s, seq):
  n = min(len(s), len(seq))
  for i in range(n):
    if s[i] != seq[i] and s[i] != '.' and s[i] != 'X' and seq[i] != '.' and seq[i] != 'X':
      return False
  return True

def divide_string(s, prefix):
  ds = []
  i = 0
  while True:
    e = s.find(prefix,i+1)
    if e == -1:
      ds.append(s[i:])
      break
    ds.append(s[i:e])
    i = e
  return ds

def create_fasta_database(mmcif_dir):
  '''Create fasta file for blast database using mmcif divided database sequences.
  Make just one entry for identical sequences that come from multiple pdbs or chains.'''
  seq_ids = {}
  from os import path, listdir
  from . import mmcif
  for dir2 in listdir(mmcif_dir):
    d2 = path.join(mmcif_dir,dir2)
    if path.isdir(d2):
      print(dir2)
      for ciffile in listdir(d2):
        if ciffile.endswith('.cif'):
          cifpath = path.join(d2, ciffile)
          cseq = mmcif.mmcif_sequences(cifpath)
          pdb_id = ciffile[:4]
          for cid, (start,seq,desc) in cseq.items():
            pt = seq_ids.setdefault(seq,{})
            if pdb_id in pt:
              pt[pdb_id][0].append(cid)
            else:
              pt[pdb_id] = ([cid],desc)
  lines = []
  for seq, ids in seq_ids.items():
    lines.append('>%s' % '|'.join('%s %s %s' % (id, ','.join(sorted(cids)), desc) for id,(cids,desc) in ids.items()))
    lines.append(seq)
  fa = '\n'.join(lines)
  return fa

def write_temporary_fasta(seq_name, seq, prefix):
  # Write FASTA sequence file
  import tempfile
  fasta_file = tempfile.NamedTemporaryFile('w', suffix = '.fasta', prefix = prefix, delete = False)
  write_fasta(seq_name, seq, fasta_file)
  fasta_file.close()
  fasta_path = fasta_file.name
  return fasta_path

def temporary_file_path(prefix = None, suffix = None):
  import tempfile
  f = tempfile.NamedTemporaryFile('w', prefix = prefix, suffix = suffix, delete = True)
  path = f.name
  f.close()
  return path

def write_fasta(name, seq, file):
  file.write('>%s\n%s\n' % (name, seq))

def fasta_sequence(path):
  f = open(path, 'r')
  lines = [line.strip() for line in f.readlines() if not line[0] in ('>', '#')]
  f.close()
  return ''.join(lines)

def run_blastp(name, fasta_path, output_path, blast_program, blast_database):
  # ../bin/blastp -db pdbaa -query 2v5z.fasta -outfmt 5 -out test.xml
  from os.path import dirname, basename
  dbdir, dbname = dirname(blast_database), basename(blast_database)
  cmd = ('env BLASTDB=%s %s -db %s -query %s -outfmt 5 -out %s' %
         (dbdir, blast_program, dbname, fasta_path, output_path))
  print (cmd)
  import os
  os.system(cmd)
  f = open(output_path)
  xml_text = f.read()
  f.close()
  results = Blast_Output_Parser(name, xml_text)
  return results

def blast_command(cmdname, args, session):

  from ..ui.commands import chain_arg, string_arg, parse_arguments
  req_args = ()
  opt_args = (('chain', chain_arg),)
  kw_args = (('sequence', string_arg),
             ('uniprot', string_arg),
             ('blastProgram', string_arg),
             ('blastDatabase', string_arg),)

  kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
  kw['session'] = session
  blast(**kw)

def blast(chain = None, session = None, sequence = None, uniprot = None,
          blastProgram = '/usr/local/ncbi/blast/bin/blastp',
          blastDatabase = '/usr/local/ncbi/blast/db/mmcif'):

  from os import path
  if not chain is None:
    molecule, chain_id = chain
    cid = chain_id.decode('utf-8')
    if not molecule.path.endswith('.cif'):
      from ..ui.commands import CommandError
      raise CommandError('blast: Can only handle sequences from mmCIF files.')
    from . import mmcif
    s = mmcif.mmcif_sequences(molecule.path)
    start,seq,desc = s[cid]
    molecule.sequence = seq
    from os import path
    mname = path.splitext(molecule.name)[0]
    seq_name = '%s chain %s' % (mname, cid)
    fasta_prefix =  '%s_%s_' % (mname, cid)
    fasta_path = write_temporary_fasta(seq_name, seq, fasta_prefix)
    blast_output = path.splitext(fasta_path)[0] + '.xml'
  elif not sequence is None:
    seq = sequence
    seq_name = '%s...%s' % (seq[:4], seq[-4:]) if len(seq) > 8 else seq
    fasta_path = write_temporary_fasta(seq_name, seq, prefix = seq[:6])
    blast_output = path.splitext(fasta_path)[0] + '.xml'
    molecule = chain_id = None
  elif not uniprot is None:
    from .fetch_uniprot import fetch_uniprot
    seq_name = 'UniProt %s' % uniprot
    fasta_path = fetch_uniprot(uniprot, session)
    seq = fasta_sequence(fasta_path)
    blast_output = temporary_file_path(prefix = uniprot, suffix = '.xml')
    molecule = chain_id = None
  else:
    from ..ui.commands import CommandError
    raise CommandError('blast: Must specify an open chain or sequence or uniprot id argument')

  # Run blastp standalone program and parse results
  session.show_status('Blast %s, running...' % (seq_name,))
  results = run_blastp(seq_name, fasta_path, blast_output, blastProgram, blastDatabase)
  matches = results.matches

  # Report number of matches
  np = len(set(ma.pdb_id for ma in matches))
  nc = len(set((ma.pdb_id,ma.chain_id) for ma in matches))
  msg = ('%s, sequence length %d\nsequence %s\n%d sequence matches, %d PDBs, %d chains'
         % (seq_name, len(seq), seq, len(matches), np, nc))
  session.show_info(msg)

  # Load matching structures
  session.show_status('Blast %s, loading %d sequence hits' % (seq_name, len(matches)))
  for ma in matches:
    ma.load_structure(session)

  # Report match metrics, align hit structures and show ribbons
  session.show_status('Blast %s, computing RMSDs...' % (seq_name,))
  # if molecule is None:
  #   # Align to best scoring hit.
  #   align_matches_to_match(matches, matches[0])
  # else:
  #   align_matches_to_chain(matches, molecule, chain_id)

  ctrees = connected_hit_groups(matches)
  align_connected(ctrees)

  text_table = match_metrics_table(matches)
  session.show_info(text_table)
  session.show_status('Blast %s, show hits as ribbons...' % (seq_name,))
#  show_matches_as_ribbons(matches, molecule, chain_id)
  session.show_status('Blast %s, done...' % (seq_name,))

  # Preserve most recent blast results for use by keyboard shortcuts
  session.blast_results = (molecule, chain_id, results)

def cycle_blast_molecule_display(session):
  cycler(session).toggle_play()
def next_blast_molecule_display(session):
  cycler(session).show_next()
def previous_blast_molecule_display(session):
  cycler(session).show_previous()
def all_blast_molecule_display(session):
  cycler(session).show_all()

def cycler(session):
  if not hasattr(session, 'blast_cycler'):
    session.blast_cycler = Blast_Display_Cycler(session)
  return session.blast_cycler

class Blast_Display_Cycler:
  def __init__(self, session):
    self.frame = None
    self.frames_per_molecule = 10
    self.session = session
    self.last_match = None
    self.match_num = 0
  def toggle_play(self):
    if self.frame is None:
      self.frame = 0
      self.show_none()
      v = self.session.view
      v.add_new_frame_callback(self.next_frame)
    else:
      self.stop_play()
      self.show_all()
  def stop_play(self):
    if self.frame is None:
      return
    self.frame = None
    v = self.session.view
    v.remove_new_frame_callback(self.next_frame)
  def matches(self):
    return self.session.blast_results[2].matches
  def show_next(self):
    self.stop_play()
    self.show_hit((self.match_num + 1) % len(self.matches()))
  def show_previous(self):
    self.stop_play()
    nh = len(self.matches())
    self.show_hit((self.match_num + nh - 1) % nh)
  def show_all(self):
    self.stop_play()
    for ma in self.matches():
      m, cid = ma.mol, ma.chain_id
      m.display = True
      m.atom_subset(chain_id = cid).show_ribbon()
    self.last_match = None
  def show_none(self):
    for ma in self.matches():
      ma.mol.display = False
    self.last_match = None
  def show_hit(self, match_num):
    self.match_num = match_num
    ma = self.matches()[match_num]
    m,c = ma.mol, ma.chain_id
    lm = self.last_match
    if not ma is lm:
      if lm:
        lm.mol.display = False
      else:
        self.show_none()
      m.display = True
      self.last_match = ma
    m.atom_subset(chain_id = c).show_ribbon(only_these = True)
    s = self.session
    from os import path
    mname = path.splitext(m.name)[0]
    rmsd = '%.2f' % ma.rmsd if not ma.rmsd is None else '.'
    sm = ma.seq_match
    s.show_status('%s chain %s, %.0f%% identity, %.0f%% coverage, rmsd %s   %s' %
                  (mname, c, 100*sm.identity(), 100*sm.coverage(),
                   rmsd, ma.description))
  def next_frame(self):
    f = self.frame
    if f == 0:
      self.show_hit(self.match_num)
    if f+1 >= self.frames_per_molecule:
      self.frame = 0
      self.match_num = (self.match_num + 1) % len(self.matches())
    else:
      self.frame += 1

