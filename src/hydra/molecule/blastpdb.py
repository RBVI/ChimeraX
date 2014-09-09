_GapChars = "-. "

class Blast_Run:
  """Parser for XML output from blastp (tested against version 2.2.29+)."""

  def __init__(self, name, fasta_path, output_path, blast_program, blast_database):

    self.name = name

    # One Match object for each sequence hit.
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

    xml_text = self.run_blastp(name, fasta_path, output_path, blast_program, blast_database)
    self.parse_results(xml_text)

  def run_blastp(self, name, fasta_path, output_path, blast_program, blast_database):
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
    return xml_text

  def query_length(self):
    return self.queryLength

  def best_match_per_residue(self):
    if not hasattr(self, 'mbest'):
      self.mbest = best_match_per_residue(self.matches, self.query_length())
    return self.mbest

  def parse_results(self, xml_text):

    # Extract information from results
    import xml.etree.ElementTree as ET
    tree = ET.fromstring(xml_text)
    if tree.tag != "BlastOutput":
      raise ValueError("Text is not BLAST XML output")
    self.parse_Root(tree)
    e = tree.find("./BlastOutput_param/Parameters")
    if e is not None:
      self.parse_Params(e)
    el = tree.findall("BlastOutput_iterations/Iteration")
    if len(el) > 1:
      raise ValueError("Multi-iteration BLAST output unsupported")
    elif len(el) == 0:
      raise ValueError("No iteration data in BLAST OUTPUT")
    iteration = el[0]
    for he in iteration.findall("./Iteration_hits/Hit"):
      self.parse_Hit(he)
    self.parse_Stats(iteration.find("./Iteration_stat/Statistics"))

#    self.matches.sort(key = lambda ma: ma.seq_match.evalue)
    self.matches.sort(key = lambda ma: ma.seq_match.score, reverse = True)
#    self.matches.sort(key = lambda ma: ma.seq_match.qEnd - ma.seq_match.qStart, reverse = True)

  def parse_Root(self, oe):
    v = xml_tag_value
    self.database = v(oe, "BlastOutput_db")
    self.query = v(oe, "BlastOutput_query-ID")
    self.queryLength = v(oe, "BlastOutput_query-len", int)
    self.reference = v(oe, "BlastOutput_reference")
    self.version = v(oe, "BlastOutput_version")

  def parse_Params(self, pe):
    v = xml_tag_value
    self.gapExistence = v(pe, "Parameters_gap-open")
    self.gapExtension = v(pe, "Parameters_gap-extend")
    self.matrix = v(pe, "Parameters_matrix")

  def parse_Stats(self, se):
    v = xml_tag_value
    self.dbSizeSequences = v(se, "Statistics_db-num")
    self.dbSizeLetters = v(se, "Statistics_db-len")

  def parse_Hit(self, he):
    v = xml_tag_value
#    hid = v(he, "Hit_id")
    hdef = v(he, "Hit_def")
    for hspe in he.findall("./Hit_hsps/Hsp"):
      self.parse_HSP(hspe, hdef)

  def parse_HSP(self, hspe, hdef):
    v = xml_tag_value
    score = v(hspe, "Hsp_bit-score", float)
    evalue = v(hspe, "Hsp_evalue", float)
    qSeq = v(hspe, "Hsp_qseq")
    qStart = v(hspe, "Hsp_query-from", int)
    qEnd = v(hspe, "Hsp_query-to", int)
    hSeq = v(hspe, "Hsp_hseq")
    hStart = v(hspe, "Hsp_hit-from", int)
    hEnd = v(hspe, "Hsp_hit-to", int)
    sm = Sequence_Match(score, evalue, qStart, qEnd, qSeq, self.queryLength, hStart, hEnd, hSeq, hdef)
    matches = [Match(pdb_id, chain_id, hdesc, sm) for pdb_id, chain_id, hdesc in parse_blast_hit_chains(hdef)]
    self.matches.extend(matches)

def xml_tag_value(parent, tag, value_type = None):
  e = parent.find(tag)
  s = e is not None and e.text.strip() or None
  v = s if value_type is None else value_type(s)
  return v

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

  def __init__(self, score, evalue, qStart, qEnd, qSeq, qLen, hStart, hEnd, hSeq, hDef):
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
    self.hDef = hDef

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
    nins = nrins = ndel = nrdel = 0
    for i in range(n):
      hstep = 0 if hs[i] in _GapChars else 1
      qstep = 0 if qs[i] in _GapChars else 1
      if hstep and qstep:
        hp[p] = h
        qp[p] = q
        p += 1
        if hs[i] == qs[i]:
          eqrnum.append(h)
      elif hstep:
        nrins += 1
        if i == 0 or not qs[i-1] in _GapChars:
          nins += 1
      elif qstep:
        nrdel += 1
        if i == 0 or not hs[i-1] in _GapChars:
          ndel += 1
      h += hstep
      q += qstep
    from numpy import resize, array, int32
    self.rnum_pairs = hqp = resize(hp, (p,)), resize(qp, (p,))
    self.rnum_equal = array(eqrnum, int32)
    self.nins, self.nrins = nins, nrins
    self.ndel, self.nrdel = ndel, nrdel
    return hqp

  def insertion_count(self):
    return self.nins
  def insertion_residue_count(self):
    return self.nrins
  def deletion_count(self):
    return self.ndel
  def deletion_residue_count(self):
    return self.nrdel

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

  def __init__(self, pdb_id, chain_id, desc, seq_match):
    self.pdb_id = pdb_id
    self.chain_id = chain_id
    self.description = desc
    self.seq_match = seq_match
    self.mol = None
    self.pairing = None
    self.align = None

  def name(self):
    return '%s %s' % (self.pdb_id, self.chain_id)

  def load_structure(self, session, cache = None):
    m = self.mol
    if m:
      return m
    id = self.pdb_id
    if cache and id in cache:
      m = cache[id]
    else:
      from ..files import fetch
      m = fetch.fetch_from_database(id, 'PDBmmCIF', session)[0]
      if not cache is None:
        cache[id] = m
    self.mol = m.copy_chain(self.chain_id)
    return self.mol

  def residues_with_coords_pairing(self):
    if self.pairing is None:
      sm = self.seq_match
      rnum, qrnum = sm.residue_number_pairing()
      rmask = residue_number_mask(self.mol, rnum.max())      # Residues with coordinates
      p = rmask[rnum].nonzero()[0]
      self.pairing = (rnum[p], qrnum[p])
    return self.pairing

  def show_ribbon(self):
    self.mol.set_ribbon_display(True)
    self.mol.display = True

  def hide_ribbon(self):
    self.mol.set_ribbon_display(False)

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
    from .molecule import chain_sequence
    cseq = chain_sequence(m, cid)
    # Check that hit sequence matches PDB sequence
    if not sequences_match(hseq,cseq):
      print ('%s %s\n%s\n%s' % (m.name, cid, cseq, hseq))

def drop_similar_chains(br, max_rmsd, session, within_pdb = True):
  matches = br.matches
  rmsds = same_sequence_rmsds(matches, within_pdb)
  msg = ('Same sequence RMSDs, %s\n' % ('within PDB' if within_pdb else 'between PDBs') +
         'PDB Chain Length  PDB Chain Length Overlap   RMSD\n')
  msg += '\n'.join('%4s %2s %6d     %4s %2s %5d %7d %8.2f' % v for v in rmsds)
  session.show_info(msg)

  drop = set((pdb_id,cid) for pdb_id0, cid0, nr0, pdb_id, cid, nr, nol, rmsd in rmsds if rmsd < max_rmsd)
  mdrop = [ma for ma in matches if (ma.pdb_id, ma.chain_id) in drop]
  if mdrop:
    remove_matches(mdrop, br, session)
    session.show_info('Dropped %d similar chains (%d matches) within %.2f rmsd' %
                      (len(drop), len(mdrop), max_rmsd))

def remove_matches(mdrop, br, session):
  if len(mdrop) == 0:
    return
  br.matches = mkeep = [ma for ma in br.matches if not ma in mdrop]
  mclose = set(ma.mol for ma in mdrop)
  mclose -= set(ma.mol for ma in mkeep)       # Multiple matches use the same molecule
  session.close_models(mclose)

def same_sequence_rmsds(matches, within_pdb = True):
  if within_pdb:
    smpairs = tuple(((ma.seq_match.hDef,ma.pdb_id), ma) for ma in matches)
  else:
    smpairs = tuple((ma.seq_match.hDef, ma) for ma in matches)
  seq_matches = {}
  for seq_id, ma in smpairs:
    seq_matches.setdefault(seq_id,[]).append(ma)
  rmsds = []
  from numpy import in1d
  for seq_id, malist in seq_matches.items():
    umalist = list(dict(((ma.pdb_id, ma.chain_id), ma) for ma in malist).values())
    if len(umalist) > 1:
      umalist.sort(key = lambda ma: (ma.pdb_id, ma.chain_id))
      ma0 = umalist[0]
      ca0 = ma0.mol.atom_subset('CA')
      rnum0 = ca0.residue_numbers()
      xyz0 = ca0.coordinates()
      for ma in umalist[1:]:
        ca = ma.mol.atom_subset('CA')
        rnum = ca.residue_numbers()
        xyz = ca.coordinates()[in1d(rnum,rnum0)]
        from . import align
        pxyz0 = xyz0[in1d(rnum0,rnum)]
        if len(xyz) != len(pxyz0):
          print('ssr', len(xyz), ma.pdb_id, ma.chain_id, ca.count(), len(pxyz0), ma0.pdb_id, ma0.chain_id, ca0.count())
          print(list(rnum))
          print(list(rnum0))
        tf, rmsd = align.align_points(xyz, pxyz0)
        rmsds.append((ma0.pdb_id, ma0.chain_id, ca0.count(), ma.pdb_id, ma.chain_id, ca.count(), len(xyz), rmsd))

  return rmsds

def best_match_per_residue(matches, qlen):
  # Matches must be ordered high score to low.
  from numpy import empty, int32, minimum
  mbest = empty((qlen+1,), int32)
  mbest[:] = len(matches)
  for mi,ma in enumerate(matches):
    rnum, qrnum = ma.residues_with_coords_pairing()
    mbest[qrnum] = minimum(mbest[qrnum], mi)
  return mbest

# Blast matches with less than 3 CA atom coordinates don't have a unique alignment.
def short_matches(matches, min_res):
  mshort = []
  for ma in matches:
    rnum, qrnum = ma.residues_with_coords_pairing()
    if len(rnum) < min_res:
      mshort.append((ma, len(rnum)))
  return mshort

# Align only short segments from left to right using 3 overlapping residues.
# This produced bad structures caused by sharp turns that resulted in steric clashes.
# The sharp turns resulted from filling small gaps -- just a 1 residue gap can
# completely change the direction the backbone goes in.
def mosaic_model(br):
  qlen = br.query_length()
  from numpy import empty, float32, zeros, int32
  xyz = empty((qlen+1,3), float32)
  found = zeros((qlen+1,), int32)
  mai = zeros((qlen+1,), int32)
  mbest = br.best_match_per_residue()
  matches = br.matches
  n = len(matches)

  # Find leftmost match
  m0 = mbest[(mbest < n).nonzero()[0][0]]
  ma = matches[m0]

  nar = 3
#  nar = 10

  # Start with 3 leftmost coordinates
  rnum, qrnum = ma.residues_with_coords_pairing()
  qi3 = qrnum[:nar]
  found[qi3] = 1
  xyz[qi3,:] = ma.mol.atom_subset('CA', residue_numbers = rnum[:nar]).coordinates()

  # Extend coordinates left to right by matching preceding 3 residues.
  qistart = qi3.max() + 1
  for qi in range(qistart, qlen):
    mi = mbest[qi]
    if mi == n:
      continue          # No coverage
    # Find the match covering position qi, that has at least 3 residues to the left with coordinates
    # that can be aligned.  These 3 residues may not be contiguous, so choose the match that has the
    # leftmost residue as far to the right as possible in the query sequence.  To break ties also 
    # have the 3 residues of the match as close to the match residue corresponding to position qi.
    # Finally consider the highest scoring match, for instance, in the common case where the previous
    # 3 contiguous residues in both query and match are available for alignment.
    smat = []
    for mj in range(mi,n):
      ma = matches[mj]
      rnum, qrnum = ma.residues_with_coords_pairing()
      if qi in qrnum:
        f = found[qrnum[qrnum<qi]]
        fs = f.sum()
        if fs >= nar:
          p = f.nonzero()[0][-nar:]
          rnum3,qrnum3 = rnum[p],qrnum[p]
          ri = rnum[(qrnum == qi).nonzero()[0][0]]
          smat.append((max(qi-qrnum3[0],ri-rnum3[0]), mj))
#          break
# TODO: Also scan right to left to fill in gaps
#    if fs < nar:
#      continue          # No sequence matches 3 already found residues.
    if len(smat) == 0:
      continue
    mi = min(smat)[1]
    ma = matches[mi]
    rnum, qrnum = ma.residues_with_coords_pairing()
    f = found[qrnum[qrnum<qi]]

    p = f.nonzero()[0][-nar:]
    rnum3,qrnum3 = rnum[p],qrnum[p]        # Residue numbers of preceding 3.
    ri = rnum[(qrnum == qi).nonzero()[0][0]]
    if qrnum3[0] + nar < qi or rnum3[0] + nar < ri:
      print('gap', qi, qrnum3, ri, rnum3, ma.pdb_id, ma.chain_id)
    xyz3 = ma.mol.atom_subset('CA', residue_numbers = rnum3).coordinates()
    from . import align
    tf, rmsd = align.align_points(xyz3, xyz[qrnum3])
    xyzi = ma.mol.atom_subset('CA', residue_numbers = [ri]).coordinates()
    xyz[qi,:] = tf*xyzi
    found[qi] = 1
    mai[qi] = mi

  fi = found.nonzero()[0]
  m = create_ca_trace(br.name, xyz[fi], fi, random_colors(mai[fi]))
  print('created mosaic model %d residues' % len(fi))
  return m

def random_colors(color_index):
  # Color residue according to which match used to extend
  from numpy import random, uint8
  colors = random.randint(100,255,(color_index.max()+1,4)).astype(uint8)
  colors[:,3] = 255
  return colors[color_index,:]

def create_ca_trace(name, xyz, rnums, colors):

  from numpy import zeros
  from . import atom_dtype
  atoms = zeros((len(xyz),), atom_dtype)
  atoms['atom_name'] = b'CA'
  atoms['element_number'] = 6
  atoms['xyz'] = xyz
  atoms['radius'] = 1.7
  atoms['residue_name'] = b'ALA'
  atoms['residue_number'] = rnums
  atoms['chain_id'] = b'A'
  atoms['atom_color'] = (255,255,0,255)
  atoms['ribbon_color'] = colors
#  atoms['ribbon_color'] = (255,0,255,255)
#  atoms['ribbon_color'][::2] = (255,255,255,255)
  atoms['atom_shown'] = 0
  atoms['ribbon_shown'] = 1

  from . import Molecule
  m = Molecule(name, atoms)
  return m

# Find groups of overlapping structures and choose pairwise alignments to align structures within each group.
# A structure is aligned to the best scoring sequence hit it overlaps.
def structure_alignment_pairs(br):
  qlen = br.query_length()
  matches = br.matches
  n = len(matches)

  from numpy import empty, int32, minimum, in1d, unique
  rgroup = empty((qlen+1,), int32)
  rgroup[:] = n
  align_pairs = []

  mbest = br.best_match_per_residue()
  for mi,ma in enumerate(matches):
    rnum, qrnum = ma.residues_with_coords_pairing()
    if len(rnum) < 3:
      continue      # Blast match does not have enough CA atom coordinates for a unique alignment.

    qrgroup = rgroup[qrnum]
    qbest = mbest[qrnum]
    # Align this match to each overlapped group.
    joined = [mi]
    for g in unique(qrgroup):
      if g < mi:
        qbestg = qbest[qrgroup == g]
        galign = qbestg.min()           # Highest score chain of group g that overlaps match.
        nol = (qbestg == galign).sum()  # Number of overlapping residues.
        if nol < 3:
          # Highest score overlapping match has fewer than 3 residues overlapping.
          # Find another match with at least 3 residues overlapping.
          qmask = integer_array_map(qrnum, 1, qlen+1)
          for mj in range(galign+1, mi):
            rnumj, qrnumj = matches[mj].residues_with_coords_pairing()
            if len(qrnumj) > 0 and rgroup[qrnumj[0]] == g:
              nol = qmask[qrnumj].sum()
              if nol >= 3:
                galign = mj
                break
        if nol >= 3:
          align_pairs.append((ma, matches[galign]))
          joined.append(g)

    rgroup[qrnum] = minimum(rgroup[qrnum],mi)
    if len(joined) > 1:
      rgroup[in1d(rgroup,joined)] = min(joined)

  return align_pairs, rgroup

def report_alignment_groups(rgroup):
  from numpy import unique
  um = unique(rgroup)[:-1]       # Remove n, always present at index 0.
  n = rgroup[0]
  unhit = (rgroup == n).sum()-1
  gra = {}
  for s,e in runs(rgroup[1:]):
    if rgroup[s+1] != n:
      gra.setdefault(rgroup[s+1],[]).append((s+1,e+1))
  groups = list(gra.values())
  groups.sort(key = lambda g: g[0][0])
  granges = [('(%s)' % ','.join('%d-%d' % rint for rint in rints)) for rints in groups]
  print('%d alignment groups %s' % (len(um), ', '.join(granges)))

def show_covering_ribbons(mbest, matches, full = False):
  mcset = set()
  rclist = []
  from numpy import unique
  for mi in unique(mbest):
    if mi >= len(matches):
      continue
    ma = matches[mi]
    ma.covering = True
    rnum, qrnum = ma.seq_match.residue_number_pairing()
    rnums = rnum if full else rnum[(mbest == mi)[qrnum]]
    m = ma.mol
    r = m.atom_subset('CA', residue_numbers = rnums)
    rclist.append(r)
    mcset.add(m)

  mols = set(ma.mol for ma in matches)
  for m in mols:
    m.display = (m in mcset)
    m.set_ribbon_display(False)
    m.atoms().hide_atoms()

  for r in rclist:
    r.show_ribbon()

def covering_model(mbest, matches):
  from numpy import zeros, float32, bool, unique, sort
  xyz = zeros((len(mbest),3), float32)
  covered = zeros((len(mbest),), bool)
  mcov = sort(unique(mbest))
  for mi in mcov:
    if mi >= len(matches):
      continue
    ma = matches[mi]
    ma.covering = True
    rnum, qrnum = ma.residues_with_coords_pairing()
    bi = (mbest == mi)[qrnum]
    rnums,qrnums = rnum[bi],qrnum[bi]
    m = ma.mol
    r = m.atom_subset('CA', residue_numbers = rnum)
    if not (r.residue_numbers() == rnum).all():
      print('oops, mismatched residue numbers', m.name, ma.chain_id, rnum, r.residue_numbers())
      m,a = r.molatoms[0]
      print (list(a), list(m.residue_nums[a]))
      raise ValueError()
    rxyz = r.coordinates()
    print ('realigning molecule', m.id, 'nres', bi.sum(), mi)
    pal = align_segment(rxyz, rnum, qrnum, bi, covered, xyz)
    if not pal is None:
      rxyz = pal * rxyz
    xyz[qrnums,:] = rxyz[bi]
    covered[qrnums] = True

  qrnums = covered.nonzero()[0]
  m = create_ca_trace('covering', xyz[covered], qrnums, random_colors(mbest[covered]))
  return m

def align_segment(rxyz, rnum, qrnum, rmask, covered, xyz, tail = 3, max_rmsd = 5.0):
  nmask = rmask.copy()
  from numpy import logical_or, logical_and
  for t in range(1,tail+1):
    logical_or(nmask[:-t], rmask[t:], nmask[:-t])
    logical_or(nmask[t:], rmask[:-t], nmask[t:])
  logical_and(nmask, covered[qrnum], nmask)
  nm = nmask.sum()
#  if nm >= 3:
  if nm >= 1:
    mrxyz = rxyz[nmask,:]
    mxyz = xyz[qrnum[nmask],:]
    dxyz = mrxyz - mxyz
    from math import sqrt
    pre_rmsd = sqrt((dxyz*dxyz).sum()/len(dxyz))
    if pre_rmsd > max_rmsd:
      nr = rmask.sum()
      if nm < 2*tail and nr > 5:
        print('ascm overlap too short', nm, nr, len(rxyz), pre_rmsd)
        return None
      from . import align
      tf, rmsd = align.align_points(mrxyz, mxyz)
      print('ascm realigned', nm, nr, len(rxyz), pre_rmsd, rmsd)
      return tf
    else:
      print ('ascm not moving %d, rmsd small %.2f' % (len(rxyz), pre_rmsd))
  else:
    print ('ascm not moving %d, overlap %d' % (len(rxyz), nm))

  return None

def random_ribbon_colors(matches):
  from random import randint as rint
  for ma in matches:
    color = (rint(100,255),rint(100,255),rint(100,255),255)
    r = ma.mol.atom_subset('CA')
    r.color_ribbon(color)

def report_best_match_coverage(mbest, matches):
  # Show text table of best e-value coverage per residue.
  csegs = []
  g = gr = 0
  segs = runs(mbest[1:])
  for s,e in segs:
    mi = mbest[s+1]
    if mi < len(matches):
      ma = matches[mi]
      m,c = ma.mol, ma.chain_id
      ev = ma.seq_match.evalue
      from os import path
      mname = path.splitext(m.name)[0]
      al = ma.align
      if al is None:
        align = ''
      else:
        amol, rmsd, npair = al
        align = '%5s %5.2f %3d' % ('#%-d' % amol.id, rmsd, npair)
      cseg = '%5d %5d  %5d %5s %4s %1s %8.0e %15s' % (s+1, e+1, e-s+1, '#%-d' % m.id, mname, c, ev, align)
    else:
      cseg = '%5d %5d  %5d %10s' % (s+1, e+1, e-s+1, 'gap')
      g += 1
      gr += e-s+1
    csegs.append(cseg)
  from numpy import unique
  nc = len(unique(mbest))-1
  print('Best E-value coverage, %d segments with %d gaps (%d of %d residues), using %d chains\n'
        % (len(segs)-g, g, gr, len(mbest)-1, nc) +
        'Query range  Length  Id    PDB   E-value Align RMSD Npair\n' + 
        '\n'.join(csegs))

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

def align_connected(ma_pairs, ma_fixed):
  g = {}
  for ma1, ma2 in ma_pairs:
    g.setdefault(ma1,[]).append(ma2)
    g.setdefault(ma2,[]).append(ma1)

  parents = set(ma_fixed)
  while parents:
    ma = parents.pop()
    if ma in g:
      children = g.pop(ma)
      if children:
        align_matches_to_match(children, ma)
        parents.update(children)
        for cma in children:
          g[cma].remove(ma)         # Children don't point to parent

def align_matches_to_chain(matches, qmol):
  qrmask = residue_number_mask(qmol, len(qmol.sequence))
  qtoref = None
  for ma in matches:
    align_match(ma, qmol, qrmask, qtoref)

def align_matches_to_match(matches, ref_match):
  rsm = ref_match.seq_match
  hrnum, qrnum = rsm.residue_number_pairing()
  qtoh = integer_array_map(qrnum, hrnum, rsm.qLen+1)
  m = ref_match.mol
  hrmask = residue_number_mask(m, qtoh.max())
  for match in matches:
    align_match(match, m, hrmask, qtoh)

def align_match(match, ref_mol, ref_rmask, qtoref):
  rnum, qrnum = match.seq_match.residue_number_pairing()
  ref_rnum = qrnum if qtoref is None else qtoref[qrnum]
  m = match.mol
  rmsd, npairs = (0,0) if m is ref_mol else align_chain(m, rnum, ref_mol, ref_rnum, ref_rmask)
  if not rmsd is None:
    match.align = (ref_mol, rmsd, npairs)

def align_chain(mol, rnum, ref_mol, ref_rnum, ref_rmask):
  # Restrict paired residues to those with CA atoms.
  rmask = residue_number_mask(mol, rnum.max())
  from numpy import logical_and
  p = logical_and(rmask[rnum],ref_rmask[ref_rnum]).nonzero()[0]
  atoms = mol.atom_subset('CA', residue_numbers = rnum[p])
  ref_atoms = ref_mol.atom_subset('CA', residue_numbers = ref_rnum[p])
  if atoms.count() == 0:
    return None, None

  # Compute RMSD of aligned hit and query.
  from . import align
#  tf, rmsd = align.align_points(atoms.coordinates(), ref_atoms.coordinates())
  dmax = 5.0
  niter = 20
  axyz, raxyz = atoms.coordinates(), ref_atoms.coordinates()
  tf, rmsd, mask = align.align_and_prune(axyz, raxyz, dmax, niter)

#  print('ac', mol.name, ref_mol.name, mask.sum() if not mask is None else 0, rmsd)
  if tf is None:
    tf, rmsd = align.align_points(axyz, raxyz)
    npairs = len(axyz)
    print('alignment pruning failed', mol.name, mol.chain_identifiers()[0], mol.id,
          ref_mol.name, ref_mol.chain_identifiers()[0], ref_mol.id, rmsd)
  else:
    # Color the atoms used for alignment.
    atoms.subset(mask.nonzero()[0]).color_atoms((255,0,0,255))
    npairs = mask.sum()

  # Align hit chain to query chain
  mol.atom_subset().move_atoms(tf)

#  if atoms.count() < 5:
#    ma = mol.blast_match
#    ref_ma = ref_mol.blast_match
#    print('aligned %d residues between %d %s %s %d-%d (max %d) %d (%d CA) and %d %s %s %d-%d (%d CA)' %
#          (atoms.count(), mol.id, mol.name, chain, ma.qStart+1, ma.qEnd+1, rnum.max(), rmask.sum(), rmask[rnum].sum(),
#           ref_mol.id, ref_mol.name, ref_chain, ref_ma.qStart+1, ref_ma.qEnd+1, ref_rmask[ref_rnum].sum()))
#    print(ma.hSeq)
#    print(ma.qSeq)
#    rn, qrn = ma.residue_number_pairing()

  return rmsd, npairs

def residue_number_mask(mol, rnmax = None):
  rnums = mol.atom_subset('CA').residue_numbers()
  n = rnums.max() if rnmax is None else max(rnums.max(),rnmax)
  from numpy import zeros, bool
  rmask = zeros((n+1,), bool)
  rmask[rnums] = True
  return rmask

def integer_array_map(key, value, max_key):
  from numpy import zeros, int32
  m = zeros((max_key+1,), int32)
  m[key] = value
  return m

def find_first(e,a):
  return (a == e).nonzero()[0][0]

def match_metrics_table(matches):
  lines = ['   Id   PDB  Align RMSD Npair Coverage(#,%) Identity(#,%) NCoord MissCrd NIns NRIns NDel NRDel E-value Description']
  sms = set()
  for ma in matches:
    sm = ma.seq_match
    nqres = sm.qLen
    npair = sm.paired_residues_count()
    neq = sm.identical_residue_count()
    srnum, sqrnum = sm.residue_number_pairing()
    rnum,qrnum = ma.residues_with_coords_pairing()
    missing_coords = (find_first(rnum[-1],srnum) - find_first(rnum[0],srnum) + 1) - len(rnum)
    # Create table output line showing how well hit matches query.
    m, cid = ma.mol, ma.chain_id
    name = m.name[:-4] if m.name.endswith('.cif') else m.name
    desc = ma.description if not sm in sms else ''
    sms.add(sm)
    al = ma.align
    if al is None:
      align = ''
    else:
      amol, rmsd, napair = al
      align = '%5s %5.2f %3d' % ('#%-d' % amol.id, rmsd, napair)
    lines.append('%5s %4s %1s %15s %5d %5.0f   %5d %5.0f    %5d %5d %5d %5d %5d %5d %8.0e  %s'
                 % ('#%-d' % m.id, name, cid, align, npair, 100*npair/nqres,
                    neq, 100.0*neq/npair, len(rnum), missing_coords,
                    sm.insertion_count(), sm.insertion_residue_count(),
                    sm.deletion_count(), sm.deletion_residue_count(),
                    sm.evalue, desc))

  return '\n'.join(lines)

def show_matches_as_ribbons(matches, ref_mol, ref_chain,
                            rescolor = (225,150,150,255), eqcolor = (225,100,100,255),
                            unaligned_rescolor = (225,225,150,255), unaligned_eqcolor = (225,225,100,255)):
  mset = set()
  for ma in matches:
    m = ma.mol
    m.single_color()
    aligned = getattr(m,'blast_match_rmsds', {})
    c1, c2 = (rescolor, eqcolor) if not ma.align is None else (unaligned_rescolor, unaligned_eqcolor)
    sm = ma.seq_match
    hrnum, qrnum = sm.residue_number_pairing()
    r = m.atom_subset(residue_numbers = hrnum)
    r.color_ribbon(c1)
    req = m.atom_subset(residue_numbers = sm.identical_residue_numbers())
    req.color_ribbon(c2)
    if not m in mset:
      hide_atoms_and_ribbons(m)
      m.set_ribbon_radius(0.25)
      mset.add(m)
    m.set_ribbon_display(True)
  if ref_mol:
    ref_mol.set_ribbon_radius(0.25)
    hide_atoms_and_ribbons(ref_mol)
    ref_mol.set_ribbon_display(True)

def hide_atoms_and_ribbons(m):
    atoms = m.atoms()
    atoms.hide_atoms()
    atoms.hide_ribbon()

def color_by_coverage(br, mol, chain,
                      c0 = (200,200,200,255), c100 = (0,255,0,255)):
  rmax = br.query_length()
  from numpy import zeros, float32, outer, uint8
  qrc = zeros((rmax+1,), float32)
  for ma in br.matches:
    hrnum, qrnum = ma.seq_match.residue_number_pairing()
    qrc[qrnum] += 1

  qrc /= len(br.matches)
  rcolors = (outer((1-qrc),c0) + outer(qrc,c100)).astype(uint8)
  mol.color_ribbon(chain, rcolors)
  return qrc[1:].min(), qrc[1:].max()

def blast_color_by_coverage(session):
  if not hasattr(session, 'blast_results'):
    return 
  br = session.blast_results
  for ma in br.matches:
    ma.mol.display = False
  cmin, cmax = color_by_coverage(br, br.query_molecule, br.query_chain_id)
  session.show_status('Residues colored by number of sequence hits (%.0f-%.0f%%)' % (100*cmin, 100*cmax))

def show_only_matched_residues(matches):
  mset = set()
  for ma in matches:
    hrnum, qrnum = ma.seq_match.residue_number_pairing()
    m = ma.mol
    if not m in mset:
      m.display = True
      hide_atoms_and_ribbons(m)
    m.atom_subset('CA', residue_numbers = hrnum).show_ribbon()

def blast_show_matched_residues(session):
  if hasattr(session, 'blast_results'):
    br = session.blast_results
    show_only_matched_residues(br.matches)

def blast_show_coverage(session):
  if hasattr(session, 'blast_results'):
    br = session.blast_results
    show_covering_ribbons(br.best_match_per_residue(), br.matches)

def blast_show_coverage_full(session):
  if hasattr(session, 'blast_results'):
    br = session.blast_results
    show_covering_ribbons(br.best_match_per_residue(), br.matches, full=True)

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

def blast_command(cmdname, args, session):

  from ..commands.parse import chain_arg, string_arg, float_arg, int_arg, parse_arguments
  req_args = ()
  opt_args = (('chain', chain_arg),)
  kw_args = (('sequence', string_arg),
             ('uniprot', string_arg),
             ('dropSimilarChains', float_arg),
             ('dropShortChains', int_arg),
             ('blastProgram', string_arg),
             ('blastDatabase', string_arg),)

  kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
  kw['session'] = session
  blast(**kw)

def blast(chain = None, session = None, sequence = None, uniprot = None,
          dropSimilarChains = 2.0, dropShortChains = 3,
          blastProgram = '/usr/local/ncbi/blast/bin/blastp',
          blastDatabase = '/usr/local/ncbi/blast/db/mmcif'):

  from os import path
  if not chain is None:
    molecule, chain_id = chain
    cid = chain_id.decode('utf-8')
    if not molecule.path.endswith('.cif'):
      from ..commands.parse import CommandError
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
    seq_name = uniprot
    fasta_path = fetch_uniprot(uniprot, session)
    seq = fasta_sequence(fasta_path)
    blast_output = temporary_file_path(prefix = uniprot, suffix = '.xml')
    molecule = chain_id = None
  else:
    from ..commands.parse import CommandError
    raise CommandError('blast: Must specify an open chain or sequence or uniprot id argument')

  # Run blastp standalone program and parse results
  session.show_status('Blast %s, running...' % (seq_name,))
  br = Blast_Run(seq_name, fasta_path, blast_output, blastProgram, blastDatabase)
  br.query_molecule = molecule
  br.query_chain_id = chain_id
  session.blast_results = br       # Preserve most recent blast results for use by keyboard shortcuts
  matches = br.matches

  # Report number of matches
  n = len(matches)
  ns = len(set(ma.seq_match.hDef for ma in matches))
  np = len(set(ma.pdb_id for ma in matches))
  nc = len(set((ma.pdb_id,ma.chain_id) for ma in matches))
  smat = {}
  for ma in matches:
    smat.setdefault(ma.seq_match.hDef,[]).append(ma)
  pc = [(len(set(ma.pdb_id for ma in malist)), len(set((ma.pdb_id,ma.chain_id) for ma in malist)))
        for sm, malist in smat.items()]
  upc = list((p,c,pc.count((p,c))) for p,c in set(pc))
  upc.sort()
  upc.reverse()
  cps = ' '.join([(('%d/%d' % (p,c)) if rep == 1 else ('%d/%d*%d' % (p,c,rep))) for p,c,rep in upc])
  msg = ('%s, sequence length %d\n' % (seq_name, len(seq)) +
         'Query sequence %s\n' % seq +
         'Matches %d unique sequences, %d PDBs, %d chains, %d chain alignments\n' % (ns, np, nc, n) +
         'PDB/chains per sequence %s' % cps)
  session.show_info(msg)
  # TODO: Report RMSD statistics for all chains with same sequence.

  # Load matching structures
  session.show_status('Blast %s, loading %d sequence hits' % (seq_name, len(matches)))
  mcache = {}
  mlist = [ma.load_structure(session, mcache) for ma in matches]
  mcache = None
  session.add_models(mlist)
  mw = session.main_window
  mw.view.initial_camera_view()
  mw.show_graphics()

  # Report match metrics, align hit structures and show ribbons
  session.show_status('Blast %s, aligning structures...' % (seq_name,))

  # Report matches with fewer than 3 CA atoms, can't uniquely align.
  if dropShortChains > 0:
    min_res = dropShortChains
    mshort = short_matches(matches, min_res)
    if mshort:
      msg = ('%d matches had less than 3 matched CA atoms and have been dropped\n %s' %
             (len(mshort),
             ', '.join('%d-%d %s %s %d' % (ma.seq_match.qStart+1, ma.seq_match.qEnd+1,
                                           ma.pdb_id, ma.chain_id, nca) for ma,nca in mshort)))
      session.show_info(msg)
      mdrop = tuple(ma for ma,len in mshort)
      remove_matches(mdrop, br, session)
      matches = br.matches

  # Report RMSDs among identical sequence matches
  if dropSimilarChains > 0:
    max_rmsd = dropSimilarChains
    drop_similar_chains(br, max_rmsd, session, within_pdb = True)
    drop_similar_chains(br, max_rmsd, session, within_pdb = False)
    matches = br.matches

  # Record sequence alignment for molecules for using align command.
  for ma in matches:
    m = ma.mol
    arnums = m.atoms().residue_numbers()
    rnum, qrnum = ma.seq_match.residue_number_pairing()
    rmap = integer_array_map(rnum, qrnum, max(arnums.max(),rnum.max()))
    m.sequence_numbers = {seq_name: rmap[arnums]}

  # Report depth of coverage, hits per residue
  from numpy import zeros, int32
  qlen = br.query_length()
  qcount = zeros((qlen+1,), int32)
  for ma in matches:
    rnum, qrnum = ma.residues_with_coords_pairing()
    qcount[qrnum] += 1
  npline = 25
  depth = '\n'.join(' '.join(('%3d' % c) for c in qcount[i:i+npline]) for i in range(1,qlen+1,npline))
  session.show_info('Hits per residue\n' + depth)

  # if molecule is None:
  #   # Align to best scoring hit.
  #   align_matches_to_match(matches, matches[0])
  # else:
  #   align_matches_to_chain(matches, molecule)
  
  align_pairs, rgroup = structure_alignment_pairs(br)
  n = len(matches)
  fixed = tuple(matches[g] for g in set(rgroup) if g < n)
  align_connected(align_pairs, fixed)
  report_alignment_groups(rgroup)

  mbest = br.best_match_per_residue()
  from numpy import unique
  ma_cover = [matches[mi] for mi in unique(mbest) if mi < n]
  random_ribbon_colors(ma_cover)
  show_covering_ribbons(mbest, matches)
  report_best_match_coverage(mbest, matches)

#  mc = covering_model(mbest, matches)
#  session.add_model(mc)

#  m = mosaic_model(br)
#  session.add_model(m)

  text_table = match_metrics_table(matches)
  session.show_info(text_table)
  session.show_status('Blast %s, show hits as ribbons...' % (seq_name,))
#  show_matches_as_ribbons(matches, molecule, chain_id)
  session.show_status('Blast %s, done...' % (seq_name,))

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
    self.blast_results = None
    self.sorted_matches = None
  def toggle_play(self):
    if self.frame is None:
      self.frame = 0
      self.show_none()
      v = self.session.view
      v.add_new_frame_callback(self.next_frame)
    else:
      self.stop_play()
  def stop_play(self):
    if self.frame is None:
      return
    self.frame = None
    v = self.session.view
    v.remove_new_frame_callback(self.next_frame)
  def matches(self):
    sm = self.sorted_matches
    br = self.session.blast_results
    if sm is None or br != self.blast_results:
      self.blast_results = br
      self.sorted_matches = sm = list(br.matches)
      sm.sort(key = lambda ma: ma.seq_match.qStart)
    return sm
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
      ma.show_ribbon()
    self.last_match = None
  def show_none(self):
    for ma in self.matches():
      if not hasattr(ma,'covering'):
        ma.hide_ribbon()
    for m in set(ma.mol for ma in self.matches()):
        m.display = False
    for m in set(ma.mol for ma in self.matches() if hasattr(ma,'covering')):
        m.display = True
    self.last_match = None
  def show_hit(self, match_num):
    self.match_num = match_num
    ma = self.matches()[match_num]
    lm = self.last_match
    if lm and not hasattr(lm,'covering'):
      lm.hide_ribbon()
    if not hasattr(ma,'covering'):
      ma.show_ribbon()
    self.last_match = ma
    s = self.session
    from os import path
    rmsd = '%.2f' % ma.align[1] if not ma.align is None else '.'
    sm = ma.seq_match
    s.show_status('%s chain %s, %.0f%% identity, %.0f%% coverage, rmsd %s   %s' %
                  (ma.pdb_id, ma.chain_id, 100*sm.identity(), 100*sm.coverage(),
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

def blast_shortcuts():
  sesarg = {'session_arg':True}
  mlmenu = 'Molecule'
  molcat = 'Molecule Display'

  sc = (
    ('/', cycle_blast_molecule_display, 'Cycle through display of blast molecules', molcat, sesarg, mlmenu),
    ('+', next_blast_molecule_display, 'Show next blast hit', molcat, sesarg, mlmenu),
    ('-', previous_blast_molecule_display, 'Show previous blast hit', molcat, sesarg, mlmenu),
    ('*', all_blast_molecule_display, 'Show all blast hits', molcat, sesarg, mlmenu),
    ('=', blast_color_by_coverage, 'Color blast query by coverage', molcat, sesarg, mlmenu),
    ('9', blast_show_matched_residues, 'Show blast hit residues that match query', molcat, sesarg, mlmenu),
    ('8', blast_show_coverage, 'Show each query residue using best e-value hit', molcat, sesarg, mlmenu),
    ('7', blast_show_coverage_full, 'Show each query residue using best e-value hit with overlap', molcat, sesarg, mlmenu),
    )

  return sc
