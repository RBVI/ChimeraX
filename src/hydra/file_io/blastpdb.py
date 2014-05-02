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
    m = Match(desc, score, evalue, qStart, qEnd, qSeq, hStart, hEnd, hSeq) #SH
    self.matches.append(m)

class Match:
  """Data from a single BLAST hit."""

  def __init__(self, desc, score, evalue, qStart, qEnd, qSeq, hStart, hEnd, hSeq): #SH
    self.description = desc
    self.score = score
    self.evalue = evalue
    self.qStart = qStart - 1        # switch to 0-base indexing
    self.qEnd = qEnd - 1
    self.qSeq = qSeq
    self.hStart = hStart - 1        # switch to 0-base indexing
    self.hEnd = hEnd - 1
    self.hSeq = hSeq
    if len(qSeq) != len(hSeq):
      raise ValueError("sequence alignment length mismatch")

  def __repr__(self):
    return "<Match %s>" % (self.name(),)

  def name(self):
    return ' '.join('%s %s' % (id, ','.join(c)) for id,c,desc in self.pdb_chains())

  def pdb_chains(self):
    '''Return PDB chains for this match as a list of 3-tuples,
    containing pdb id, list of chain ids, and pdb description.'''

    pdbs = {}
    for pc in self.description.split('|'):
      pdb_id, chains, desc = pc.split(maxsplit = 2)
      cids = chains.split(',')
      pdbs[pdb_id] = (cids,desc)

    pcd = [(pdb_id,c,desc) for pdb_id,(c,desc) in pdbs.items()]
    pcd.sort()
    return pcd

  def load_structures(self, session, mmcif_dir):
    mols = []
    from . import mmcif
    for pdb_id,chains,desc in self.pdb_chains():
      m = mmcif.load_mmcif_local(pdb_id, session, mmcif_dir)
      if m:
        m.blast_match = self
        m.blast_match_chains = chains
        m.blast_match_description = desc
        mols.append(m)
    return mols

  # Map hit residue number to query residue number.  One is first character in sequence.
  def residue_number_map(self):
    rmap = {}
    hs, qs = self.hSeq, self.qSeq
    h, q = self.hStart+1, self.qStart+1
    n = min(len(hs), len(qs))
    for i in range(n):
      hstep = 0 if hs[i] in _GapChars else 1
      qstep = 0 if qs[i] in _GapChars else 1
      if hstep and qstep:
        rmap[h] = q
      h += hstep
      q += qstep
    return rmap

def check_hit_sequences_match_mmcif_sequences(mols):

  for m in mols:
    ma = m.blast_match
    chains = m.blast_match_chains

    # Compute gapless hit sequence from match
    hseq = ma.hSeq
    for c in _GapChars:
      hseq = hseq.replace(c,'')
    hseq = '.'*ma.hStart + hseq

    # Using mmcif files the residue number (label_seq_id) is the index into the sequence.
    # This is not true of PDB files.
    from ..molecule.molecule import chain_sequence
    for cid in chains:
      cseq = chain_sequence(m, cid)
      # Check that hit sequence matches PDB sequence
      if not sequences_match(hseq,cseq):
        print ('%s %s\n%s\n%s' % (m.name, cid, cseq, hseq))

def report_match_metrics(molecule, chain, mols):
  from ..molecule.molecule import residue_number_to_name
  qres = residue_number_to_name(molecule, chain)
  qatoms = molecule.atom_subset('CA', chain)
  lines = [' PDB Chain  RMSD  Coverage(#,%) Identity(#,%) Score  Description']
  for m in mols:
    ma = m.blast_match
    chains = m.blast_match_chains
    rmap = ma.residue_number_map()      # Hit to query residue number map.

    for cid in chains:
      hres = residue_number_to_name(m, cid)
      if len(hres) == 0:
        # TODO: This indicates that blast database chain identifier is not present in
        # the mmCIF file.  This can happen if the blast database was built using PDB
        # chain identifiers which can differ from mmcif chain identifiers.
        print ('Warning: mmCIF %s has no chain sequence %s' % (m.name, cid))
        continue

      # Compute sequence identity between hit and query.
      pairs = eqpairs = 0
      for hi,qi in rmap.items():
        if hi in hres and qi in qres:
          pairs += 1
          if hres[hi] == qres[qi]:
            eqpairs += 1

      # Find paired hit and query residues for doing an alignment.
      qrnum = set(qres.keys())
      hpres = set(r for r in hres.keys() if r in rmap and rmap[r] in qrnum)
      qpres = set(rmap[r] for r in hpres)
      hatoms = m.atom_subset('CA', cid)
      hpatoms = hatoms.subset([i for i,r in enumerate(hatoms.residue_numbers()) if r in hpres])
      qpatoms = qatoms.subset([i for i,r in enumerate(qatoms.residue_numbers()) if r in qpres])

      # Check that number of paired CA atoms is same for hit and query.  Sanity check.
      if hpatoms.count() != qpatoms.count():
        print (m.name, cid, hpatoms.count(), qpatoms.count(), len(rmap))
        print (hpatoms.names())
        print (qpatoms.names())
        continue

      # Compute RMSD of aligned hit and query.
      from ..molecule import align
      tf, rmsd = align.align_points(hpatoms.coordinates(), qpatoms.coordinates())

      # Create table output line showing how well hit matches query.
      name = m.name[:-4] if m.name.endswith('.cif') else m.name
      desc = m.blast_match_description if cid == chains[0] else ''
      lines.append('%4s %3s %7.2f %5d %5.0f   %5d %5.0f  %5d    %s'
                   % (name, cid, rmsd, pairs, 100*float(pairs)/len(qres),
                      eqpairs, 100.0*eqpairs/pairs, ma.score, desc))

  print('\n'.join(lines))

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

def summarize_results(results):
  r = results
  np = sum(len(m.pdb_chains()) for m in r.matches)
  nc = sum(sum(len(c) for id,c,desc in m.pdb_chains()) for m in r.matches)
  lines = ['%s %d sequence matches, %d PDBs, %d chains' % (results.name, len(r.matches), np, nc)]
#  for m in r.matches:
#    lines.append('%d' % m.score)
#    for id,chains,desc in m.pdb_chains():
#      lines.append(' %s %s %s' % (id, ','.join(chains), desc))
  msg = '\n'.join(lines)
  return msg

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

def write_fasta(name, seq, file):
  file.write('>%s\n%s\n' % (name, seq))

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

  from ..ui.commands import molecule_arg, string_arg, parse_arguments
  req_args = (('molecule', molecule_arg),
              ('chain', string_arg),)
  opt_args = ()
  kw_args = (('blastProgram', string_arg),
             ('blastDatabase', string_arg),
             ('mmcifDirectory', string_arg),)

  kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
  kw['session'] = session
  blast(**kw)

def blast(molecule, chain, session,
          blastProgram = '/usr/local/ncbi/blast/bin/blastp',
          blastDatabase = '/usr/local/ncbi/blast/db/mmcif',
          mmcifDirectory = '/usr/local/mmCIF'):

  # Write FASTA sequence file for molecule
  from . import mmcif
  s = mmcif.mmcif_sequences(molecule.path)
  start,seq,desc = s[chain]
  from os.path import basename, splitext
  prefix = splitext(basename(molecule.path))[0]
  import tempfile
  fasta_file = tempfile.NamedTemporaryFile('w', suffix = '.fasta', prefix = prefix+'_', delete = False)
  sname = '%s %s' % (prefix, chain)
  write_fasta(sname, seq, fasta_file)
  fasta_file.close()

  # Run blastp standalone program and parse results
  blast_output = splitext(fasta_file.name)[0] + '.xml'
  results = run_blastp(molecule.name, fasta_file.name, blast_output, blastProgram, blastDatabase)

  # Load matching structures and report match metrics
  print (summarize_results(results))
  mols = sum([m.load_structures(session, mmcifDirectory) for m in results.matches], [])
  check_hit_sequences_match_mmcif_sequences(mols)
  report_match_metrics(molecule, chain, mols)
