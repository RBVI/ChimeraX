_GapChars = "-. "

class Blast_Output_Parser:
        """Parser for XML output from blastp (tested against version 2.2.29+)."""

        def __init__(self, name, xmlText):

                self.name = name

                # Bookkeeping data
                self.matches = []
                self.matchDict = {}
                self._gapCount = None

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

                # Go back and fix up hit sequences so that they all align
                # with the query sequence
                self._alignSequences()

        def _text(self, parent, tag):
                e = parent.find(tag)
                return e is not None and e.text.strip() or None

        def _extractRoot(self, oe):
                self.database = self._text(oe, "BlastOutput_db")
                self.query = self._text(oe, "BlastOutput_query-ID")
                self.queryLength = int(self._text(oe, "BlastOutput_query-len"))
                self._gapCount = [ 0 ] * self.queryLength
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
                hidParts = hid.split('|')
                if len(hidParts) % 2 != 1:
                        import sys
                        print >> sys.stderr, "Skipping unexpected id: %s" % hid
                        return
                chain = hidParts.pop(-1).strip()
                gi = None
                pdb = None
                for i in range(0, len(hidParts), 2):
                        if hidParts[i] == "gi":
                                gi = hidParts[i + 1].strip()
                        elif hidParts[i] == "pdb":
                                pdb = hidParts[i + 1].strip()
                                if chain:
                                        pdb = pdb + '_' + chain
                desc = self._text(he, "Hit_def")
                for hspe in he.findall("./Hit_hsps/Hsp"):
                        self._extractHSP(hspe, gi, pdb, desc)

        def _extractHSP(self, hspe, gi, pdb, desc):
                score = int(float(self._text(hspe, "Hsp_bit-score"))) #SH
                evalue = float(self._text(hspe, "Hsp_evalue"))
                qSeq = self._text(hspe, "Hsp_qseq")
                qStart = int(self._text(hspe, "Hsp_query-from"))
                qEnd = int(self._text(hspe, "Hsp_query-to"))
                self._updateGapCounts(qSeq, qStart, qEnd)
                hSeq = self._text(hspe, "Hsp_hseq")
                hStart = int(self._text(hspe, "Hsp_hit-from"))
                hEnd = int(self._text(hspe, "Hsp_hit-to"))
                m = Match(gi, pdb, desc, score, evalue, qStart, qEnd, qSeq, hStart, hEnd, hSeq) #SH
                self.matches.append(m)
                self.matchDict[gi] = m

        def _updateGapCounts(self, seq, start, end):
                start -= 1        # Switch to 0-based indexing
                count = 0
                for c in seq:
                        if c in _GapChars:
                                count += 1
                        else:
                                oldCount = self._gapCount[start]
                                self._gapCount[start] = max(oldCount, count)
                                start += 1
                                count = 0

        def _alignSequences(self):
                for m in self.matches:
                        m.matchSequenceGaps(self._gapCount)

class Match:
        """Data from a single BLAST hit."""

        def __init__(self, gi, pdb, desc, score, evalue, qStart, qEnd, qSeq, hStart, hEnd, hSeq): #SH
                self.gi = gi
                self.pdb = pdb
                self.description = desc.strip()
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
                self.sequence = ""

        def __repr__(self):
                return "<Match %s (gi=%s)>" % (self.pdb, self.gi)

        def name(self):
                return self.pdb or self.gi

        def printSequence(self, f, prefix, perLine=60):
                for i in range(0, len(self.sequence), perLine):
                        f.write("%s%s\n" % (prefix, self.sequence[i:i+perLine]))

        def matchSequenceGaps(self, gapCount):
                seq = []
                # Insert gap for head of query sequence that did not match
                for i in range(self.qStart):
                        seq.append('.' * (gapCount[i] + 1))
                start = self.qStart
                count = 0
                # Add all the sequence data from this HSP
                for i in range(len(self.qSeq)):
                        if self.qSeq[i] in _GapChars:
                                # If this is a gap in the query sequence,
                                # then the hit sequence must be an insertion.
                                # Add the insertion to the final sequence
                                # and increment number of gaps we have added
                                # thus far.
                                seq.append(self.hSeq[i])
                                count += 1
                        else:
                                # If this is not a gap, then we have to make
                                # sure that we have inserted enough gaps for
                                # the longest insertion by any sequence (as
                                # computed in "gapCount").  Then we add the
                                # hit sequence character that matches this
                                # query sequence character, and increment
                                # out query sequence index ("start").
                                if count > gapCount[start]:
                                        print ("start", start)
                                        print ("count", count, ">", gapCount[start])
                                        raise ValueError("cannot align sequences")
                                if count < gapCount[start]:
                                        seq.append('-' * (gapCount[start] - count))
                                seq.append(self.hSeq[i])
                                count = 0
                                start += 1
                # Append gap for tail of query sequence that did not match
                while start < len(gapCount):
                        seq.append('.' * (gapCount[start] + 1))
                        start += 1
                self.sequence = ''.join(seq)

        def pdbIds(self):
                sd = split_description(self.description)
                ids = []
                for i, d in enumerate(sd):
                        if i == 0:
                                ids.append((self.pdb, d))
                        else:
                                j = d.find('|pdb|')
                                if j >= 0:
                                        id = d[j+5:][:6].replace('|','_')
                                        de = d[j+11:].strip()
                                        ids.append((id,de))
                return ids

        def chains(self):
                pdb = {}
                for idc, desc in self.pdbIds():
                        id, c = idc.split('_')
                        if id in pdb:
                                pdb[id][0].append(c)
                        else:
                                if desc.startswith('Chain '):
                                        desc = desc[9:]
                                pdb[id] = ([c],desc)
                pcd = [(id,c,desc) for id,(c,desc) in pdb.items()]
                pcd.sort()
                return pcd

        def dump(self, f):
                print >> f, self
                self.printSequence(f, '')

        def load_structures(self, session, mmcif_dir):

                mols = []
                for id,chains,desc in self.chains():
                        m = load_mmcif(id, session, mmcif_dir)
                        if m:
                                m.blast_match = self
                                m.blast_match_chains = chains
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
                for cid in chains:
                        cseq = chain_sequence(m, cid)
                        # Check that hit sequence matches PDB sequence
                        if not sequences_match(hseq,cseq):
                                print ('%s %s\n%s\n%s' % (m.name, cid, cseq, hseq))

def report_match_metrics(molecule, chain, mols):
        qres = residue_number_to_name(molecule, chain)
        qatoms = molecule.atom_subset('CA', chain)
        qrnum = set(qres.keys())
        for m in mols:
                ma = m.blast_match
                chains = m.blast_match_chains
                rmap = ma.residue_number_map()
                for cid in chains:
                        mres = residue_number_to_name(m, cid)
                        if len(mres) == 0:
                                # TODO: This indicates that mmCIF uses a different chain identifier
                                # from the PDB file. The blast database is using the PDB chain identifier.
                                print ('Warning: mmCIF %s has no chain sequence %s' % (m.name, cid))
                                continue
                        pairs = eqpairs = 0
                        for hi,qi in rmap.items():
                                if hi in mres and qi in qres:
                                        pairs += 1
                                        if mres[hi] == qres[qi]:
                                                eqpairs += 1
#                        print ('%s %s %d %d' % (m.name, cid, pairs, eqpairs))
#                        print ('%s\n%s' % (ma.qSeq, ma.hSeq))

                        # TODO: Need hit and query CA atoms that are matched and exist in hit/query structures.
                        hatoms = m.atom_subset('CA', cid)
                        hpres = set(r for r in mres.keys() if r in rmap and rmap[r] in qrnum)
                        hpatoms = hatoms.subset([i for i,r in enumerate(hatoms.residue_numbers()) if r in hpres])
                        qpres = set(rmap[r] for r in hpres)
                        qpatoms = qatoms.subset([i for i,r in enumerate(qatoms.residue_numbers()) if r in qpres])

                        if hpatoms.count() != qpatoms.count():
                                print (m.name, cid, hpatoms.count(), qpatoms.count(), len(rmap))
                                print (hpatoms.names())
                                print (qpatoms.names())
                                continue

                        from ..molecule import align
                        tf, rmsd = align.align_points(hpatoms.coordinates(), qpatoms.coordinates())
                        print ('%s %s rmsd %.2f, paired residues %d, identity %.0f%%'
                               % (m.name, cid, rmsd, pairs, 100.0*eqpairs/pairs))

def residue_number_to_name(mol, chain_id):
        atoms = mol.atom_subset('CA', chain_id)
        rnums = atoms.residue_numbers()
        rnames = atoms.residue_names()
        return dict(zip(rnums, rnames))

def chain_sequence(mol, chain_id):
        atoms = mol.atom_subset('CA', chain_id)
        if atoms.count() == 0:
                return ''
        rnums = atoms.residue_numbers()
        rnames = atoms.residue_names()
        nr = rnums.max()
        seq = ['.']*nr
        from ..molecule.residue_codes import res3to1
        for i,n in zip(rnums,rnames):
                seq[i-1] = res3to1(n.tostring().decode('utf-8'))
        cseq = ''.join(seq)
        return cseq

def sequences_match(s, seq):
        n = min(len(s), len(seq))
        for i in range(n):
                if s[i] != seq[i] and s[i] != '.' and s[i] != 'X' and seq[i] != '.' and seq[i] != 'X':
                        return False
        return True

def split_description(desc):
        ds = []
        d = desc
        while True:
                i = d.find(' >gi')
                if i == -1:
                        break
                ds.append(d[:i])
                d = d[i+1:]
        ds.append(d)
        return ds

def load_pdb(id, session, pdb_dir = '/usr/local/pdb'):

        from os.path import join, exists
        p = join(pdb_dir, id[1:3].lower(), 'pdb%s.ent' % id.lower())
        if not exists(p):
                return None
        from . import pdb
        m = pdb.open_pdb_file(p, session)
        return m

def load_mmcif(id, session, mmcif_dir):

        from os.path import join, exists
        p = join(mmcif_dir, id[1:3].lower(), '%s.cif' % id.lower())
        if not exists(p):
                return None
        from . import pdb
        m = pdb.open_mmcif_file(p, session)
        return m

def pdb_residue_numbering(path):
        rnum = {}
        f = open(path)
        while True:
                line = f.readline()
                if line.startswith('DBREF'):
                        if line[26:32] == 'PDB   ':
                                rnum[line[12]] = int(line[14:18])
                elif line.startswith('SEQRES') or line == '':
                        break
        f.close()
        return rnum

def summarize_results(results):
        r = results
        np = sum(len(m.chains()) for m in r.matches)
        nc = sum(sum(len(c) for id,c,desc in m.chains()) for m in r.matches)
        lines = ['%s %d matches, %d pdbs, %d chains' % (results.name, len(r.matches), np, nc)]
        for m in r.matches:
                lines.append('%d' % m.score)
                for id,chains,desc in m.chains():
                        lines.append(' %s %s %s' % (id, ','.join(chains), desc))
        msg = '\n'.join(lines)
        return msg

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

def mmcif_sequences(mmcif_path):
        '''
        Read an mmcif file to find how residue numbers map to sequence positions.
        This is not available in PDB format.
        '''
        eps, sa = read_mmcif_tables(mmcif_path, ('_entity_poly_seq', '_struct_asym'))
        if sa is None or eps is None:
                print('Missing sequence info in mmCIF file %s (_entity_poly_seq and _struct_asym tables)' % mmcif_path)
                return {}
        ce = sa.mapping('id', 'entity_id')
        es = eps.mapping('num', 'mon_id', foreach = 'entity_id')

        eseq = {}
        from ..molecule.residue_codes import res3to1
        for eid, seq in es.items():
                rnums = [int(i) for i in seq.keys()]
                rnums.sort()
                r0,r1 = rnums[0], rnums[-1]
                if rnums != list(range(r0,r1+1)):
                        from os.path import basename
                        print(basename(mmcif_path), 'non-contiguous sequence for entity', eid, 'residue numbers', rnums)
                        continue
                eseq[eid] = (r0, ''.join(res3to1(seq[str(i)]) for i in rnums))

        cseq = {}
        for cid, eid in ce.items():
                if eid in eseq:
                        cseq[cid] = eseq[eid]
        
        return cseq

def read_mmcif_tables(mmcif_path, table_names):
        f = open(mmcif_path)
        tables = {}
        tname = None
        while True:
                line = f.readline()
                if tname is None:
                        if line == '':
                                break
                        for tn in table_names:
                                if line.startswith(tn + '.'):
                                        tname = tn
                                        tags = [line.split('.')[1].strip()]
                                        values = []
                                        break
                elif line.startswith(tname + '.'):
                        tags.append(line.split('.')[1].strip())
                elif line.startswith('#') or line == '':
                        tables[tname] = mmCIF_Table(tname, tags, values)
                        tname = None
                else:
                        values.append(line.split())
        f.close()
        tlist = [tables.get(tn, None) for tn in table_names]
        return tlist

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

def write_fasta(name, seq, file):
        file.write('>%s\n%s\n' % (name, seq))

def run_blastp(name, fasta_path, output_path, blast_program, blast_database):
        # ../bin/blastp -db pdbaa -query 2v5z.fasta -outfmt 5 -out test.xml
        cmd = ('env BLASTDB=%s %s -db pdbaa -query %s -outfmt 5 -out %s' %
               (blast_database, blast_program, fasta_path, output_path))
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
          blastDatabase = '/usr/local/ncbi/blast/db',
          mmcifDirectory = '/usr/local/mmCIF'):

        # Write FASTA sequence file for molecule
        s = mmcif_sequences(molecule.path)
        start,seq = s[chain]
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
        print (summarize_results(results))

        # Load matching structures and report match metrics
        mols = sum([m.load_structures(session, mmcifDirectory) for m in results.matches], [])
        check_hit_sequences_match_mmcif_sequences(mols)
        report_match_metrics(molecule, chain, mols)
