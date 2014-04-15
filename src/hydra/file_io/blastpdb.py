_GapChars = "-. "

class Blast_Output_Parser:
        """Parser for XML output from blastp (tested against version 2.2.29+)."""

        def __init__(self, xmlText):
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

def load_pdb(id, session, dbdir = '/usr/local/pdb'):

        from os.path import join, exists
        p = join(dbdir, id[1:3].lower(), 'pdb%s.ent' % id.lower())
        if not exists(p):
                return None
        from . import pdb
        m = pdb.open_pdb_file(p, session)
        return m

def load_mmcif(id, session, dbdir = '/usr/local/mmCIF'):

        from os.path import join, exists
        p = join(dbdir, id[1:3].lower(), '%s.cif' % id.lower())
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

def report_blast_results(mol, xml_path, session):
        f = open(xml_path)
        xml_text = f.read()
        f.close()

        p = Blast_Output_Parser(xml_text)
        np = sum(len(m.chains()) for m in p.matches)
        nc = sum(sum(len(c) for id,c,desc in m.chains()) for m in p.matches)
        print ('%s %d matches, %d pdbs, %d chains' % (mol.name, len(p.matches), np, nc))
#        for m in p.matches:
#                print(m.score)
#                for id,chains,desc in m.chains():
#                        print(' ', id, chains, desc)

        mols = []
        for match in p.matches:
                for id,chains,desc in match.chains():
                        m = load_mmcif(id, session)
                        if m:
#                                csnum = sequence_residue_numbers(m.path)
#                                m.chain_first_residue_number = rnum = pdb_residue_numbering(m.path)
                                mols.append(m)
                                from ..molecule.residue_codes import res3to1
                                hseq = match.hSeq
                                for c in _GapChars:
                                        hseq = hseq.replace(c,'')
                                hseq = '.'*match.hStart + hseq
                                for cid in chains:
                                        atoms = m.atom_subset('CA', cid)
                                        rnums = atoms.residue_numbers()
                                        rnames = atoms.residue_names()
                                        rseq = [res3to1(nm.tostring().decode('utf-8')) for nm in rnames]
#                                        if not cid in csnum:
#                                                print('No sequence numbering info %s, chain %s' % (m.name, cid))
#                                                continue
#                                        snum = csnum[cid]
#                                        roff = rnum.get(cid,1) + match.hStart   # Residue number for start of hit sequence
                                        hlen = len(hseq)
                                        # Check that hit sequence matches PDB sequence
#                                        print ('sm', len(hseq), rnums)
#                                        hs = [hseq[i-roff] for i in rnums if i-roff < hlen]
#                                        sm = sum(rn == hseq[i-roff] for i,rn in zip(rnums,rseq) if i-roff < hlen)
#                                        hs = [hseq[snum[i]-1] for i in rnums if snum[i] <= hlen]
#                                        sm = sum(rn == hseq[snum[i]-1] for i,rn in zip(rnums,rseq) if snum[i] <= hlen)
                                        hs = [hseq[i-1] for i in rnums if i <= hlen]
                                        sm = sum((i > hlen or rn == hseq[i-1] or hseq[i-1] == '.' or hseq[i-1] == 'X')
                                                 for i,rn in zip(rnums,rseq))
                                        if sm != len(rnums):
                                                print (m.name, cid, len(rnums), sm)
                                                print (''.join(rseq))
                                                print (''.join(hs))
#                                                print (rnums[0], roff, match.hStart, match.hEnd, len(rseq))
                                                print (rnums[0], match.hStart, match.hEnd, len(rseq))
                                                print (hseq)
                                                
#                                print (m.name, len(m.xyz))

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

def blast_command(cmdname, args, session):

        from ..ui.commands import molecule_arg, parse_arguments
        req_args = (('molecule', molecule_arg),)
        opt_args = ()
        kw_args = ()

        kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
        kw['session'] = session
        blast(**kw)

def blast(molecule, session):
        # ../bin/blastp -db pdbaa -query 2v5z.fasta -outfmt 5 -out test.xml
        path = '/usr/local/ncbi/blast/db/test.xml'
        report_blast_results(molecule, path, session)
#        path = '/Users/goddard/Downloads/Chimera/PDB/1GOS.cif'
#        srn = sequence_residue_numbers(path)
#        print (srn)
