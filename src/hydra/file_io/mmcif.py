def open_mmcif_file(path, session):
  '''
  Open an mmCIF file.
  '''
  from os.path import basename
  from time import time
  ft0 = time()
  f = open(path, 'r')
  text = f.read()
  f.close()
  ft1 = time()
  from .. import _image3d
  t0 = time()
  xyz, element_nums, chain_ids, res_nums, res_names, atom_names = \
      _image3d.parse_mmcif_file(text)
  t1 = time()
#  session.show_info('Read %s %d atoms at %d per second\n' %
#                    (basename(path), len(xyz), int(len(xyz)/(ft1-ft0))))
#  session.show_info('Parsed %s %d atoms at %d per second\n' %
#                    (basename(path), len(xyz), int(len(xyz)/(t1-t0))))
#  session.show_info('Read+Parsed %s %d atoms at %d per second\n' %
#                    (basename(path), len(xyz), int(len(xyz)/((t1-t0)+(ft1-ft0)))))
  session.show_info('Read %s %d atoms\n' % (basename(path), len(xyz)))
  from ..molecule import Molecule
  m = Molecule(path, xyz, element_nums, chain_ids, res_nums, res_names, atom_names)
  return m

def load_mmcif_local(id, session, mmcif_dir):
    '''Load an mmCIF file given its id from a local copy of the "divided" database.'''
    from os.path import join, exists
    p = join(mmcif_dir, id[1:3].lower(), '%s.cif' % id.lower())
    if not exists(p):
        return None
    m = open_mmcif_file(p, session)
    return m

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
