# vim:set et sw=4:
import os
import sys

PDB_DIR = '/databases/mol/pdb'
MMCIF_DIR = '/databases/mol/mmCIF'


class FlushFile:
    def __init__(self, fd):
        self.fd = fd

    def write(self, x):
        ret = self.fd.write(x)
        self.fd.flush()
        return ret

    def writelines(self, lines):
        ret = self.writelines(line)
        self.fd.flush()
        return ret

    def flush(self):
        return self.fd.flush()

    def close(self):
        return self.fd.close()

    def fileno(self):
        return self.fd.fileno()

# always flush stdout, even if output is to a file
sys.stdout = FlushFile(sys.stdout)


def compare(pdb_id, pdb_path, mmcif_path):
    # return True if they differ
    print('Comparing %s' % pdb_id)
    from chimera.core import io
    try:
        pdb_models = io.open(Chimera2_session, pdb_path)[0]
    except Exception as e:
        print("error: unable to open pdb file %s: %s" % (pdb_id, e))
        return True
    try:
        mmcif_models = io.open(Chimera2_session, mmcif_path)[0]
    except Exception as e:
        print("error: unable to open mmcif file %s: %s" % (pdb_id, e))
        return
    if len(pdb_models) != len(mmcif_models):
        print("error: %s: pdb version has %d models, mmcif version has %d" %
              (pdb_id, len(pdb_models), len(mmcif_models)))
        all_same = False
    else:
        all_same = True
        for p, m in zip(pdb_models, mmcif_models):
            same = True
            diff = p.mol_blob.num_atoms - m.mol_blob.num_atoms
            if diff != 0:
                print("error: %s: pdb has %d atoms %s than mmcif" % (
                    pdb_id, abs(diff), "fewer" if diff < 0 else "more"))
                same = False
            diff = p.mol_blob.num_bonds - m.mol_blob.num_bonds
            if diff != 0:
                print("error: %s: pdb has %d bonds %s than mmcif" % (
                    pdb_id, abs(diff), "fewer" if diff < 0 else "more"))
                same = False
            diff = p.mol_blob.num_residues - m.mol_blob.num_residues
            if diff != 0:
                print("error: %s: pdb has %d residues %s than mmcif" % (
                    pdb_id, abs(diff), "fewer" if diff < 0 else "more"))
                same = False
            diff = p.mol_blob.num_chains - m.mol_blob.num_chains
            if diff != 0:
                print("error: %s: pdb has %d chains %s than mmcif" % (
                    pdb_id, abs(diff), "fewer" if diff < 0 else "more"))
                same = False
            diff = p.mol_blob.num_coord_sets - m.mol_blob.num_coord_sets
            if diff != 0:
                print("error: %s: pdb has %d coord_sets %s than mmcif" % (
                    pdb_id, abs(diff), "fewer" if diff < 0 else "more"))
                same = False
            all_same = all_same and same
    if all_same:
        print('same: %s' % pdb_id)
    for m in pdb_models:
        m.delete()
    for m in mmcif_models:
        m.delete()
    return all_same


def file_gen(dir):
    # generate files in 2-character subdirectories of dir
    for root, dirs, files in os.walk(dir):
        if root == dir:
            root = ''
        else:
            root = root[len(dir) + 1:]
        # dirs = [d for d in dirs if len(d) == 2]
        dirs.sort()
        if not root:
            files = []
        else:
            files.sort()
        for f in files:
            yield root, f


def next_info(gen):
    try:
        return next(gen)
    except StopIteration:
        return None


def pdb_id(pdb_file):
    if len(pdb_file) != 11:
        return None
    n, ext = os.path.splitext(pdb_file)
    if ext != '.ent' or not n.startswith('pdb'):
        return None
    return n[3:]


def mmcif_id(mmcif_file):
    if len(mmcif_file) != 8:
        return None
    n, ext = os.path.splitext(mmcif_file)
    if ext != '.cif':
        return None
    return n


def compare_all():
    pdb_files = file_gen(PDB_DIR)
    mmcif_files = file_gen(MMCIF_DIR)

    pdb_info = next_info(pdb_files)
    mmcif_info = next_info(mmcif_files)

    all_same = True
    while pdb_info and mmcif_info:
        pdb_dir, pdb_file = pdb_info
        pid = pdb_id(pdb_file)
        mmcif_dir, mmcif_file = mmcif_info
        mid = mmcif_id(mmcif_file)
        if (pdb_dir < mmcif_dir or
                (pdb_dir == mmcif_dir and
                 (pid is not None and mid is not None and pid < mid) or
                 ((pid is None or mid is None) and pdb_file < mmcif_file))):
            print('Skipping pdb:', os.path.join(pdb_dir, pdb_file))
            pdb_info = next_info(pdb_files)
            continue
        if (mmcif_dir < pdb_dir or
                (pdb_dir == mmcif_dir and
                 (pid is not None and mid is not None and mid < pid) or
                 ((pid is None or mid is None) and mmcif_file < pdb_file))):
            print('Skipping mmcif:', os.path.join(mmcif_dir, mmcif_file))
            mmcif_info = next_info(mmcif_files)
            continue
        assert(pid == mid)
        same = compare(pid, os.path.join(PDB_DIR, pdb_dir, pdb_file),
                os.path.join(MMCIF_DIR, mmcif_dir, mmcif_file))
        all_same = all_same and same
        pdb_info = next_info(pdb_files)
        mmcif_info = next_info(mmcif_files)
    Chimera2_session.logger.clear()
    raise SystemExit(os.EX_OK if all_same else os.EX_DATAERR)


def compare_id(pdb_id):
    if len(pdb_id) != 4:
        print('PDB ids should be 4 characters long')
        raise SystemExit(os.EX_DATAERR)
    pdb_id = pdb_id.lower()
    if os.path.exists(PDB_DIR):
        pdb_path = os.path.join(PDB_DIR, pdb_id[1:3], 'pdb%s.ent' % pdb_id)
    else:
        pdb_path = "pdb:%s" % pdb_id
    if os.path.exists(MMCIF_DIR):
        mmcif_path = os.path.join(MMCIF_DIR, pdb_id[1:3], '%s.cif' % pdb_id)
    else:
        mmcif_path = "mmcif:%s" % pdb_id
    same = compare(pdb_id, pdb_path, mmcif_path)
    Chimera2_session.logger.clear()
    raise SystemExit(os.EX_OK if same else os.EX_DATAERR)


def usage():
    import sys
    print('%s: [-a] [-h] [-i pdbid] [--all] [--help] [--id pdb_id]' %
          sys.argv[0])


def main():
    import getopt
    import sys
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "ahi:", ["all", "help", "id="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        Chimera2_session.logger.clear()
        raise SystemExit(os.EX_USAGE)
    all = False
    pdb_id = None
    for opt, arg in opts:
        if opt in ('-a', '--all'):
            all = True
        elif opt in ('-h', '--help'):
            usage()
            Chimera2_session.logger.clear()
            raise SystemExit(os.EX_OK)
        elif opt in ('-i', '--id'):
            pdb_id = arg
    if not all and not pdb_id:
        usage()
        Chimera2_session.logger.clear()
        raise SystemExit(os.EX_USAGE)
    if all:
        if not os.path.exists(PDB_DIR) or not os.path.exists(MMCIF_DIR):
            print("pdb and/or mmCIF databases missing")
            Chimera2_session.logger.clear()
            raise SystemExit(os.EX_DATAERR)
        compare_all()
    if pdb_id:
        compare_id(pdb_id)


if __name__ == '__main__':
    main()
