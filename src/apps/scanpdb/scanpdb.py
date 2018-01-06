# vim:set et sw=4:

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

import os
import sys
from chimerax.core import logger

MMCIF_DIR = '/databases/mol/mmCIF'

session = session  # noqa
session.silent = False

show_memory = False
previous_maxrss = None

# always line-buffer stdout, even if output is to a file
if not sys.stdout.line_buffering:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), sys.stdout.encoding,
                                  sys.stdout.errors, line_buffering=True)


def memory_usage():
    global previous_maxrss
    import resource
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    total = rusage.ru_maxrss  # in kilobytes
    if previous_maxrss is None:
        previous_maxrss = total
        return
    delta = total - previous_maxrss
    if delta < 1024:
        delta_suffix = 'K'
    else:
        delta /= 1024
        if delta < 1024:
            delta_suffix = 'M'
        else:
            delta /= 1024
            delta_suffix = 'G'
    previous_maxrss = total
    if total < 1024:
        total_suffix = 'K'
    else:
        total /= 1024
        if total < 1024:
            total_suffix = 'M'
        else:
            total /= 1024
            total_suffix = 'G'
    print('  used %.3g%s, max %.3g%s' % (delta, delta_suffix, total, total_suffix))


class MyLog(logger.PlainTextLog):
    # Only want warnings and errors
    excludes_other_logs = True
    ignore_info = True

    def log(self, level, msg):
        if self.ignore_info and level == logger.Log.LEVEL_INFO:
            return True
        import sys
        encoding = sys.stdout.encoding.lower()
        if encoding != 'utf-8' and isinstance(msg, str):
            msg = msg.encode(encoding, 'replace').decode(encoding)
        print(msg)
        return True

    def status(self, msg, color, secondary):
        return True

class MyCollator(logger.CollatingLog):

    def log(self, level, msg):
        if level == logger.Log.LEVEL_INFO:
            return True
        return super().log(level, msg)

    def log_summary(self, logger, summary_title, collapse_similar=True):
        my_log = list(logger.logs)[-1]
        my_log.ignore_info = False
        super().log_summary(logger, summary_title, collapse_similar)
        my_log.ignore_info = True
logger.CollatingLog = MyCollator

def check(session, pdb_id, mmcif_path):
    from chimerax.core.commands.open import open
    print('checking: %s' % pdb_id)
    try:
        mmcif_models = open(session, pdb_id, format='mmcif')
    except Exception as e:
        session.logger.error("%s: unable to open mmcif format %s" % (pdb_id, e))
        return
    from chimerax.core.commands.close import close
    close(session, mmcif_models)
    if show_memory:
        memory_usage()


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


def mmcif_id(mmcif_file):
    if len(mmcif_file) != 8:
        return None
    n, ext = os.path.splitext(mmcif_file)
    if ext != '.cif':
        return None
    return n


def check_all(session, start_pdb_id):
    from datetime import datetime, timedelta
    start_time = datetime.now()
    mmcif_files = file_gen(MMCIF_DIR)

    mmcif_info = next_info(mmcif_files)

    while mmcif_info:
        mmcif_dir, mmcif_file = mmcif_info
        pid = mmcif_id(mmcif_file)
        if start_pdb_id:
            if pid == start_pdb_id:
                start_pdb_id = None
        else:
            check(session, pid, os.path.join(MMCIF_DIR, mmcif_dir, mmcif_file))
        mmcif_info = next_info(mmcif_files)
    end_time = datetime.now()
    delta = end_time - start_time
    days = delta // timedelta(days=1)
    delta -= days * timedelta(days=1)
    hours = delta // timedelta(hours=1)
    delta -= hours * timedelta(hours=1)
    minutes = delta // timedelta(minutes=1)
    delta -= minutes * timedelta(minutes=1)
    seconds = delta // timedelta(seconds=1)
    delta -= seconds * timedelta(seconds=1)
    microseconds = delta // timedelta(microseconds=1)
    print('Total time: %d days, %d hours, %d minutes, %d seconds, %d microseconds' % (days, hours, minutes, seconds, microseconds))
    print('Total time: %s' % (end_time - start_time))
    raise SystemExit(os.EX_OK)


def check_id(session, pdb_id):
    if len(pdb_id) != 4:
        session.logger.error('PDB ids should be 4 characters long')
        raise SystemExit(os.EX_DATAERR)
    if os.path.exists(MMCIF_DIR):
        mmcif_path = os.path.join(MMCIF_DIR, pdb_id[1:3], '%s.cif' % pdb_id)
    else:
        mmcif_path = pdb_id
    return check(session, pdb_id, mmcif_path)


def usage():
    import sys
    print(
        'usage: %s: [-a|--all] [-h|--help] [-m|--memory] [pdb_id(s)]' % sys.argv[0])
    print(
        '''Check structures produced by the mmCIF reader.
        Give one or more pdb identifiers to check just those structures.
        Or give --all (-a) option to check all of the structures in
        /databases/mol/mmCIF/.

        If --all and a pdb_id are given, then start at the PDB id after it.''')

def main():
    global show_memory
    import getopt
    import sys
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "ahms:", ["all", "help"])
    except getopt.GetoptError as err:
        session.logger.error(err)
        usage()
        raise SystemExit(os.EX_USAGE)
    all = False
    for opt, arg in opts:
        if opt in ('-a', '--all'):
            all = True
        elif opt in ('-m', '--memory'):
            show_memory = True
        elif opt in ('-h', '--help'):
            usage()
            raise SystemExit(os.EX_OK)
    if (not all and not args) or (all and len(args) > 1):
        usage()
        raise SystemExit(os.EX_USAGE)

    session.logger.add_log(MyLog())
    if show_memory:
        memory_usage()
    if all:
        if not os.path.exists(MMCIF_DIR):
            session.logger.error("mmCIF database is missing")
            raise SystemExit(os.EX_DATAERR)
        okay = check_all(session, args[0] if args else None)
    else:
        okay = True
        for pdb_id in args:
            is_okay = check_id(session, pdb_id)
            if not is_okay:
                okay = False
    raise SystemExit(os.EX_OK if okay else os.EX_DATAERR)


if __name__ == '__main__':
    main()
