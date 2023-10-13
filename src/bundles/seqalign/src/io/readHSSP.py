# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
reads a HSSP format file
"""

from chimerax.atomic import Sequence
from ..parse import FormatSyntaxError, make_readable

def read(session, f):
    doing = None
    sequences = []
    header_ok = False
    line_num = 0
    align_start_index = None
    for line in f.readlines():
        if doing == 'alignments':
            # don't strip() alignment section since it has significant leading spaces
            line = line.rstrip()
        else:
            line = line.strip()
        line_num += 1
        if not header_ok:
            if line.lower().startswith("hssp"):
                header_ok = True
                continue
            raise FormatSyntaxError("No initial HSSP header line")
        if line.startswith('##'):
            if doing == 'proteins' and not sequences:
                raise FormatSyntaxError("No entries in PROTEINS section")
            try:
                doing = line.split()[1].lower()
            except IndexError:
                doing = None
            if doing == 'alignments':
                try:
                    hashes, alignments, begin, dash, end = line.strip().split()
                    begin = int(begin)
                    end = int(end)
                except ValueError:
                    raise FormatSyntaxError("ALIGNMENTS line (line #%d) not of the form: "
                        "## ALIGNMENTS (number) - (number)" % line_num)
            continue
        if doing == 'proteins':
            if not line[0].isdigit():
                continue
            try:
                seq_name = line.split()[2]
            except IndexError:
                raise FormatSyntaxError("Line %d in PROTEINS section does not start with "
                    "[integer] : [sequence name]" % line_num)
            sequences.append(Sequence(name=make_readable(seq_name)))
        elif doing == 'alignments':
            if line.lstrip().lower().startswith('seqno'):
                try:
                    align_start_index = line.index('.')
                except Exception:
                    raise FormatSyntaxError("No indication of alignment starting column "
                        "('.' character) in SeqNo line in ALIGNMENTS section")
                continue
            if align_start_index == None:
                raise FormatSyntaxError("No initial SeqNo line in ALIGNMENTS section")
            block = line[align_start_index:]
            if not block:
                raise FormatSyntaxError("No alignment block given on line %d" % line_num)
            block_len = end - begin + 1
            if len(block) > block_len:
                raise FormatSyntaxError("Too many characters (%d, only %d sequences) in "
                    "alignment block given on line %d" % (len(block), block_len, line_num))
            block = block + ' ' * (block_len - len(block))
            for seq, c in zip(sequences[begin-1:end], block):
                seq.append(c)
    f.close()
    return sequences, {}, {}
