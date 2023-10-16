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
MSF reads a Multiple Sequence Format (MSF) file.
See Wisconsin Package (GCG) User's Guide, pp 2-28.
"""

from ..parse import FormatSyntaxError, make_readable

def read(session, f):
    msf = MSF(session, f)
    file_markups = {}
    if msf.multalin:
        for i, seq in enumerate(msf.sequence_list):
            if seq.name.lower() == "consensus":
                c = msf.sequence_list.pop(i)
                file_markups["multalin consensus"] = str(c)
                break
    return msf.sequence_list, msf.file_attrs, file_markups

import re
class MSF:

    _Hdr = re.compile('\s*(\S*)\s*'            # name
                'MSF:\s*(\S*)\s*'    # length?
                'Type:\s*(\S*)\s*'    # type
                '(.*)\s*'        # date/time
                'Check:\s*(\S*)\s*'    # checksum
                '\.\.')            # signature
    _MultalinHdr = re.compile('\s*(\S.*)\s*'    # name
                'MSF:\s*(\S*)\s*'    # length?
                            # missing type
                            # missing date/time
                'Check:\s*(\S*)\s*'    # checksum
                '\.\.')            # signature

    _Sum = re.compile('\s*Name:\s*(\S*)\s*o*\s*'    # name
                'Len:\s*(\S*)\s*'    # length
                'Check:\s*(\S*)\s*'    # checksum
                'Weight:\s*(\S*)\s*')    # weight

    def __init__(self, session, f):
        self.session = session
        self.file_attrs = {}
        if isinstance(f, str):
            file = open(f)
            self._read_msf(file)
            file.close()
        else:
            self._read_msf(f)

    def _read_msf(self, f):
        self._read_header(f)
        self._read_sequences(f)
        self._read_alignment(f)

    def _read_header(self, f):
        while True:
            line = f.readline()
            if not line:
                raise FormatSyntaxError("Unexpected end of file while reading MSF header")
            m = MSF._Hdr.match(line)
            if m is not None:
                name = m.group(1).strip()
                if name:
                    self.file_attrs['MSF name'] = name
                self.file_attrs['MSF length'] = m.group(2)
                self.file_attrs['MSF type'] = m.group(3)
                date = m.group(4).strip()
                if date:
                    self.file_attrs['date'] = date
                self.file_attrs['MSF checksum'] = m.group(5)
                self.multalin = False
                break
            m = MSF._MultalinHdr.match(line)
            if m is not None:
                self.file_attrs['MSF name'] = m.group(1)
                self.file_attrs['MSF length'] = m.group(2)
                self.file_attrs['MSF checksum'] = m.group(3)
                self.multalin = True
                break
            try:
                self.file_attrs['MSF header'] += line
            except KeyError:
                self.file_attrs['MSF header'] = line
        try:
            self.file_attrs['MSF header'] = self.file_attrs['MSF header'].strip()
        except KeyError:
            self.file_attrs['MSF header'] = ''

    def _read_sequences(self, f):
        from chimerax.atomic import Sequence
        self.sequence_list = []
        register_weight = False
        while 1:
            line = f.readline()
            if not line:
                raise FormatSyntaxError('no alignment separator')
            if line == '//\n' or line == '//\r\n':
                break
            m = MSF._Sum.match(line)
            if m is not None:
                name = m.group(1)
                length = m.group(2)
                check = m.group(3)
                try:
                    weight = float(m.group(4))
                except ValueError:
                    weight = 1.0
                else:
                    register_weight = True
                s = Sequence(name=make_readable(name))
                self.sequence_list.append(s)
                s.attrs = {}
                s.attrs['MSF length'] = length
                s.attrs['MSF check'] = check
                s.attrs['MSF weight'] = s.weight = weight
        if not self.sequence_list:
            raise FormatSyntaxError('No sequences found in header')
        if register_weight:
            Sequence.register_attr(self.session, "weight", "MSF reader", attr_type=float)

    def _read_alignment(self, f):
        line = f.readline()
        if not line:
            raise FormatSyntaxError('no alignment data')
        while self._read_block(f):
            pass

    def _read_block(self, f):
        line = f.readline()
        if not line:
            return False
        if line == '\n' or line == '\r\n':
            return True    # ignore empty line
        # check (and skip) any column numbering
        if "".join(line.split()).isdigit():
            line = f.readline()
            if not line:
                raise FormatSyntaxError('unexpected EOF')
        seqIndex = 0
        while 1:
            if line.isspace():
                break
            field = line.split()
            try:
                seq = self.sequence_list[seqIndex]
            except IndexError:
                raise FormatSyntaxError('more sequences'
                    ' in actual alignment than in header')
            for block in field[1:]:
                seq.append(block)
            line = f.readline()
            if not line:
                # allow for files that don't end in newline
                if self.sequence_list[-1] == seq \
                and len(seq) == int(seq.attrs['MSF length']):
                    return False
                raise FormatSyntaxError('unexpected EOF')
            seqIndex += 1
        return True
