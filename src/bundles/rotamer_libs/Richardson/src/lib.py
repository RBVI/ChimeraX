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

_common_cache = {}
_mode_cache = {}

from chimerax.rotamers import RotamerLibrary, RotamerParams, UnsupportedResTypeError, NoResidueRotamersError

class RichardsonBase(RotamerLibrary):

    @property
    def citation(self):
        return """SC Lovell, JM Word, JS Richardson and DC Richardson (2000)
The penultimate rotamer library
Proteins: Structure Function and Genetics 40: 389-408."""

    @property
    def cite_pubmed_id(self):
        return 10861930

    def _get_params(self, res_name, file_name, cis, cache, archive):
        """Richardson zip files have different param layout"""
        try:
            return cache[file_name]
        except KeyError:
            pass
        if res_name not in self.residue_names:
            raise UnsupportedResTypeError(
                "%s library does not support residue type '%s'" % (self.ui_name, res_name))
        if res_name == "PRO":
            if cis:
                file_name = file_name + '-cis'
            else:
                file_name = file_name + '-trans'
        import os.path, inspect
        my_dir = os.path.split(inspect.getfile(self.__class__))[0]
        from zipfile import ZipFile
        zf = ZipFile(os.path.join(my_dir, archive), "r")
        try:
            data = zf.read(file_name)
        except KeyError:
            raise NoResidueRotamersError(
                "'%s' library has no rotamers for '%s'" % (self.ui_name, file_name))
        from struct import unpack, calcsize
        sz1 = calcsize("!ii")
        num_rotamers, num_params, = unpack("!ii", data[:sz1])
        sz2 = calcsize("!f%di" % (num_params-1))
        rotamers = []
        for i in range(num_rotamers):
            params = unpack("!f%di" % (num_params-1), data[sz1 + i * sz2 : sz1 + (i+1) * sz2])
            p = params[0]
            chis = params[1:]
            rotamers.append(RotamerParams(p, chis))
        cache[file_name] = rotamers
        return rotamers

class RichardsonCommonRotamerLibrary(RichardsonBase):

    def rotamer_params(self, res_name, phi, psi, *, cis=False):
        return self._get_params(res_name, res_name, cis, _common_cache, "common.zip")

class RichardsonModeRotamerLibrary(RichardsonBase):

    def rotamer_params(self, res_name, phi, psi, *, cis=False):
        return self._get_params(res_name, res_name, cis, _mode_cache, "mode.zip")
