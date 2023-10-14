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

# ensure C++ shared libs we need are linkable by us
import chimerax.atomic_lib  # noqa
import chimerax.pdb_lib  # noqa

from ._pdbio import standard_polymeric_res_names  # this also gets shared lib loaded
from .pdb import open_pdb, save_pdb
from .pdb import process_chem_name, format_nonstd_res_info

from chimerax.core.toolshed import BundleAPI

class _PDBioAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            if name == "PDB":
                from chimerax.open_command import OpenerInfo
                class Info(OpenerInfo):
                    def open(self, session, data, file_name, **kw):
                        from . import pdb
                        return pdb.open_pdb(session, data, file_name, **kw)

                    @property
                    def open_args(self):
                        from chimerax.core.commands import BoolArg, IntArg, FloatArg, EnumOf
                        return {
                            'atomic': BoolArg,
                            'auto_style': BoolArg,
                            'combine_sym_atoms': BoolArg,
                            'coordsets': BoolArg,
                            'log_info': BoolArg,
                            'max_models': IntArg,
                            'segid_chains': BoolArg,
                            'slider': BoolArg,
                            'missing_coordsets': EnumOf(('fill','ignore','renumber')),
                        }
            else:
                from chimerax.open_command import FetcherInfo
                from . import pdb
                fetcher = {
                    'pdb': pdb.fetch_pdb,
                    'pdbe': pdb.fetch_pdb_pdbe,
                    'pdbj': pdb.fetch_pdb_pdbj
                }[name]
                class Info(FetcherInfo):
                    def fetch(self, session, ident, format_name, ignore_cache, fetcher=fetcher, **kw):
                        return fetcher(session, ident, ignore_cache=ignore_cache, **kw)

                    @property
                    def fetch_args(self):
                        from chimerax.core.commands import BoolArg, IntArg, FloatArg
                        return {
                            'over_sampling': FloatArg,
                            'structure_factors': BoolArg,
                        }
        else:
            from chimerax.save_command import SaverInfo
            class Info(SaverInfo):
                def save(self, session, path, **kw):
                    from . import pdb
                    pdb.save_pdb(session, path, **kw)

                @property
                def save_args(self):
                    from chimerax.core.commands import BoolArg, ModelsArg, ModelArg, EnumOf
                    return {
                        'all_coordsets': BoolArg,
                        'displayed_only': BoolArg,
                        'models': ModelsArg,
                        'pqr': BoolArg,
                        'rel_model': ModelArg,
                        'selected_only': BoolArg,
                        'serial_numbering': EnumOf(("amber","h36"))
                    }

                def save_args_widget(self, session):
                    from .gui import SaveOptionsWidget
                    return SaveOptionsWidget(session)

                def save_args_string_from_widget(self, widget):
                    return widget.options_string()

        return Info()


bundle_api = _PDBioAPI()
