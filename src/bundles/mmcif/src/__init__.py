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

# ensure C++ shared libs are linkable by us
import chimerax.atomic_lib  # noqa
import chimerax.pdb_lib  # noqa

from .mmcif import (  # noqa
    get_cif_tables, get_mmcif_tables, get_mmcif_tables_from_metadata,
    open_mmcif, fetch_mmcif, citations,
    TableMissingFieldsError, CIFTable,
    find_template_residue, load_mmCIF_templates,
    add_citation, add_software,
)
from .corecif import (  # noqa
    open_corecif, fetch_cod, fetch_pcod,
)

from chimerax.core.toolshed import BundleAPI


class _mmCIFioAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            if name == "mmCIF":
                from chimerax.open_command import OpenerInfo

                class Info(OpenerInfo):
                    def open(self, session, data, file_name, **kw):
                        from . import mmcif
                        return mmcif.open_mmcif(session, data, file_name, **kw)

                    @property
                    def open_args(self):
                        from chimerax.core.commands import BoolArg, IntArg
                        return {
                            'atomic': BoolArg,
                            'auto_style': BoolArg,
                            'combine_sym_atoms': BoolArg,
                            'coordsets': BoolArg,
                            'log_info': BoolArg,
                            'max_models': IntArg,
                            'slider': BoolArg,
                        }
            elif name == "ccd":
                from chimerax.open_command import FetcherInfo
                from . import mmcif

                class Info(FetcherInfo):
                    def fetch(self, session, ident, format_name, ignore_cache,
                              fetcher=mmcif.fetch_ccd, **kw):
                        return fetcher(session, ident, ignore_cache=ignore_cache, **kw)
            elif name == "Small Molecule CIF":
                from chimerax.open_command import OpenerInfo
                from . import corecif

                class Info(OpenerInfo):
                    def open(self, session, data, file_name, **kw):
                        return corecif.open_corecif(session, data, file_name, **kw)
            elif name in ("cod", "pcod"):
                from chimerax.open_command import FetcherInfo
                from . import corecif
                fetcher = {
                    "cod": corecif.fetch_cod,
                    "pcod": corecif.fetch_pcod,
                }[name]

                class Info(FetcherInfo):
                    def fetch(self, session, ident, format_name, ignore_cache,
                              fetcher=fetcher, **kw):
                        return fetcher(session, ident, ignore_cache=ignore_cache, **kw)
            else:
                from chimerax.open_command import FetcherInfo
                from . import mmcif
                fetcher = {
                    "pdb": mmcif.fetch_mmcif,
                    "pdbe": mmcif.fetch_mmcif_pdbe,
                    "pdbe_updated": mmcif.fetch_mmcif_pdbe_updated,
                    "pdbj": mmcif.fetch_mmcif_pdbj,
                }[name]

                class Info(FetcherInfo):
                    def fetch(self, session, ident, format_name, ignore_cache,
                              fetcher=fetcher, **kw):
                        return fetcher(session, ident, ignore_cache=ignore_cache, **kw)

                    @property
                    def fetch_args(self):
                        from chimerax.core.commands import BoolArg, FloatArg
                        return {
                            'over_sampling': FloatArg,
                            'structure_factors': BoolArg,
                        }
        elif mgr == session.save_command:
            from chimerax.save_command import SaverInfo

            class Info(SaverInfo):
                def save(self, session, path, **kw):
                    from . import mmcif_write
                    mmcif_write.write_mmcif(session, path, **kw)

                @property
                def save_args(self):
                    from chimerax.core.commands import BoolArg, ModelsArg, ModelArg
                    return {
                        'all_coordsets': BoolArg,
                        'displayed_only': BoolArg,
                        'models': ModelsArg,
                        'rel_model': ModelArg,
                        'selected_only': BoolArg,
                        'fixed_width': BoolArg,
                        'best_guess': BoolArg,
                        'computed_sheets': BoolArg,
                    }

                def save_args_widget(self, session):
                    from .gui import SaveOptionsWidget
                    return SaveOptionsWidget(session)

                def save_args_string_from_widget(self, widget):
                    return widget.options_string()
        else:
            from .build_ui import CCDProvider
            return CCDProvider(session)

        return Info()


bundle_api = _mmCIFioAPI()
