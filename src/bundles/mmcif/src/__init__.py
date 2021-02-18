# vim: set expandtab shiftwidth=4 softtabstop=4:

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
                        }
            elif name == "ccd":
                from chimerax.open_command import FetcherInfo
                from . import mmcif

                class Info(FetcherInfo):
                    def fetch(self, session, ident, format_name, ignore_cache,
                              fetcher=mmcif.fetch_ccd, **kw):
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
        else:
            from chimerax.save_command import SaverInfo

            class Info(SaverInfo):
                def save(self, session, path, **kw):
                    from . import mmcif_write
                    mmcif_write.write_mmcif(session, path, **kw)

                @property
                def save_args(self):
                    from chimerax.core.commands import BoolArg, ModelsArg, ModelArg
                    return {
                        'displayed_only': BoolArg,
                        'models': ModelsArg,
                        'rel_model': ModelArg,
                        'selected_only': BoolArg,
                        'fixed_width': BoolArg,
                        'best_guess': BoolArg,
                    }

                def save_args_widget(self, session):
                    from .gui import SaveOptionsWidget
                    return SaveOptionsWidget(session)

                def save_args_string_from_widget(self, widget):
                    return widget.options_string()

        return Info()


bundle_api = _mmCIFioAPI()
