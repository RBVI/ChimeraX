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

from chimerax.core.toolshed import BundleAPI

class _PubChemAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr, *, widget_info=None, **kw):
        if mgr == session.open_command:
            from chimerax.open_command import FetcherInfo
            class PubchemFetcherInfo(FetcherInfo):
                def fetch(self, session, ident, format_name, ignore_cache, **kw):
                    from . import pubchem
                    return pubchem.fetch_pubchem(session, ident, ignore_cache=ignore_cache, **kw)
            return PubchemFetcherInfo()
        elif widget_info is not None:
            widget, fill = widget_info
            if fill:
                # fill parameters widget
                from .build_ui import fill_widget
                fill_widget(widget)
            else:
                # process parameters widget to generate provider command (sub)string;
                # will not be called if 'indirect' was specified as 'true' in the provider info
                # (e.g. it links to another tool/interface);
                # if 'new_model_only' in the provider info was 'true' then the returned string
                # should be the _entire_ command for opening the model, not a substring of the
                # 'build' command
                from .build_ui import process_widget
                return process_widget(widget)

bundle_api = _PubChemAPI()
