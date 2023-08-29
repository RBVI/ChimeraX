# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        from chimerax.open_command import OpenerInfo
        class MaestroOpenerInfo(OpenerInfo):
            def open(self, session, path, file_name, *, show_tool=True, **kw):
                from .io import open_mae
                models, status = open_mae(session, path, file_name, True, True)
                all_models = sum([m.all_models() for m in models], start=[])
                if show_tool and session.ui.is_gui and len(all_models) > 1:
                    for m in all_models:
                        if hasattr(m, 'viewdockx_data'):
                            show_dock = True
                            break
                    else:
                        show_dock = False
                    if show_dock:
                        from Qt.QtCore import QTimer
                        from chimerax.core.commands import run, concise_model_spec
                        QTimer.singleShot(0,
                            lambda *args, run=run, ses=session, spec=concise_model_spec, models=all_models:
                                run(ses, "viewdockx %s" % spec(ses, models)))
                return models, status

            @property
            def open_args(self):
                from chimerax.core.commands import BoolArg
                return { 'show_tool': BoolArg }

        return MaestroOpenerInfo()


bundle_api = _MyAPI()
