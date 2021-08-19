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

class _MDCrdsBundleAPI(BundleAPI):
    
    from chimerax.atomic import StructureArg

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            if name == "gro":
                class MDInfo(OpenerInfo):
                    def open(self, session, data, file_name, **kw):
                        from .read_gro import read_gro
                        return read_gro(session, data, file_name, **kw)
                    @property
                    def open_args(self):
                        from chimerax.core.commands import BoolArg
                        return { 'auto_style': BoolArg }
            else:
                class MDInfo(OpenerInfo):
                    def open(self, session, data, file_name, *, structure_model=None,
                            md_type=name, replace=True, **kw):
                        if structure_model is None:
                            from chimerax.core.errors import UserError
                            raise UserError("Must specify a structure model to read the"
                                " coordinates into")
                        from .read_coords import read_coords
                        num_coords = read_coords(session, data, structure_model, md_type,
                            replace=replace)
                        if replace:
                            return [], "Replaced existing frames of %s with  %d new frames" \
                                % (structure_model, num_coords)
                        return [], "Added %d frames to %s" % (num_coords, structure_model)

                    @property
                    def open_args(self):
                        from chimerax.atomic import StructureArg
                        from chimerax.core.commands import BoolArg
                        return {
                            'structure_model': StructureArg,
                            'replace': BoolArg
                        }
        else:
            from chimerax.save_command import SaverInfo
            class MDInfo(SaverInfo):
                def save(self, session, path, *, models=None, **kw):
                    from chimerax import atomic
                    if models is None:
                        models = atomic.all_structures(session)
                    else:
                        models = [m for m in models if isinstance(m, atomic.Structure)]
                    if len(models) == 0:
                        from chimerax.core.errors import UserError
                        raise UserError("Must specify models to write DCD coordinates")
                    # Check that all models have same number of atoms.
                    nas = set(m.num_atoms for m in models)
                    if len(nas) != 1:
                        from chimerax.core.errors import UserError
                        raise UserError("Models have different number of atoms")
                    from .write_coords import write_coords
                    write_coords(session, path, "dcd", models)

                @property
                def save_args(self):
                    from chimerax.core.commands import ModelsArg
                    return { 'models': ModelsArg }

                def save_args_widget(self, session):
                    from .gui import SaveOptionsWidget
                    return SaveOptionsWidget(session)

                def save_args_string_from_widget(self, widget):
                    return widget.options_string()

        return MDInfo()

bundle_api = _MDCrdsBundleAPI()
