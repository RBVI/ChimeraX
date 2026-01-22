# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI

class _MDCrdsBundleAPI(BundleAPI):
    
    from chimerax.atomic import StructureArg

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        # --- Open Command Manager ---
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            
            # Topology / Structure formats
            if name in ("psf", "data", "gro"):
                class MDStructureInfo(OpenerInfo):
                    def open(self, session, data, file_name, *, slider=True, format_name=name, **kw):
                        from .read_structure import read_structure
                        models, status = read_structure(session, data, file_name, format_name=format_name, **kw)
                        
                        # Set up slider if multiple frames were loaded (e.g. from GRO or if coords arg used)
                        if slider and session.ui.is_gui:
                            from chimerax.std_commands.coordset import coordset_slider
                            coordset_slider(session, models)
                        return models, status

                    @property
                    def open_args(self):
                        from chimerax.core.commands import BoolArg, OpenFileNameArg, PositiveIntArg
                        return {
                            'auto_style': BoolArg,
                            'coords': OpenFileNameArg,
                        }
                return MDStructureInfo()
            
            # Trajectory formats
            else:
                class MDTrajectoryInfo(OpenerInfo):
                    def open(self, session, data, file_name, *, structure_model=None,
                            md_type=name, replace=None, slider=True, start=1, step=1, end=None, **kw):
                        
                        # 1. Resolve Structure Model
                        if structure_model is None:
                            from chimerax.core.errors import UserError, CancelOperation
                            from chimerax.atomic import Structure
                            structures = [s for s in session.models if isinstance(s, Structure)]
                            if len(structures) == 0:
                                raise UserError("No atomic models open to read the coordinates into!")
                            elif len(structures) == 1:
                                structure_model = structures[0]
                            else:
                                if session.ui.is_gui and not session.in_script:
                                    # Simple dialog to pick structure
                                    from Qt.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDialogButtonBox
                                    from chimerax.atomic.widgets import StructureMenuButton
                                    
                                    class GetStructureDialog(QDialog):
                                        def __init__(self, session):
                                            super().__init__()
                                            self.setWindowTitle("Choose Structure")
                                            layout = QVBoxLayout(self)
                                            row = QHBoxLayout()
                                            row.addWidget(QLabel("Structure:"))
                                            self.btn = StructureMenuButton(session)
                                            row.addWidget(self.btn)
                                            layout.addLayout(row)
                                            bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                                            bbox.accepted.connect(self.accept)
                                            bbox.rejected.connect(self.reject)
                                            layout.addWidget(bbox)
                                        @property
                                        def model(self): return self.btn.value

                                    dlg = GetStructureDialog(session)
                                    if not dlg.exec() or dlg.model is None:
                                        raise CancelOperation("No structure specified")
                                    structure_model = dlg.model
                                else:
                                    raise UserError("Must specify 'structure_model' to read coordinates into.")

                        # 2. Determine Append vs Replace
                        if replace is None:
                            replace = structure_model.num_coordsets < 2

                        # 3. Read Coordinates
                        from .read_coords import read_coords
                        num_coords = read_coords(session, data, structure_model, md_type,
                            replace=replace, start=start, step=step, end=end)

                        if slider and session.ui.is_gui:
                            from chimerax.std_commands.coordset import coordset_slider
                            coordset_slider(session, [structure_model])

                        action = "Replaced" if replace else "Added"
                        return [], f"{action} {num_coords} frames for {structure_model.name}"

                    @property
                    def open_args(self):
                        from chimerax.atomic import StructureArg
                        from chimerax.core.commands import BoolArg, PositiveIntArg
                        return {
                            'end': PositiveIntArg,
                            'replace': BoolArg,
                            'slider': BoolArg,
                            'start': PositiveIntArg,
                            'step': PositiveIntArg,
                            'structure_model': StructureArg,
                        }
                return MDTrajectoryInfo()

        # --- Save Command Manager ---
        if mgr == session.save_command:
            from chimerax.save_command import SaverInfo
            class MDSaveInfo(SaverInfo):
                def save(self, session, path, *, models=None, **kw):
                    # Legacy write support
                    from chimerax import atomic
                    if models is None:
                        models = atomic.all_structures(session)
                    else:
                        models = [m for m in models if isinstance(m, atomic.Structure)]
                    
                    if not models:
                        from chimerax.core.errors import UserError
                        raise UserError("No models to save.")
                        
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

            return MDSaveInfo()

        # --- MD Plotting Manager (Unchanged) ---
        if kw.get('check_relevance', False):
            return True
            
        if name == "distance":
            a1, a2 = kw['atoms']
            from chimerax.geometry import distance
            values = {}
            for cs_id in kw['structure'].coordset_ids:
                values[cs_id] = distance(a1.get_coordset_coord(cs_id), a2.get_coordset_coord(cs_id))
            return values
            
        elif name == "angle":
            from chimerax.geometry import angle
            values = {}
            for cs_id in kw['structure'].coordset_ids:
                values[cs_id] = angle(*[a.get_coordset_coord(cs_id) for a in kw['atoms']])
            return values
            
        elif name == "torsion":
            from chimerax.geometry import dihedral
            values = {}
            for cs_id in kw['structure'].coordset_ids:
                values[cs_id] = dihedral(*[a.get_coordset_coord(cs_id) for a in kw['atoms']])
            return values
            
        elif name == "surface":
            from .providers import sasa
            return sasa(session, mgr, **kw)
            
        elif name == "rmsd":
            from .providers import rmsd
            return rmsd(session, mgr, **kw)
            
        elif name == "hbonds":
            from .providers import hbonds
            return hbonds(session, mgr, **kw)
            
        raise ValueError("Unknown plotting type: %s" % name)

bundle_api = _MDCrdsBundleAPI()
