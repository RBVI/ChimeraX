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

# -----------------------------------------------------------------------------
# Dialog for making copies of PDB molecule to fill out crystal unit cell.
#
from chimerax.core.tools import ToolInstance
class UnitCellGUI(ToolInstance):

    help = 'help:user/tools/unitcell.html'

    def __init__(self, session, tool_name):

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Make menu to choose atomic structure
        self._structure_menu = sm = self._create_structure_menu(parent)
        layout.addWidget(sm.frame)

        # Unit cell info
        inf = self._create_info_labels(parent)
        layout.addWidget(inf)
        
        # Add buttons
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        # Options panel
        options = self._create_options_pane(parent)
        layout.addWidget(options)

        layout.addStretch(1)    # Extra space at end

        # Show values for shown volume
        self._structure_chosen()
        
        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, UnitCellGUI, 'Unit Cell', create=create)
    
    # ---------------------------------------------------------------------------
    #
    def _create_structure_menu(self, parent):
        from chimerax.atomic import AtomicStructure
        from chimerax.ui.widgets import ModelMenu
        # Do not list unit cell copies in menu.
        model_filter = lambda m: not hasattr(m, '_unit_cell_copy') and hasattr(m, 'metadata') and len(m.metadata) > 0
        m = ModelMenu(self.session, parent, label = 'Molecule',
                      model_types = [AtomicStructure], model_filter = model_filter,
                      model_chosen_cb = self._structure_chosen)
        return m

    # ---------------------------------------------------------------------------
    #
    def _create_info_labels(self, parent):
        from Qt.QtWidgets import QFrame, QLabel
        frame = QFrame(parent)

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(frame)
        
        labels = []
        for text in ('Space group: ',
                     'Cell size: ',
                     'Cell angles: ',
                     'Crystal symmetries in file: ',
                     'Space group symmetries: ',
                     'Non-crystal symmetries in file: '):
            lbl = QLabel(text, frame)
            layout.addWidget(lbl)
            labels.append(lbl)

        (self._space_group,
         self._cell_size,
         self._cell_angles,
         self._smtry_count,
         self._sg_smtry_count,
         self._mtrix_count) = labels

        return frame
    
    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f = button_row(parent,
                       [('Make Copies', self._make_copies),
                        ('Outline', self._show_or_hide_outline),
                        ('Delete Copies', self._delete_copies),
                        ('Options', self._show_or_hide_options),
                        ('Help', self._show_help)],
                       spacing = 10)
        return f
    
    # ---------------------------------------------------------------------------
    #
    def _create_options_pane(self, parent):

        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title = None)
        f = p.content_area

        from chimerax.ui.widgets import EntriesRow
        sf = EntriesRow(f, True, 'Use crystal symmetries from file')
        self._use_smtry_records = sf.values[0]

        ucr = EntriesRow(f, True, 'Use space group if symmetries are missing')
        self._use_cryst1_smtry = ucr.values[0]

        umr = EntriesRow(f, True, 'Use non-crystallographic symmetry')
        self._use_mtrix_records = umr.values[0]

        pam = EntriesRow(f, True, 'Pack molecules in unit cell')
        self._pack_molecules = pam.values[0]

        co = EntriesRow(f, 'Cell origin ', '')
        self._grid_orig = go = co.values[0]
        go.value = '0 0 0'
        go.widget.returnPressed.connect(self._origin_changed)

        nc = EntriesRow(f, 'Number of cells', '', 'offset', '')
        self._ncells = nce = nc.values[0]
        nce.value = '1 1 1'
        nce.widget.returnPressed.connect(self._make_copies)
        self._ocells = oce = nc.values[1]
        oce.value = '0 0 0'
        oce.widget.returnPressed.connect(self._make_copies)

        return p

    # ---------------------------------------------------------------------------
    #
    def _make_copies(self):

        m = self._structure_menu.value
        if m is None:
            return
    
        tflist = self._transforms(m)
        group_model = self._copies_group_model(m)
        _place_molecule_copies(m, group_model, tflist)
        _remove_extra_copies(m, group_model, len(tflist))

    # ---------------------------------------------------------------------------
    #
    def _copies_group_model(self, m, create = True):
        gname = m.name + ' unit cell'
        gm = _find_model_by_name(m.session, gname)
        if gm is None and create:
            from chimerax.core.models import Model
            gm = Model(gname, m.session)
            gm.model_panel_show_expanded = False
            m.session.models.add([gm])
        return gm
    
    # ---------------------------------------------------------------------------
    #
    def _transforms(self, molecule):

        from chimerax import pdb_matrices as pm, crystal

        from chimerax.geometry import Places, Place
        sm = Places([])
        if self._use_smtry_records.enabled:
            sm = pm.crystal_symmetries(molecule, use_space_group_table = False)
        if len(sm) == 0 and self._use_cryst1_smtry.enabled:
            sm = pm.space_group_symmetries(molecule)

        mm = Places()
        if self._use_mtrix_records.enabled:
            mm = pm.noncrystal_symmetries(molecule)

        if len(sm) > 0:
            tflist = sm * mm
        else:
            tflist = mm

        # Adjust transforms so centers of models are in unit cell box
        cp = pm.unit_cell_parameters(molecule)
        uc = cp[:6] if cp else None
        if self._pack_molecules.enabled and uc:
            mc = _molecule_center(molecule)
            tflist = crystal.pack_unit_cell(uc, self._grid_origin(), mc, tflist)

        # Make multiple unit cells
        nc = self._number_of_cells()
        if nc != (1,1,1) and uc:
            # Compute origin.
            oc = tuple((((o+n-1)%n)-(n-1)) for o,n in zip(self._cell_offset(), nc))
            tflist = crystal.unit_cell_translations(uc, oc, nc, tflist)

        return tflist
    
    # ---------------------------------------------------------------------------
    # Find origin of cell in unit cell grid containing specified point.
    #
    def _grid_origin(self):

        try:
            gorigin = [float(s) for s in self._grid_orig.value.split()]
        except ValueError:
            # TODO: should warn about unparsable origin values
            gorigin = (0,0,0)

        if len(gorigin) != 3:
            gorigin = (0,0,0)

        return gorigin

    # ---------------------------------------------------------------------------
    #
    def _number_of_cells(self):

        try:
            nc = tuple(int(s) for s in self._ncells.value.split())
        except ValueError:
            # TODO: should warn about unparsable origin values
            nc = (1,1,1)

        if len(nc) != 3:
            nc = (1,1,1)

        return nc

    # ---------------------------------------------------------------------------
    #
    def _cell_offset(self):

        try:
            oc = tuple(int(s) for s in self._ocells.value.split())
        except ValueError:
            # TODO: should warn about unparsable origin values
            oc = (1,1,1)

        if len(oc) != 3:
            oc = (1,1,1)

        return oc
    
    # ---------------------------------------------------------------------------
    #
    def _origin_changed(self):

        m = self._structure_menu.value
        if m is None:
            return

        om = _find_model_by_name(m.session, self._outline_model_name(m))
        if om:
            self._show_outline_model(m, om)

        if _find_model_by_name(m.session, m.name + ' #2'):
            self._make_copies()
  
    # ---------------------------------------------------------------------------
    #
    def _delete_copies(self):

        m = self._structure_menu.value
        if m is None:
            return

        gm = self._copies_group_model(m, create = False)
        if gm:
            self.session.models.close([gm])

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_outline(self):
        m = self._structure_menu.value
        if m is None:
            return

        name = self._outline_model_name(m)
        om = _find_model_by_name(m.session, name)       # Close outline if shown
        if om:
            self.session.models.close([om])
        else:
            self._show_outline_model(m)

    # ---------------------------------------------------------------------------
    #
    def _show_outline_model(self, m, om = None):
        from chimerax import pdb_matrices as pm, crystal
        cp = pm.unit_cell_parameters(m)
        if cp is None:
            return
        a, b, c, alpha, beta, gamma, space_group, zvalue = cp

        axes = crystal.unit_cell_axes(a, b, c, alpha, beta, gamma)
        mc = _molecule_center(m)
        origin = crystal.cell_origin(self._grid_origin(), axes, mc)
        color = (1,1,1)                     # white

        if om is None:
            name = self._outline_model_name(m)
            s = _new_outline_box(m.session, name, origin, axes, color)
            s.scene_position = m.scene_position
        else:
            _update_outline_box(om, origin, axes, color)

    # ---------------------------------------------------------------------------
    #
    def _outline_model_name(self, molecule):
        return molecule.name + ' unit cell outline'

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

    # ---------------------------------------------------------------------------
    #
    def _structure_chosen(self):
        self._update_info()

    # ---------------------------------------------------------------------------
    #
    def _update_info(self):

        m = self._structure_menu.value
        if m is None:
            return

        from chimerax import pdb_matrices as pm
        cp = pm.unit_cell_parameters(m)
        if cp:
            a, b, c, alpha, beta, gamma, space_group, zvalue = cp
            cs = '%7.3f %7.3f %7.3f' % (a,b,c) if not None in (a,b,c) else ''
            if None in (alpha,beta,gamma):
                ca = ''
            else:
                import math
                radians_to_degrees = 180 / math.pi
                alpha_deg = radians_to_degrees * alpha
                beta_deg = radians_to_degrees * beta
                gamma_deg = radians_to_degrees * gamma
                ca = '%6.2f %6.2f %6.2f' % (alpha_deg,beta_deg,gamma_deg)
            if space_group is None:
                sg = sgsc = ''
            else:
                sg = space_group
                from chimerax import crystal
                sgm = crystal.space_group_matrices(space_group, a, b, c,
                                                   alpha, beta, gamma)
                sgsc = '%d' % len(sgm) if sgm else '0'
        else:
            sg = cs = ca = sgsc = ''

        self._space_group.setText('Space group: ' + sg)
        self._cell_size.setText('Cell size: ' + cs)
        self._cell_angles.setText('Cell angles: ' + ca)
        self._sg_smtry_count.setText('Space group symmetries: ' + sgsc)

        sm = pm.crystal_symmetries(m, use_space_group_table = False)
        self._smtry_count.setText('Crystal symmetries in file: %d' % len(sm))

        mm = pm.noncrystal_symmetries(m, add_identity = False)
        self._mtrix_count.setText('Non-crystal symmetries in file: %d' % len(mm))

    # ---------------------------------------------------------------------------
    #
    def warn(self, message):
        log = self.session.logger
        log.warning(message)
        log.status(message, color='red')

# -----------------------------------------------------------------------------
#
def _place_molecule_copies(m, parent_model, tflist):

    clist = []
    for i,tf in enumerate(tflist):
#        if tf.is_identity():
#            continue
        name = m.name + (' #%d' % (i+1))
        c = _find_model_by_name(m.session, name, parent = parent_model)
        if c is None:
            c = m.copy(name)
            c._unit_cell_copy = m
            _transform_atom_positions(c.atoms, tf)
            clist.append(c)
        else:
            _transform_atom_positions(c.atoms, tf, m.atoms)
        c.scene_position = m.scene_position
    m.session.models.add(clist, parent = parent_model)
  
# -----------------------------------------------------------------------------
# Move atoms in molecule coordinate system using a 3 by 4 matrix.
#
def _transform_atom_positions(atoms, tf, from_atoms = None):
    from_coords = atoms.scene_coords if from_atoms is None else from_atoms.scene_coords
    atoms.scene_coords = tf * from_coords
    
# -----------------------------------------------------------------------------
#
def _remove_extra_copies(m, parent_model, nkeep):

  clist = []
  while True:
    name = m.name + (' #%d' % (len(clist)+nkeep+1))
    c = _find_model_by_name(m.session, name, parent = parent_model)
    if c is None:
      break
    clist.append(c)
  m.session.models.close(clist)

# -----------------------------------------------------------------------------
#
def _find_model_by_name(session, name, parent = None):

  mlist = session.models.list() if parent is None else parent.child_models()
  for m in mlist:
    if m.name == name:
      return m
  return None

# -----------------------------------------------------------------------------
#
def _molecule_center(m):
    return m.atoms.scene_coords.mean(axis = 0)

# -----------------------------------------------------------------------------
#
def _new_outline_box(session, name, origin, axes, rgb):
    from chimerax.core.models import Surface
    surface_model = s = Surface(name, session)
    s.display_style = s.Mesh
    s.use_lighting = False
    s.casts_shadows = False
    s.pickable = False
    s.outline_box = True
    _update_outline_box(s, origin, axes, rgb)
    session.models.add([s])
    return s

# -----------------------------------------------------------------------------
#
def _update_outline_box(surface_model, origin, axes, rgb):

    a0, a1, a2 = axes
    from numpy import array, float32, int32, uint8
    c000 = array(origin, float32)
    c100 = c000 + a0
    c010 = c000 + a1
    c001 = c000 + a2
    c110 = c100 + a1
    c101 = c100 + a2
    c011 = c010 + a2
    c111 = c011 + a0
    va = array((c000, c001, c010, c011, c100, c101, c110, c111), float32)
    ta = array(((0,4,5), (5,1,0), (0,2,6), (6,4,0),
                (0,1,3), (3,2,0), (7,3,1), (1,5,7),
                (7,6,2), (2,3,7), (7,5,4), (4,6,7)), int32)

    b = 2 + 1    # Bit mask, edges are bits 4,2,1
    hide_diagonals = array((b,b,b,b,b,b,b,b,b,b,b,b), uint8)

    # Replace the geometry of the surface
    surface_model.set_geometry(va, None, ta)
    surface_model.edge_mask = hide_diagonals

    rgba = tuple(rgb) + (1,)
    surface_model.color = tuple(int(255*r) for r in rgba)
