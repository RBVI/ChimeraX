# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def hbonds(session, mgr, **kw):
    s = kw['structure']
    atoms = kw['atoms']
    from Qt.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox as qbbox
    dialog = QDialog()
    dialog.setWindowTitle(f"Trajectory H-Bonds for {s}")
    layout = QVBoxLayout()
    dialog.setLayout(layout)

    # H-bonds GUI
    from chimerax.hbonds.gui import HBondsGUI
    hb_gui = HBondsGUI(session, settings_name="MD plotting", inter_model=False, restrict="cross"
        if atoms.selecteds.any() else None, show_inter_model=False, show_intra_model=False, show_log=False,
        show_model_restrict=False, show_save_file=False, show_select=False)
    layout.addWidget(hb_gui)

    # button box
    bbox = qbbox(qbbox.Ok | qbbox.Cancel)
    layout.addWidget(bbox)

    from chimerax.core.errors import CancelOperation
    results = []
    def ok_cb(*, results=results):
        cmd_name, spec, cmd_args = hb_gui.get_command()
        need_sel = len(atoms) != s.num_atoms
        from chimerax.atomic import all_atomic_structures
        if len(all_atomic_structures(session)) > 1:
            spec = s.atomspec + " & sel" if need_sel else s.atomspec
        else:
            spec = "sel" if need_sel else ""
        # running after ok_cb returns doesn't get the dialog to disappear any quicker
        # because we still don't reach the event loop
        from chimerax.core.commands import run
        results.extend(run(session, f"{cmd_name} {spec} {cmd_args}"))
        dialog.accept()
    bbox.accepted.connect(ok_cb)
    bbox.rejected.connect(dialog.reject)

    if dialog.exec() == QDialog.Rejected:
        raise CancelOperation("H-Bond plotting cancelled")

    values = {}
    for cs_id, hbonds in zip(s.coordset_ids, results):
        values[cs_id] = len(hbonds)
    return values

def rmsd(session, mgr, **kw):
    s = kw['structure']
    atoms = kw['atoms']
    ref = kw["ref_frame"]
    with s.suppress_coordset_change_notifications():
        s.active_coordset_id = ref
        ref_coords = atoms.coords
        values = {}
        from chimerax.geometry import align_points
        for i, cs_id in enumerate(s.coordset_ids):
            s.active_coordset_id = cs_id
            xform, rmsd = align_points(atoms.coords, ref_coords)
            values[cs_id] = rmsd
    return values

def sasa(session, mgr, **kw):
    s = kw['structure']
    atoms = kw['atoms']
    categories = set(atoms.structure_categories)
    from chimerax.atomic import Atoms
    full_atom_set = Atoms([a for a in s.atoms if a.structure_category in categories])
    from math import log2
    status_frequency = max(1, 250000 // len(full_atom_set))
    cur_cs_id = s.active_coordset_id
    s.active_coordset_change_notify = False
    values = {}
    # Emulate the behavior of chimerax.surface.measure_sasacmd.measure_sasa, but without the logging
    r = full_atom_set.radii
    r += 1.4
    from chimerax.surface import spheres_surface_area
    try:
        for i, cs_id in enumerate(s.coordset_ids):
            s.active_coordset_id = cs_id
            areas = spheres_surface_area(full_atom_set.coords, r)
            a = areas[full_atom_set.mask(atoms)]
            sarea = a.sum()
            values[cs_id] = sarea
            if (i+1) % status_frequency == 0:
                session.logger.status("Computed SASA for %d of %d frames" % (i+1, s.num_coordsets))
    finally:
        s.active_coordset_id = cur_cs_id
        s.active_coordset_change_notify = True
        session.logger.status("")
    return values
