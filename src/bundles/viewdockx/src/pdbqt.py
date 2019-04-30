# vim: set expandtab shiftwidth=4 softtabstop=4:

def open_pdbqt(session, path, file_name, auto_style, atomic):
    structures = session.models.open(path, format="pdb", log_errors=False)
    with open(path) as f:
        _extract_metadata(session, f, structures)
    status = "Opened %s containing %d structures (%d atoms, %d bonds)" % (
                    file_name, len(structures),
                    sum([s.num_atoms for s in structures]),
                    sum([s.num_bonds for s in structures]))
    return [], status


def _extract_metadata(session, f, structures):
    in_model = False
    model_index = -1
    vina_values = {}
    vina_labels = ["Score", "RMSD l.b.", "RMSD u.b."]
    vina_marker = "VINA RESULT:"
    for line in f:
        record_type = line[:6]
        if record_type == "REMARK":
            # Vina has one "VINA RESULT" record per model
            if in_model and vina_marker in line:
                vina_values.update(zip(vina_labels,
                                       line.split(vina_marker)[1].split()))
        elif record_type == "MODEL ":
            model_index += 1
            in_model = True
        elif record_type == "ENDMDL":
            if vina_values:
                from chimerax.atomic import Structure as SC
                SC.register_attr(session, "viewdockx_data", "ViewDockX")
                structures[model_index].viewdockx_data = vina_values
                vina_values = {}
            in_model = False
