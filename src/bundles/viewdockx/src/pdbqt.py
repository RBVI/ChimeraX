# vim: set expandtab shiftwidth=4 softtabstop=4:

def open_pdbqt(*args):
    encodings = ['utf-8', 'utf-16', 'utf-32']
    for encoding in encodings:
        try:
           return  _open_pdbqt(*args, encoding)
        except UnicodeDecodeError:
            if encoding == encodings[-1]:
                raise

def _open_pdbqt(session, path, file_name, auto_style, atomic, encoding):
    from chimerax.io import open_input, open_output
    from tempfile import TemporaryDirectory
    # clean up columns that foul up PDB reader
    with TemporaryDirectory() as d:
        import os
        prefix = os.path.splitext(os.path.basename(path))[0]
        cleaned_file = os.path.join(d, prefix + ".pdb")
        with open_output(cleaned_file, encoding) as out:
            with open_input(path, encoding) as f:
                for line in f:
                    line = line[:-1]
                    if line.startswith("ATOM "):
                        if len(line) > 78 and line[78].isupper():
                            line = line[:78] + ' ' + line[79:]
                        if len(line) > 70:
                            line = line[:70] + '      ' + line[76:]
                        if line[17:20] == '***':
                            line = line[:17] + 'UNL' + line[20:]
                        if line[25] == '*':
                            line = line[:25] + '1' + line[26:]
                    print(line, file=out)
        structures, _status = session.open_command.open_data(cleaned_file, format="pdb", log_errors=False)
    with open_input(path, encoding='utf-8') as f:
        _extract_metadata(session, f, structures)
    status = "Opened %s containing %d structures (%d atoms, %d bonds)" % (
                    file_name, len(structures),
                    sum([s.num_atoms for s in structures]),
                    sum([s.num_bonds for s in structures]))
    return structures, status


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
