# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def modelcif_pae(session, structure, metric_name = None, palette = None, range = None,
                 default_score = 100, json_output_path = None):
    '''Read pairwise residue scores from a ModelCIF file and plot them.'''

    from chimerax.alphafold.pae import per_residue_pae
    atom_pae = [r for r in structure.residues if not per_residue_pae(r)]
    if len(atom_pae) > 0:
        rnames = ', '.join(f'{r.name} {r.number}' for r in atom_pae)
        from chimerax.core.errors import UserError
        raise UserError(f'Cannot display PAE data for structures with non-polymer or modified residues.  Structure {structure} has {len(atom_pae)} non-polymer or modified residues: {rnames}.')
        
    matrix = read_pairwise_scores(structure, metric_name = metric_name, default_score = default_score)

    if json_output_path is None:
        import tempfile
        temp = tempfile.NamedTemporaryFile(prefix = 'modelcif_pae_', suffix = '.json')
        json_output_path = temp.name

    write_json_pae_file(json_output_path, matrix)

    # Open PAE plot
    from chimerax.alphafold.pae import alphafold_pae
    alphafold_pae(session, structure = structure, file = json_output_path, palette = palette, range = range)

def read_pairwise_scores(structure, metric_name = 'PAE', default_score = 100):
    if not hasattr(structure, 'filename'):
        from chimerax.core.errors import UserError
        raise UserError(f'Structure {structure} has no associated file')
    mmcif_path = structure.filename

    # fetch data from ModelCIF
    values, metrics = read_ma_qa_metric_local_pairwise_table(structure.session, mmcif_path)

    if len(values) == 0:
        from chimerax.core.errors import UserError
        raise UserError(f'Structure file {mmcif_path} contains no pairwise residue scores (i.e. no table "ma_qa_metric_local_pairwise")')

    # use only the scores with the given metric id.
    metrics_dict = {
        met_id: (met_name, met_type) \
        for met_id, met_name, met_type in metrics
    }
    if metric_name is None:
        met_ids = [met_id for met_id, _, met_type in metrics if met_type == "PAE"]        # look for PAE type
        if len(met_ids) == 0:
            met_ids = [met_id for met_id, _, _ in metrics]
    else:
        met_ids = [met_id for met_id, met_name, _ in metrics if met_name == metric_name]
        if len(met_ids) == 0:
            mnames = ', '.join([f'"{met_name}"' for _, met_name, _ in metrics])
            from chimerax.core.errors import UserError
            raise UserError(f'Structure file {mmcif_path} has no metric with name "{metric_name}", available names are: {mnames}')
    metric_id = met_ids[0] if met_ids else values[0][5]
        
    msg = f"Displaying local-pairwise metric with ID {metric_id} "
    if metric_id in metrics_dict:
        met_name, met_type = metrics_dict[metric_id]
        msg += f"with type '{met_type}' and name '{met_name}'"
    structure.session.logger.info(msg)

    values = [v for v in values if v[5] == metric_id]
    if len(values) == 0:
        from chimerax.core.errors import UserError
        raise UserError(f'Structure file {mmcif_path} has no scores for metric id "{metric_id}"')

    # fill matrix
    matrix_index = {(r.chain_id,r.number):ri for ri,r in enumerate(structure.residues)}

    nr = structure.num_residues
    from numpy import empty, float32
    matrix = empty((nr,nr), float32)
    matrix[:] = default_score
    
    for model_id, chain_id_1, res_num_1, chain_id_2, res_num_2, metric_id, metric_value in values:
        res_num_1, res_num_2, metric_value = int(res_num_1), int(res_num_2), float(metric_value)
        r1 = matrix_index[(chain_id_1, res_num_1)]
        r2 = matrix_index[(chain_id_2, res_num_2)]
        matrix[r1,r2] = metric_value

    return matrix

def read_ma_qa_metric_local_pairwise_table(session, path):
    """Get relevant data from ModelCIF file.
    Returns tuple (values, metrics) with
    - values = list of pairwise metric values stored as tuple
      (model_id, chain_id_1, res_num_1, chain_id_2, res_num_2, metric_id, metric_value)
    - metrics = list of available pairwise metrics stored as tuple
      (metric_id, metric_name, metric_type)
    """
    from chimerax.mmcif import get_cif_tables
    table_names = [
        'ma_qa_metric_local_pairwise',
        'ma_entry_associated_files',
        'ma_associated_archive_file_details',
        'ma_qa_metric'
    ]
    tables = get_cif_tables(path, table_names)
    if len(tables) != 4:
        return None, None

    # check different ways of storing QE
    values = []
    ma_qa_metric_local_pairwise = tables[0]
    ma_entry_associated_files = tables[1]
    ma_associated_archive_file_details = tables[2]
    ma_qa_metric = tables[3]

    # get info on available pairwise metrics
    if ma_qa_metric.num_rows() > 0:
        field_names = ['id', 'mode', 'name', 'type']
        all_metrics = ma_qa_metric.fields(field_names)
        metrics = [
            (metric_id, metric_name, metric_type) \
            for metric_id, metric_mode, metric_name, metric_type in all_metrics \
            if metric_mode == "local-pairwise"
        ]
        # for metric_id, metric_name, metric_type in metrics:
        #     session.logger.info(
        #         f"Available local-pairwise metric with ID {metric_id} " \
        #         f"with type '{metric_type}' and name '{metric_name}'"
        #     )
    else:
        # no metrics here
        metrics = []

    # option 1: it's directly in the file
    if ma_qa_metric_local_pairwise.num_rows() > 0:
        field_names = ['model_id',
                       'label_asym_id_1', 'label_seq_id_1',
                       'label_asym_id_2', 'label_seq_id_2',
                       'metric_id', 'metric_value']
        values.extend(ma_qa_metric_local_pairwise.fields(field_names))
    
    # option 2: it's in ma_entry_associated_files
    associated_files = []
    if ma_entry_associated_files.num_rows() > 0 \
       and ma_entry_associated_files.has_field("file_content"):
        field_names = ['id', 'file_url', 'file_content']
        associated_files = ma_entry_associated_files.fields(field_names)
    qa_files_to_load = [] # do it later
    zip_files = {}
    from os.path import basename, splitext
    file_prefix = splitext(basename(path))[0]
    for file_id, file_url, file_content in associated_files:
        if file_content == "local pairwise QA scores":
            assoc_file_path = fetch_file_url(session, file_url, path, f"{file_prefix}_assoc.cif")
            if assoc_file_path is not None:
                qa_files_to_load.append(assoc_file_path)
        elif file_content == "archive with multiple files":
            zip_files[file_id] = file_url

    # option 3: it's listed in ma_associated_archive_file_details
    associated_qa_files = []
    if ma_associated_archive_file_details.num_rows() > 0 \
       and ma_associated_archive_file_details.has_field("file_content"):
        field_names = ['archive_file_id', 'file_path', 'file_content']
        row_fields = ma_associated_archive_file_details.fields(field_names)
        for archive_file_id, file_path, file_content in row_fields:
            if file_content == "local pairwise QA scores":
                if archive_file_id in zip_files:
                    associated_qa_files.append((zip_files[archive_file_id], file_path))
                else:
                    session.logger.warning(f'Structure file {path} has faulty archive_file_id for {file_path}.')
    for zip_file_url, file_name in associated_qa_files:
        zip_file_path = fetch_file_url(session, zip_file_url, path, f"{file_prefix}_assoc.zip")
        if zip_file_path is not None:
            from os.path import splitext, exists
            zip_dir = splitext(zip_file_path)[0]
            if not exists(zip_dir):
                from os import mkdir
                mkdir(zip_dir)
            import zipfile, os.path
            zip_member_path = os.path.join(zip_dir, file_name)
            if exists(zip_member_path):
                qa_files_to_load.append(zip_member_path)
                continue
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                if file_name in zip_ref.namelist():
                    zip_ref.extract(file_name, zip_dir)
                    qa_files_to_load.append(zip_member_path)
                else:
                    session.logger.warning(f'Could not find {file_name} in ZIP file from {zip_file_url}.')

    # load from associated files
    for assoc_file_path in qa_files_to_load:
        new_values, _ = read_ma_qa_metric_local_pairwise_table(session, assoc_file_path)
        if new_values is not None:
            values.extend(new_values)
    
    return values, metrics

def fetch_file_url(session, file_url, structure_path, save_name):
    """Path can be local file or file from web.
    Returns path to file and flag if file is downloaded temporary file."""
    import os
    dir_name = os.path.dirname(structure_path)
    file_path = os.path.join(dir_name, file_url)
    if os.path.exists(file_path):
        return file_path
    else:
        # try to get from the web
        from chimerax.core.fetch import fetch_file
        try:
            file_path = fetch_file(
                session, file_url,
                name=f'remote associated file for {structure_path}',
                save_name=save_name, save_dir='ModelArchive',
            )
            return file_path
        except:
            session.logger.warning(f"Failed to load {file_url} for {structure_path}")
            return None

def write_json_pae_file(json_output_path, matrix):
    # Write matrix in JSON AlphaFold PAE format
    # {"pae": [[17.14, 18.75, 17.91, ...], [5.32, 8.23, ...], ... ]}
    n = matrix.shape[0]
    dists = ', '.join(('[ ' + ', '.join('%.2f' % matrix[i,j] for j in range(n)) + ' ]')
                      for i in range(n))
    with open(json_output_path, 'w') as file:
        file.write('{"pae": [')
        file.write(dists)
        file.write(']}')

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, FloatArg, ColormapArg, ColormapRangeArg, SaveFileNameArg
    from chimerax.atomic import StructureArg
    desc = CmdDesc(
        required = [('structure', StructureArg)],
        keyword = [('metric_name', StringArg),
                   ('palette', ColormapArg),
                   ('range', ColormapRangeArg),
                   ('default_score', FloatArg),
                   ('json_output_path', SaveFileNameArg)],
        synopsis = 'Plot ModelCIF pairwise residue scores'
    )
    register('modelcif pae', desc, modelcif_pae, logger=logger)
