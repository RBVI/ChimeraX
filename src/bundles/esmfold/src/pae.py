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

def esmfold_pae(session, structure = None, file = None, mgnify_id = None,
                palette = None, range = None, plot = None, divider_lines = None,
                color_domains = False, connect_max_pae = 5, cluster = 0.5, min_size = 10,
                version = None):
    '''Load ESM Metagenomics Atlas predicted aligned error file and show plot or color domains.'''

    if mgnify_id:
        from .database import esmfold_pae_url
        pae_url, file_name = esmfold_pae_url(session, mgnify_id, database_version=version)
        from chimerax.core.fetch import fetch_file
        file = fetch_file(session, pae_url, 'ESM Metagenomics Atlas PAE %s' % mgnify_id,
                          file_name, 'ESMFold',  check_certificates = False, error_status = False)
        
    if file:
        from chimerax.alphafold.pae import AlphaFoldPAE
        pae = AlphaFoldPAE(file, structure)
        pae._plddt_palette = 'esmfold'
        if structure:
            if not pae.reduce_matrix_to_residues_in_structure():
                from chimerax.core.errors import UserError
                raise UserError(f'Structure {structure} does not match PAE matrix size {pae.matrix_size}.'
                                f'The structure has {pae.num_residue_rows} polymer residues and {pae.num_atom_rows} non-polymer atoms'
                                '\n\nThis can happen if chains or atoms were deleted from the AlphaFold model or if the PAE data was applied to a structure that was not the one predicted by AlphaFold.  Use the full-length AlphaFold model to show predicted aligned error.')
            structure.esmfold_pae = pae
    elif structure is None:
        from chimerax.core.errors import UserError
        raise UserError('No structure or PAE file specified.')
    else:
        pae = getattr(structure, 'esmfold_pae', None)
        if pae is None:
            from chimerax.core.errors import UserError
            raise UserError('No predicted aligned error (PAE) data opened for structure #%s'
                            % structure.id_string)

    if plot is None:
        plot = not color_domains	# Plot by default if not coloring domains.
        
    if plot:
        from chimerax.core.colors import colormap_with_range
        colormap = colormap_with_range(palette, range, default_colormap_name = 'pae',
                                       full_range = (0,30))
        p = getattr(structure, '_esmfold_pae_plot', None)
        if p is None or p.closed():
            dividers = True if divider_lines is None else divider_lines
            from chimerax.alphafold.pae import AlphaFoldPAEPlot
            p = AlphaFoldPAEPlot(session, 'ESMFold Predicted Aligned Error', pae,
                                 colormap=colormap, divider_lines=dividers)
            if structure:
                structure._esmfold_pae_plot = p
        else:
            p.display(True)
            if palette is not None or range is not None:
                p.set_colormap(colormap)
            if divider_lines is not None:
                p.show_chain_dividers(divider_lines)

    pae.set_default_domain_clustering(connect_max_pae, cluster)
    if color_domains:
        if structure is None:
            from chimerax.core.errors import UserError
            raise UserError('Must specify structure to color domains.')
        pae.color_domains(connect_max_pae, cluster, min_size)

# -----------------------------------------------------------------------------
#
def register_esmfold_pae_command(logger):
    from chimerax.core.commands import CmdDesc, register, OpenFileNameArg, ColormapArg, ColormapRangeArg, BoolArg, FloatArg, IntArg, StringArg
    from chimerax.atomic import AtomicStructureArg
    desc = CmdDesc(
        optional = [('structure', AtomicStructureArg)],
        keyword = [('file', OpenFileNameArg),
                   ('mgnify_id', StringArg),
                   ('palette', ColormapArg),
                   ('range', ColormapRangeArg),
                   ('plot', BoolArg),
                   ('color_domains', BoolArg),
                   ('connect_max_pae', FloatArg),
                   ('cluster', FloatArg),
                   ('min_size', IntArg),
                   ('version', IntArg)],
        synopsis = 'Show ESMFold predicted aligned error'
    )
    
    register('esmfold pae', desc, esmfold_pae, logger=logger)

# -----------------------------------------------------------------------------
#
from chimerax.alphafold.pae import OpenPredictedAlignedError
class OpenESMFoldPAE(OpenPredictedAlignedError):
    method = 'ESMFold'
    database_key = 'MGnify'
    command = 'esmfold'
    name = 'ESMFold Error Plot'
    help = 'help:user/tools/esmfold.html#pae'

    def is_predicted_model(self, m):
        from .panel import _is_esmfold_model
        return _is_esmfold_model(m)

    def predicted_structure_version(self, structure): 
        return _esmfold_db_structure_version(structure)

    def guess_database_id(self, path):
        return _guess_mgnify_id(path)
        
# ---------------------------------------------------------------------------
#
def _esmfold_db_structure_version(structure):
    '''
    Parse the structure filename to get the ESMFold database version.
    Example database file name MGYP000456789012_v0.pdb
    '''
    if structure is None:
        return None
    path = getattr(structure, 'filename', None)
    if path is None:
        return None
    from os.path import split, splitext
    filename = split(path)[1]
    if filename.startswith('MGYP') and (filename.endswith('.cif') or filename.endswith('.pdb')):
        fields = splitext(filename)[0].split('_')
        if len(fields) > 1 and fields[-1].startswith('v'):
            try:
                version = int(fields[-1][1:])
            except ValueError:
                return None
            return version

    return None

# ---------------------------------------------------------------------------
#
def _guess_mgnify_id(structure_path):
    from os.path import split
    basename = split(structure_path)[1]
    if '_' in basename:
        basename = basename.split('_')[0]
    if basename.startswith('MGYP') and len(basename) == 16:
        return basename
    return None

# -----------------------------------------------------------------------------
#
def esmfold_error_plot_panel(session, create = False):
    return OpenESMFoldPAE.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_esmfold_error_plot_panel(session):
    p = esmfold_error_plot_panel(session, create = True)
    p.display(True)
    return p
