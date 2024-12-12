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

def fetch_model_archive(session, ma_identifier, ignore_cache = False, pae = False, in_file_history = True):

    # fetch file
    from chimerax.core.fetch import fetch_file
    file_url = f"https://www.modelarchive.org/doi/10.5452/{ma_identifier}.cif"
    model_name = f'MA {ma_identifier}'
    file_path = fetch_file(
        session, file_url, name=model_name,
        save_name=f"{ma_identifier}.cif", save_dir="ModelArchive",
        ignore_cache = ignore_cache
    )

    # open file
    models, status = session.open_command.open_data(file_path, format = 'mmCIF',
                                                    name = model_name,
                                                    in_file_history = in_file_history)

    if pae:
        # try to add PAE
        from .modelcif_pae import modelcif_pae
        from chimerax.core.errors import UserError
        try:
            modelcif_pae(session, models[0])
        except UserError as e:
            # ok for it not to have PAE...
            session.logger.info(f'Could not open PAE data for {model_name} found: {str(e) }')

    return models, status
