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

from chimerax.atomic import Element

def determine_element_from_mass(mass, *, consider_hydrogens=True):
    H = Element.get_element('H')
    nearest = None
    for high in range(1, Element.NUM_SUPPORTED_ELEMENTS+1):
        if Element.get_element(high).mass > mass:
            break
    else:
        high = Element.NUM_SUPPORTED_ELEMENTS

    if high == 1:
        return H

    if consider_hydrogens:
        max_hyds = 6
    else:
        max_hyds = 0

    for num_hyds in range(max_hyds+1):
        adj_mass = mass - num_hyds * H.mass
        low_mass = Element.get_element(high-1).mass
        while low_mass > adj_mass and high > 1:
            high -= 1
            low_mass = Element.get_element(high-1).mass
        high_mass = Element.get_element(high).mass
        low_diff = abs(adj_mass - low_mass)
        high_diff = abs(adj_mass - high_mass)
        if low_diff < high_diff:
            diff = low_diff
            element = high-1
        else:
            diff = high_diff
            element = high
        if nearest is None or diff < nearest[1]:
            nearest = (element, diff)
    return Element.get_element(nearest[0])

def prep_coords(session, coords_file, input, format_name, *, file_type="coordinates"):
    from chimerax.core.errors import UserError, CancelOperation
    if coords_file is None:
        if session.ui.is_gui and not session.in_script:
            import os
            if isinstance(input, str):
                path = input
            else:
                path = os.path.dirname(os.path.realpath(input.name))
            from Qt.QtWidgets import QFileDialog
            # Don't use a native dialog so that the caption is actually shown;
            # otherwise the dialog is totally mystifying
            coords, types = QFileDialog.getOpenFileName(
                caption=f"Specify {file_type} file for {format_name}",
                directory=os.path.dirname(path), options=QFileDialog.DontUseNativeDialog)
            if not coords:
                raise CancelOperation(f"No coordinates file specified for {format_name}")
            session.logger.info("Coordinates file: %s" % coords)
        else:
            raise UserError("'coords' keyword with coordinate-file argument must be supplied")
    else:
        coords = coords_file
    from chimerax.data_formats import NoFormatError
    try:
        data_fmt = session.data_formats.open_format_from_file_name(coords)
    except NoFormatError as e:
        raise UserError("Cannot determine format of coordinates file '%s' from suffix" % coords)
    return coords, data_fmt
