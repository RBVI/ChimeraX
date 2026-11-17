# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California. All rights reserved.
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


"""Restore sessions saved when SNFG symbols lived in their own models."""

from chimerax.core.models import Model, Surface

from .definitions import DEFAULT_SIZE


class SNFGModel(Model):
    """Placeholder that moves its symbols into the structure, then closes."""

    SESSION_SAVE = False

    @classmethod
    def restore_snapshot(cls, session, data):
        model = cls("SNFG symbols", session)
        Model.set_state_from_snapshot(model, session, data)
        structure = data.get('structure')
        shapes = data.get('shapes_drawing') or {}
        residues = shapes.get('shape_residues', [])
        sizes = shapes.get('shape_sizes', [])
        if len(sizes) != len(residues):
            sizes = [DEFAULT_SIZE] * len(residues)
        hidden = [r for r in data.get('hidden_residues', []) if r is not None]
        state = {
            'sizes': {r: s for r, s in zip(residues, sizes) if r is not None},
            'show_atoms': not hidden,
            'hidden': hidden,
        }

        def convert(trigger_name, trigger_data):
            from chimerax.core.triggerset import DEREGISTER
            from .snfg import _manager
            if structure is not None and not structure.deleted:
                _manager(session)._restore(structure, state)
            if not model.deleted:
                session.models.close([model])
            return DEREGISTER
        session.triggers.add_handler("end restore session", convert)
        return model


class SNFGConnectionsDrawing(Surface):
    """Closed along with its parent SNFGModel."""

    SESSION_SAVE = False

    @classmethod
    def restore_snapshot(cls, session, data):
        m = cls(data.get('name', 'SNFG connections'), session)
        m.set_state_from_snapshot(session, data)
        return m
