# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
"""Plug-in interface for format-specific segmentation behavior.

The segmentations bundle is format-agnostic; format-specific decisions (Is this
volume a stored segmentation? What is its reference volume?) are delegated to
handlers registered with the ``segmentation formats`` manager. The DICOM
handler lives in the external DICOM bundle; new formats can be added by
registering more providers.
"""
from abc import ABC, abstractmethod
from typing import Optional

from chimerax.core.toolshed import ProviderManager


class SegmentationFormatHandler(ABC):
    """One handler per data format (DICOM, NIfTI, ...).

    Subclasses override the methods that apply to their format and leave the
    rest as their default no-op implementations.
    """

    @abstractmethod
    def matches(self, volume) -> bool:
        """Return True if this handler can interpret the given Volume."""

    def is_segmentation_volume(self, volume) -> bool:
        """Return True if this volume is itself a stored segmentation."""
        return False

    def find_reference_volume_for(self, session, segmentation):
        """Find the Volume that segmentation should attach to, or None.

        The handler is expected to walk whatever data/model structures it
        knows about (e.g. a DICOM hierarchy) to locate the reference volume.
        """
        return None

    def find_orphans_for(self, session, volume, orphans):
        """Return the subset of orphan segmentations that belong to volume."""
        return []

    def physical_position_label(self, grid_data, axis, plane_index) -> Optional[str]:
        """Return a physical-units position label (e.g. ``"12.50mm"``), or None."""
        return None


class SegmentationFormatManager(ProviderManager):
    """Holds the set of registered SegmentationFormatHandlers."""

    def __init__(self, session, name):
        self.session = session
        self._handlers: list[tuple[str, SegmentationFormatHandler]] = []
        super().__init__(name)

    def add_provider(self, bundle_info, name, **kw):
        if not bundle_info.installed:
            return
        handler = bundle_info.run_provider(self.session, name, self)
        if handler is not None:
            self._handlers.append((name, handler))

    def end_providers(self):
        pass

    def handler_for(self, volume) -> Optional[SegmentationFormatHandler]:
        for _, h in self._handlers:
            try:
                if h.matches(volume):
                    return h
            except Exception:
                continue
        return None

    def is_segmentation_volume(self, volume) -> bool:
        h = self.handler_for(volume)
        return bool(h and h.is_segmentation_volume(volume))

    def find_reference_volume_for(self, segmentation):
        h = self.handler_for(segmentation)
        if h is None:
            return None
        return h.find_reference_volume_for(self.session, segmentation)

    def find_orphans_for(self, volume, orphans):
        h = self.handler_for(volume)
        if h is None:
            return []
        return h.find_orphans_for(self.session, volume, orphans)

    def physical_position_label(self, grid_data, axis, plane_index) -> Optional[str]:
        for _, h in self._handlers:
            label = h.physical_position_label(grid_data, axis, plane_index)
            if label is not None:
                return label
        return None


def get_manager(session) -> Optional[SegmentationFormatManager]:
    """Convenience accessor; returns None if the manager hasn't been initialized."""
    return getattr(session, "segmentation_formats", None)
