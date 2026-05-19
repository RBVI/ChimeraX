# vim: set expandtab ts=4 sw=4:

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
# VERSION 2.1#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
from collections import defaultdict
from typing import Optional

from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
from chimerax.map import Volume

from chimerax.segmentations.segmentation import Segmentation

import chimerax.segmentations.triggers
from chimerax.segmentations.triggers import Trigger

_tracker = None

# TODO: Is this a StateManager?
class SegmentationTracker:
    def __init__(self):
        self._active_segmentation: Optional[Segmentation] = None
        self._segmentations = defaultdict(set)
        self._unparented_segmentations = set()

    def segmentations_for_volume(self, volume) -> set[Segmentation]:
        # Make the defaultdict register the volume as a key
        try:
            return self._segmentations[volume]
        except KeyError:
            return set()

    def add_segmentation(self, segmentation):
        if segmentation.reference_volume is not None:
            self._segmentations[segmentation.reference_volume].add(segmentation)
        else:
            self._unparented_segmentations.add(segmentation)
        chimerax.segmentations.triggers.activate_trigger(
            Trigger.SegmentationAdded, segmentation
        )

    def remove_segmentation(self, segmentation):
        if segmentation is self.active_segmentation:
            self.active_segmentation = None
        if segmentation.reference_volume is not None:
            self._segmentations[segmentation.reference_volume].remove(segmentation)
        else:
            self._unparented_segmentations.remove(segmentation)
        chimerax.segmentations.triggers.activate_trigger(
            Trigger.SegmentationRemoved, segmentation
        )

    def __delitem__(self, item):
        segmentations = self.segmentations_for_volume(item)
        for segmentation in segmentations:
            segmentation.reference_volume = None
        self._unparented_segmentations |= segmentations
        self._segmentations.__delitem__(item)

    def __contains__(self, item):
        return item in self._segmentations

    @property
    def active_segmentation(self):
        return self._active_segmentation

    @active_segmentation.setter
    def active_segmentation(self, segmentation: Segmentation):
        if (
            segmentation is not None
            and segmentation.reference_volume not in self._segmentations
        ):
            raise ValueError(
                f"Segmentation {segmentation} is not associated with any open volumes."
            )
        if self._active_segmentation:
            self._active_segmentation.active = False
        self._active_segmentation = segmentation
        if self._active_segmentation:
            self._active_segmentation.active = True
        chimerax.segmentations.triggers.activate_trigger(
            Trigger.ActiveSegmentationChanged, segmentation
        )



def on_model_added_to_session(session, _, models):
    tracker = get_tracker()
    from chimerax.segmentations.format_manager import get_manager
    formats = get_manager(session)
    for model in models:
        # Case 1: I've opened a model, then a segmentation from a file
        if isinstance(model, Segmentation):
            if model.reference_volume is not None:
                tracker.add_segmentation(model)
                continue
            ref = formats.find_reference_volume_for(model) if formats else None
            if ref is not None:
                model.reference_volume = ref
                tracker.add_segmentation(model)
            else:
                session.logger.warning(
                    "Added a segmentation without a determinable reference volume; it is unparented for now and will not be shown in the tool but is addressable by the command. You may reparent it manually if desired."
                )
                tracker.add_segmentation(model)
        # Case 2: I've opened a model, see if any orphan segmentations belong to it
        elif isinstance(model, Volume):
            # tap the tracker's dictionary to add this volume as a key
            _ = tracker.segmentations_for_volume(model)
            if not tracker._unparented_segmentations or formats is None:
                continue
            matched = formats.find_orphans_for(
                model, list(tracker._unparented_segmentations)
            )
            for segmentation in matched:
                segmentation.reference_volume = model
                tracker._unparented_segmentations.discard(segmentation)
                tracker.add_segmentation(segmentation)


def on_model_removed_from_session(session, _, models):
    tracker = get_tracker()
    for model in models:
        if isinstance(model, Segmentation):
            tracker.remove_segmentation(model)
        if model in tracker:
            del tracker[model]


def get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = SegmentationTracker()
    return _tracker

def register_model_trigger_handlers(session):
    _ = get_tracker()
    session.triggers.add_handler(
        ADD_MODELS, lambda *args, ses=session: on_model_added_to_session(ses, *args)
    )
    session.triggers.add_handler(
        REMOVE_MODELS,
        lambda *args, ses=session: on_model_removed_from_session(ses, *args),
    )
