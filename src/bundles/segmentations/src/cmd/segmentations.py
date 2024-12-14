import sys

from typing import Optional, Union, Annotated

from chimerax.core.commands import (
    CmdDesc,
    ModelIdArg,
    EnumOf,
    register,
    run,
    Or,
    IntArg,
    Int2Arg,
    Int3Arg,
    OnOffArg,
)
from chimerax.core.errors import UserError
from chimerax.ui.cmd import ui_tool_show
from chimerax.map import Volume

from chimerax.segmentations.dicom_segmentations import (
    PlanePuckSegmentation,
    SphericalSegmentation,
)

from chimerax.segmentations.segmentation import Segmentation, segment_volume
from chimerax.segmentations.ui.segmentation_mouse_mode import (
    save_mouse_bindings,
    restore_mouse_bindings,
    save_hand_bindings,
    restore_hand_bindings,
)
from chimerax.segmentations.settings import get_settings
from chimerax.segmentations.types import Axis
from chimerax.segmentations.segmentation_tracker import get_tracker
from chimerax.segmentations.ui.segmentation_mouse_mode import (
    mouse_bindings_saved,
    hand_bindings_saved,
)

import chimerax.segmentations.triggers
from chimerax.segmentations.triggers import Trigger

actions = [
    "add",
    "remove",
    "create",
]


def segmentations(
    session,
    action=None,
    modelSpecifier=None,
    axis: Optional[str] = "axial",
    # Axial, Coronal, Sagittal slice segmentations
    center: Optional[
        Union[
            tuple[int, int, int],
            Annotated[list[int], 3],
        ]
    ] = None,
    slice: Optional[int] = None,
    radius: Optional[int] = None,
    minIntensity: Optional[int] = None,
    maxIntensity: Optional[int] = None,
    mouseModes: Optional[bool] = None,
    handModes: Optional[bool] = None,
):
    """Set or restore mouse and hand modes; or create and modify segmentations."""
    settings = get_settings(session)
    tracker = get_tracker()
    if action == "create":
        if not modelSpecifier:
            raise UserError("No model specified")
        reference_model = [
            model for model in session.models if model.id == modelSpecifier
        ][0]
        if not isinstance(reference_model, Volume):
            raise UserError(
                "Must specify a volume to segment; try narrowing your model specifier (e.g. #1 --> #1.1)"
            )
        # TODO The tracker can probably keep track of this without us having to recompute
        # the length every time
        num_preexisting_segmentations = len(
            tracker.segmentations_for_volume(reference_model)
        )
        new_seg = segment_volume(reference_model, num_preexisting_segmentations + 1)
        new_seg.set_parameters(surface_levels=[0.501])
        new_seg.set_step(1)
        new_seg.set_transparency(
            int((settings.default_segmentation_opacity / 100) * 255)
        )
        session.models.add([new_seg])
    elif action in ("add", "remove"):
        if not modelSpecifier:
            raise UserError("No segmentation specified")
        model = [model for model in session.models if model.id == modelSpecifier][0]
        if isinstance(model, Segmentation):
            value = 1 if action == "add" else 0
            if not value and (minIntensity or maxIntensity):
                session.logger.info(
                    "Ignoring the intensity parameters for removing regions from a segmentation"
                )
                minIntensity = maxIntensity = None
            if len(center) < 3 and axis:
                axis = Axis.from_string(axis)
                segment_in_circle(
                    model,
                    axis,
                    slice,
                    center,
                    radius,
                    minIntensity,
                    maxIntensity,
                    value,
                )
            else:
                model_center = model.world_coordinates_for_data_point(center)
                segment_in_sphere(
                    model, model_center, radius, minIntensity, maxIntensity, value
                )
            chimerax.segmentations.triggers.activate_trigger(
                Trigger.SegmentationModified, model
            )
        else:
            raise UserError("Can't operate on a non-segmentation")
    else:
        if mouseModes is not None:
            if mouseModes and not mouse_bindings_saved():
                save_mouse_bindings(session)
                run(session, "ui mousemode shift wheel 'resize segmentation cursor'")
                run(session, "ui mousemode right 'create segmentations'")
                run(session, "ui mousemode shift right 'erase segmentations'")
                run(session, "ui mousemode shift middle 'move segmentation cursor'")
            elif mouseModes and mouse_bindings_saved():
                session.logger.warning(
                    "Mouse bindings already saved; ignoring 'mouseModes true'"
                )
            elif not mouseModes and not mouse_bindings_saved():
                session.logger.warning(
                    "Mouse bindings not saved; ignoring 'mouseModes false'"
                )
            elif not mouseModes and mouse_bindings_saved():
                restore_mouse_bindings(session)
        if handModes is not None:
            if sys.platform != "win32":
                session.logger.warning(
                    "VR is only available on Windows, ignoring handModes and its argument"
                )
                return
            if handModes and not hand_bindings_saved():
                is_vr = save_hand_bindings(session, settings.vr_handedness)
                if is_vr:
                    if settings.vr_handedness == "right":
                        offhand = "left"
                    else:
                        offhand = "right"
                    run(
                        session,
                        f"vr button b 'erase segmentations' hand { str(settings.vr_handedness).lower() }",
                    )
                    run(
                        session,
                        f"vr button a 'create segmentations' hand { str(settings.vr_handedness).lower() }",
                    )
                    run(
                        session,
                        f"vr button x 'toggle segmentation visibility' hand { offhand }",
                    )
                    run(
                        session,
                        f"vr button thumbstick 'resize segmentation cursor' hand { str(settings.vr_handedness).lower() }",
                    )
                    run(
                        session,
                        f"vr button grip 'move segmentation cursor' hand { str(settings.vr_handedness).lower() }",
                    )
                else:
                    session.logger.warning(
                        "Segmentations thinks VR is not on; ignoring request to save hand modes."
                    )
            elif handModes and hand_bindings_saved():
                session.logger.warning(
                    "Hand bindings already saved; ignoring 'handModes true'"
                )
            elif not handModes and not hand_bindings_saved():
                session.logger.warning(
                    "Hand bindings not saved; ignoring 'handModes false'"
                )
            elif not handModes and hand_bindings_saved():
                is_vr = restore_hand_bindings(session)
                if not is_vr:
                    session.logger.warning(
                        "Segmentations thinks VR is not on; ignoring request to restore hand modes."
                    )


def segment_in_sphere(
    model: Segmentation,
    origin: Union[tuple[int, int, int], Annotated[list[int], 3]],
    radius: int,
    minimum_intensity: int,
    maximum_intensity: int,
    value: int = 1,
) -> None:
    segmentation_strategy = SphericalSegmentation(origin, radius, value)
    if value != 0:
        segmentation_strategy.min_threshold = minimum_intensity
        segmentation_strategy.max_threshold = maximum_intensity
    model.segment(segmentation_strategy)


def segment_in_circle(
    model: Segmentation,
    axis,
    slice,
    center,
    radius,
    min_intensity,
    max_intensity,
    value=1,
):
    center_x, center_y = center
    positions = [(center_x, center_y, radius)]
    segmentation_strategy = PlanePuckSegmentation(axis, slice, positions, value)
    if value != 0:
        segmentation_strategy.min_threshold = min_intensity
        segmentation_strategy.max_threshold = max_intensity
    model.segment(segmentation_strategy)


def open_segmentation_tool(session):
    ui_tool_show(session, "segmentations")


def get_segmentation_tool(session):
    from chimerax.segmentations.ui import find_segmentation_tool

    open_segmentation_tool(session)
    tool = find_segmentation_tool(session)
    return tool


segmentations_desc = CmdDesc(
    optional=[("action", EnumOf(actions)), ("modelSpecifier", ModelIdArg)],
    keyword=[
        ("mouseModes", OnOffArg),
        ("handModes", OnOffArg),
        ("axis", EnumOf([str(axis) for axis in [*Axis]])),
        ("center", Or(Int2Arg, Int3Arg)),
        ("slice", IntArg),
        ("radius", IntArg),
        ("minIntensity", IntArg),
        ("maxIntensity", IntArg),
    ],
    synopsis=segmentations.__doc__.split("\n")[0].strip(),
)


def register_seg_cmds(logger):
    register(
        "segmentations",
        segmentations_desc,
        segmentations,
        logger=logger,
    )
