# vim: set expandtab softtabstop=4 shiftwidth=4:
from abc import abstractmethod

from chimerax.core.models import Model

from chimerax.map.volume import (
    Volume,
    any_volume_open,
    show_volume_dialog,
    default_settings,
    set_data_cache,
    MapChannelsModel,
    MultiChannelSeries,
    set_initial_volume_color,
    set_initial_region_and_style,
    data_already_opened,
    volume_from_grid_data,
    _reset_color_sequence,
)
from chimerax.save_command import SaverInfo


class SegmentationStrategy:
    """Implements an algorithm for segmentation"""

    @abstractmethod
    def execute(self, grid, reference_grid):
        pass


class Segmentation(Volume):
    """A segmentation is a Volume that's based on another volume"""

    def __init__(self, session, grid_data, rendering_options=None):
        Volume.__init__(self, session, grid_data, rendering_options=rendering_options)
        self.active = False
        self.reference_volume = None

    def copy(self):
        v = segmentation_from_grid_data(
            self.data, self.session, open_model=False, style=None, show_dialog=False
        )
        v.copy_settings_from(self)
        return v

    # TODO: Should probably be upstream in Volume
    def set_step(self, step):
        self.new_region(ijk_step=[step, step, step], adjust_step=False)

    def segment(self, strategy: SegmentationStrategy):
        strategy.execute(self.data, self.reference_volume.data)
        self.data.values_changed()

    def save(self, path, saver: SaverInfo):
        # TODO: Saver could be one of the Manager-Provider interfaces
        saver.save(self.session, path)

    def take_snapshot(self, session, flags):
        data = super().take_snapshot(session, flags)
        data["reference_volume"] = self.reference_volume
        data["active"] = self.active
        return data

    @staticmethod
    def restore_snapshot(session, data):
        grid_data = data["grid data state"].grid_data
        if grid_data is None:
            return
        dv = Segmentation(session, grid_data)
        Model.set_state_from_snapshot(dv, session, data["model state"])
        dv._style_when_shown = None
        dv.reference_volume = data["reference_volume"]
        dv.active = data["active"]
        from chimerax.map.session import set_map_state
        from chimerax.map.volume import show_volume_dialog

        set_map_state(data["volume state"], dv)
        dv._drawings_need_update()
        show_volume_dialog(session)
        return dv


def segment_volume(volume, number: int) -> Segmentation:
    """Segment the Volume and return an object of type Segmentation. The caller is responsible
    for adding the segmentation to the session."""
    from chimerax.map_data.arraygrid import ArrayGridData
    from numpy import zeros

    new_grid_array = zeros(volume.data.size[::-1])
    new_grid = ArrayGridData(
        array=new_grid_array,
        origin=volume.data.origin,
        step=volume.data.step,
        cell_angles=volume.data.cell_angles,
        rotation=volume.data.rotation,
        symmetries=volume.data.symmetries,
        name="segmentation of %s (#%d)" % (volume.name, number),
    )
    new_grid.initial_plane_display = False
    new_seg_model = open_grids_as_segmentation(
        volume.session, [new_grid], name="new segmentation"
    )[0][0]
    new_seg_model.reference_volume = volume
    return new_seg_model


# Volume.copy could accomplish this if the open_model parameter was hoisted from
# volume_from_grid_data
def copy_volume_for_auxiliary_display(volume):
    """Copies a volume but does not add it to the session so that the volume can
    be displayed in views other than the 3D view and those views can take control
    of its presentation."""
    v = volume_from_grid_data(
        volume.data, volume.session, style=None, show_dialog=False, open_model=False
    )
    v.copy_settings_from(volume)
    return v


# TODO: Maybe unncessary?
def segmentation_from_grid_data(
    grid_data, session, style="auto", open_model=True, model_id=None, show_dialog=True
):
    """
    Supported API.
    Create a new :class:`.Volume` model from a :class:`~.data.GridData` instance and set its initial
    display style and color and add it to the session open models.

    Parameters
    ----------
    grid_data : :class:`~.data.GridData`
      Use this GridData to create the Volume.
    session : :class:`~chimerax.core.session.Session`
      The session that the Volume will belong to.
    style : 'auto', 'surface', 'mesh' or 'image'
      The initial display style.
    open_model : bool
      Whether to add the Volume to the session open models.
    model_id : tuple of integers
      Model id for the newly created Volume.
      It is an error if the specifid id equals the id of an existing model.
    show_dialog : bool
      Whether to show the Volume Viewer user interface panel.

    Returns
    -------
    volume : the created :class:`.Volume`
    """

    set_data_cache(grid_data, session)

    ds = default_settings(session)
    ro = ds.rendering_option_defaults()
    if getattr(grid_data, "polar_values", None):
        ro.flip_normals = True
        ro.cap_faces = False
    if hasattr(grid_data, "initial_rendering_options"):
        for oname, ovalue in grid_data.initial_rendering_options.items():
            setattr(ro, oname, ovalue)

    # Create volume model
    d = data_already_opened(grid_data.path, grid_data.grid_id, session)
    if d:
        grid_data = d

    v = Segmentation(session, grid_data, rendering_options=ro)

    # Set display style
    if style == "auto":
        # Show single plane data in image style.
        single_plane = [s for s in grid_data.size if s == 1]
        style = "image" if single_plane else "surface"
    if style is not None:
        v._style_when_shown = style

    if grid_data.rgba is None:
        if not any_volume_open(session):
            _reset_color_sequence(session)
        set_initial_volume_color(v, session)

    if model_id is not None:
        if session.models.have_id(model_id):
            from chimerax.core.errors import UserError

            raise UserError(
                "Tried to create model #%s which already exists"
                % ".".join("%d" % i for i in model_id)
            )

        v.id = model_id

    if open_model:
        session.models.add([v])

    if show_dialog:
        show_volume_dialog(session)

    return v


# There's really zero difference between this function, and volume_from_grid_data, and the
# previously extant dicom_volume_from_grid_data, except the opening function that gets called.
# Here, it's segmentation_from_grid_data, there it's volume_from_grid_data. Volumes and Segmentations
# should have a from_grid_data method instead, and the GridData function should be refactored to take
# a class whose constructor it will call.
def open_grids_as_segmentation(session, grids, name, **kw):
    level = kw.get("initial_surface_level", None)
    if level is not None:
        for g in grids:
            g.initial_surface_level = level

    if kw.get("polar_values", False) or kw.get("difference", False):
        for g in grids:
            g.polar_values = True
        if g.rgba is None:
            g.rgba = (0, 1, 0, 1)  # Green

    channel = kw.get("channel", None)
    if channel is not None:
        for g in grids:
            g.channel = channel

    series = kw.get("vseries", None)
    if series is not None:
        if series:
            for i, g in enumerate(grids):
                if tuple(g.size) != tuple(grids[0].size):
                    gsizes = "\n".join((g.name + (" %d %d %d" % g.size)) for g in grids)
                    from chimerax.core.errors import UserError

                    raise UserError(
                        "Cannot make series from volumes with different sizes:\n%s"
                        % gsizes
                    )
                g.series_index = i
        else:
            for g in grids:
                if hasattr(g, "series_index"):
                    delattr(g, "series_index")

    maps = []
    if "show" in kw:
        show = kw["show"]
    else:
        show = len(grids) >= 1 and getattr(grids[0], "show_on_open", True)
    si = [d.series_index for d in grids if hasattr(d, "series_index")]
    is_series = len(si) == len(grids) and len(set(si)) > 1
    cn = [d.channel for d in grids if d.channel is not None]
    is_multichannel = len(cn) == len(grids) and len(set(cn)) > 1
    for d in grids:
        show_data = show
        if is_series or is_multichannel:
            show_data = (
                False  # MapSeries or MapChannelsModel classes will decide which to show
            )
        vkw = {"show_dialog": False}
        if hasattr(d, "initial_style") and d.initial_style in (
            "surface",
            "mesh",
            "image",
        ):
            vkw["style"] = d.initial_style
        v = segmentation_from_grid_data(d, session, open_model=False, **vkw)
        maps.append(v)
        if not show_data:
            v.display = False
        set_initial_region_and_style(v)

    show_dialog = kw.get("show_dialog", True)
    if maps and show_dialog:
        show_volume_dialog(session)

    msg = ""
    if is_series and is_multichannel:
        cmaps = {}
        for m in maps:
            cmaps.setdefault(m.data.channel, []).append(m)
        if len(set(len(cm) for cm in cmaps.values())) > 1:
            session.logger.warning(
                "Map channels have differing numbers of series maps: %s"
                % ", ".join("%d (%d)" % (c, cm) for c, cm in cmaps.items())
            )
        from chimerax.map_series import MapSeries

        ms = [MapSeries("channel %d" % c, cm, session) for c, cm in cmaps.items()]
        mc = MultiChannelSeries(name, ms, session)
        models = [mc]
    elif is_series:
        from chimerax.map_series import MapSeries

        ms = MapSeries(name, maps, session)
        ms.display = show
        models = [ms]
    elif is_multichannel:
        mc = MapChannelsModel(name, maps, session)
        mc.display = show
        mc.show_n_channels(3)
        models = [mc]
    elif len(maps) == 0:
        msg = "No map data opened"
        session.logger.warning(msg)
        models = maps
    else:
        models = maps

    # Create surfaces before adding to session so that initial view can use corrrect bounds.
    for v in maps:
        if v.display:
            v.update_drawings()

    return models, msg
