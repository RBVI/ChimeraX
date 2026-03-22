"""
Persistent settings for the Metal graphics backend.
"""

from chimerax.core.settings import Settings


class MetalSettings(Settings):
    """Persistent Metal backend preferences stored in ChimeraX user config."""

    EXPLICIT_SAVE = {
        "use_metal": True,
        "auto_enable": True,
        # Device selection: empty = system default (always correct on Apple Silicon)
        "presentation_device": "",
        # Compute offload: empty = disabled
        "compute_offload_device": "",
        "enable_ray_tracing": False,
        "enable_argument_buffers": True,
    }

    # Defaults
    use_metal = True
    auto_enable = True
    presentation_device = ""
    compute_offload_device = ""
    enable_ray_tracing = False
    enable_argument_buffers = True


_settings_instance = None


def get_settings(session) -> MetalSettings:
    """Return (creating if needed) the singleton MetalSettings object."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = MetalSettings(session, "graphics_metal")
    return _settings_instance


def register_commands(session):
    """Register the 'graphics metal set' command."""
    from chimerax.core.commands import CmdDesc, BoolArg, StringArg, register

    desc = CmdDesc(
        keyword=[
            ("use_metal", BoolArg),
            ("auto_enable", BoolArg),
            ("presentation_device", StringArg),
            ("compute_offload_device", StringArg),
            ("enable_ray_tracing", BoolArg),
            ("enable_argument_buffers", BoolArg),
        ],
        synopsis="Configure the Metal graphics backend",
    )
    register("graphics metal set", desc, _set_metal_cmd, logger=session.logger)


def _set_metal_cmd(
    session,
    use_metal=None,
    auto_enable=None,
    presentation_device=None,
    compute_offload_device=None,
    enable_ray_tracing=None,
    enable_argument_buffers=None,
):
    prefs = get_settings(session)

    if use_metal is not None:
        prefs.use_metal = use_metal
    if auto_enable is not None:
        prefs.auto_enable = auto_enable
    if presentation_device is not None:
        prefs.presentation_device = presentation_device
    if compute_offload_device is not None:
        prefs.compute_offload_device = compute_offload_device
    if enable_ray_tracing is not None:
        prefs.enable_ray_tracing = enable_ray_tracing
    if enable_argument_buffers is not None:
        prefs.enable_argument_buffers = enable_argument_buffers

    # Apply device changes live if Metal is active.
    from chimerax.graphics.render_backend import active_backend
    backend = active_backend(session)
    if backend and backend.name == "metal":
        if presentation_device is not None and hasattr(backend, '_multi_gpu') \
                and backend._multi_gpu is not None:
            backend._multi_gpu.selectPresentationDevice(presentation_device)
        if compute_offload_device is not None:
            backend.select_compute_device(compute_offload_device)

    session.logger.info(
        "Metal settings updated:\n"
        f"  use_metal={prefs.use_metal}\n"
        f"  auto_enable={prefs.auto_enable}\n"
        f"  presentation_device='{prefs.presentation_device}'\n"
        f"  compute_offload_device='{prefs.compute_offload_device}'\n"
        f"  enable_ray_tracing={prefs.enable_ray_tracing}\n"
        f"  enable_argument_buffers={prefs.enable_argument_buffers}"
    )
