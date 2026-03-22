"""
Bundle initialisation for ChimeraX-GraphicsMetal.

Called once when the bundle is loaded.  Registers the Metal backend with
the graphics.View backend registry and, when auto_enable is set, switches
to it immediately.
"""


def init(session, bundle_info):
    from .metal_graphics import is_metal_supported

    if not is_metal_supported():
        session.logger.info(
            "Metal graphics backend not available on this platform."
        )
        return

    # Register persistent settings and the 'graphics metal set' command.
    from .preferences import get_settings, register_commands
    register_commands(session)

    # Register the Metal backend factory with the View backend registry.
    from chimerax.graphics.render_backend import register_backend
    from .metal_graphics import MetalBackend

    register_backend("metal", MetalBackend, session)
    session.logger.info("Metal graphics backend registered.")

    # Register ChimeraX commands.
    from chimerax.core.commands import CmdDesc, BoolArg, StringArg, register

    register(
        "graphics metal",
        CmdDesc(synopsis="Switch to the Metal graphics backend"),
        _cmd_use_metal,
        logger=session.logger,
    )
    register(
        "graphics opengl",
        CmdDesc(synopsis="Switch back to the OpenGL graphics backend"),
        _cmd_use_opengl,
        logger=session.logger,
    )
    register(
        "graphics devices",
        CmdDesc(synopsis="List available Metal GPU devices"),
        _cmd_list_devices,
        logger=session.logger,
    )

    # Auto-enable if the user preference says so.
    prefs = get_settings(session)
    if prefs.use_metal and prefs.auto_enable:
        try:
            _switch_to("metal", session)
            session.logger.info("Auto-switched to Metal graphics backend.")
        except Exception as exc:
            session.logger.warning(
                f"Auto-switch to Metal failed ({exc}); staying on OpenGL."
            )


def finish(session, bundle_info):
    """Restore OpenGL on bundle unload if Metal is still active."""
    from chimerax.graphics.render_backend import active_backend, switch_backend
    backend = active_backend(session)
    if backend and backend.name == "metal":
        switch_backend("opengl", session)
        session.logger.info("Switched back to OpenGL (Metal bundle unloaded).")


# ------------------------------------------------------------------
# Command implementations
# ------------------------------------------------------------------

def _cmd_use_metal(session):
    from .metal_graphics import is_metal_supported
    if not is_metal_supported():
        from chimerax.core.errors import UserError
        raise UserError("Metal is not supported on this system.")
    _switch_to("metal", session)
    session.logger.info("Switched to Metal graphics backend.")


def _cmd_use_opengl(session):
    _switch_to("opengl", session)
    session.logger.info("Switched to OpenGL graphics backend.")


def _cmd_list_devices(session):
    from chimerax.graphics.render_backend import active_backend
    backend = active_backend(session)
    if backend and backend.name == "metal":
        devices = backend.available_devices()
        if devices:
            lines = ["Available Metal GPU devices:"]
            for d in devices:
                tag = " [primary]" if d.get("is_primary") else ""
                mem_mb = d.get("memory_size", 0) // (1024 * 1024)
                unified = " unified" if d.get("unified_memory") else ""
                lines.append(f"  {d['name']}{tag} — {mem_mb} MB{unified}")
            session.logger.info("\n".join(lines))
        else:
            session.logger.info("No Metal devices found.")
    else:
        session.logger.info(
            "Metal backend not active. Use 'graphics metal' first."
        )


def _switch_to(backend_name: str, session):
    from chimerax.graphics.render_backend import switch_backend
    switch_backend(backend_name, session)
