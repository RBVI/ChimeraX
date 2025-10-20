from chimerax.core.settings import Settings


class _MCPServerSettings(Settings):
    EXPLICIT_SAVE = {
        "auto_start": False,
        "port": 3001,
    }

    AUTO_SAVE = {}


# Global settings instance
_settings = None


def get_settings(session):
    """Get the MCP server settings instance"""
    global _settings
    if _settings is None:
        _settings = _MCPServerSettings(session, "mcp_server")
    return _settings


def register_settings_options(session):
    """Register MCP server settings in ChimeraX main settings menu"""
    from chimerax.ui.options import BooleanOption, IntOption

    settings = get_settings(session)

    settings_info = {
        "auto_start": (
            "Auto-start MCP server",
            BooleanOption,
            "Automatically start the MCP server when ChimeraX launches",
        ),
        "port": (
            "Default MCP server port",
            IntOption,
            "Default port number for the MCP server (requires restart to take effect)",
        ),
    }

    for setting, setting_info in settings_info.items():
        opt_name, opt_class, balloon = setting_info
        if isinstance(opt_class, tuple):
            opt_class, kw = opt_class
        else:
            kw = {}
        opt = opt_class(
            opt_name,
            getattr(settings, setting),
            None,
            attr_name=setting,
            settings=settings,
            balloon=balloon,
            **kw,
        )
        session.ui.main_window.add_settings_option("MCP Server", opt)

