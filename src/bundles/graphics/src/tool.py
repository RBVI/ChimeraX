# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from enum import StrEnum
from typing import Dict, Optional, Union

from Qt.QtWidgets import (
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QWidget,
    QSpinBox,
    QAbstractSpinBox,
    QStackedWidget,
    QPlainTextEdit,
    QLineEdit,
    QStackedWidget,
)

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.ui.options import (
    CategorizedSettingsPanel,
    SettingsPanel,
    IntOption,
    BooleanOption,
)
from chimerax.core.commands import run

from chimerax.graphics.settings import (
    get_graphics_settings,
    setting_display_name,
    GraphicsSetting,
    GraphicsSettingCategory,
)


class RenderingOptionsTool(ToolInstance):

    def __init__(self, session):
        self.display_name = "Graphics Options"
        super().__init__(session, self.display_name)
        settings = get_graphics_settings(session)

        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area
        parent.setLayout(QHBoxLayout())

        self.options_widget = CategorizedSettingsPanel(
            help_cb=lambda *, category=None, ses=session, run=run: run(
                ses,
                "help help:user/preferences.html"
                + ("" if category is None else "#" + category.replace(" ", "").lower()),
            )
        )

        check_uniforms = BooleanOption(
            name=setting_display_name(GraphicsSetting.CHECK_SHADER_UNIFORMS),
            default=None,
            attr_name=GraphicsSetting.CHECK_SHADER_UNIFORMS,
            settings=settings,
            callback=self._on_check_uniforms_changed,
        )

        depth_peeling = BooleanOption(
            name=setting_display_name(GraphicsSetting.DEPTH_PEELING),
            default=None,
            attr_name=GraphicsSetting.DEPTH_PEELING,
            settings=settings,
            callback=None,
        )

        depth_peeling_layers = IntOption(
            name=setting_display_name(GraphicsSetting.DEPTH_PEELING_LAYERS),
            default=None,
            attr_name=GraphicsSetting.DEPTH_PEELING_LAYERS,
            settings=settings,
            callback=None,
        )

        show_debug_menu = BooleanOption(
            name=setting_display_name(GraphicsSetting.SHOW_DEBUG_MENU),
            default=None,
            attr_name=GraphicsSetting.SHOW_DEBUG_MENU,
            settings=settings,
            callback=self._on_show_debug_menu_changed,
        )

        vertical_sync = BooleanOption(
            name=setting_display_name(GraphicsSetting.VSYNC),
            default=None,
            attr_name=GraphicsSetting.VSYNC,
            settings=settings,
            callback=self._on_vertical_sync_changed,
            balloon="Synchronize rendering so only one frame is shown per display refresh",
        )

        show_framerate = BooleanOption(
            name=setting_display_name(GraphicsSetting.SHOW_FRAMERATE),
            default=None,
            attr_name=GraphicsSetting.SHOW_FRAMERATE,
            settings=settings,
            callback=self._on_show_framerate_changed,
            balloon="Report the framerate in the status line",
        )

        max_framerate = IntOption(
            name=setting_display_name(GraphicsSetting.MAX_FRAMERATE),
            default=None,
            attr_name=GraphicsSetting.MAX_FRAMERATE,
            settings=settings,
            callback=self._on_max_framerate_changed,
        )

        silhouettes = BooleanOption(
            name=setting_display_name(GraphicsSetting.SILHOUETTES),
            default=None,
            attr_name=GraphicsSetting.SILHOUETTES,
            settings=settings,
            callback=self._on_silhouettes_changed,
        )

        screen_widget, screen_suboptions = self.options_widget.add_option_group(
            GraphicsSettingCategory.GENERAL, group_label="Screen"
        )

        advanced_widget, advanced_suboptions = self.options_widget.add_option_group(
            GraphicsSettingCategory.GENERAL, group_label="Advanced"
        )

        key_light_widget, key_light_suboptions = self.options_widget.add_option_group(
            GraphicsSettingCategory.LIGHTING, group_label="Key Light"
        )

        fill_light_widget, fill_light_suboptions = self.options_widget.add_option_group(
            GraphicsSettingCategory.LIGHTING, group_label="Fill Light"
        )

        ambient_light_widget, ambient_light_suboptions = (
            self.options_widget.add_option_group(
                GraphicsSettingCategory.LIGHTING, group_label="Ambient Light"
            )
        )

        advanced_suboptions.add_option(show_debug_menu)
        advanced_widget.setLayout(QVBoxLayout())
        advanced_widget.layout().addWidget(advanced_suboptions)

        screen_suboptions.add_option(show_framerate)
        screen_suboptions.add_option(vertical_sync)
        screen_suboptions.add_option(max_framerate)
        screen_widget.setLayout(QVBoxLayout())
        screen_widget.layout().addWidget(screen_suboptions)

        self.options_widget.add_option(
            GraphicsSettingCategory.DEBUGGING, check_uniforms
        )

        self.options_widget.add_option(GraphicsSettingCategory.DEPICTION, silhouettes)

        self.options_widget.add_option(
            GraphicsSettingCategory.TRANSPARENCY, depth_peeling
        )

        self.options_widget.add_option(
            GraphicsSettingCategory.TRANSPARENCY, depth_peeling_layers
        )

        parent.layout().addWidget(self.options_widget)

        if not settings.show_debug_menu:
            self.options_widget.hide_tab(GraphicsSettingCategory.DEBUGGING)

        self.tool_window.manage("side")

    def _on_silhouettes_changed(self, silhouettes):
        run(self.session, f"graphics silhouettes {str(silhouettes.value).lower()}")

    def _on_vertical_sync_changed(self, sync):
        run(self.session, f"graphics rate waitForVsync {str(sync.value).lower()}")

    def _on_show_framerate_changed(self, show):
        run(self.session, f"graphics rate {str(show.value).lower()}")

    def _on_max_framerate_changed(self, max):
        run(self.session, f"graphics rate maxFrameRate {max.value}")

    def _on_check_uniforms_changed(self, check):
        # TODO: These strings can change. There should be a command API that calls this
        if check.value:
            run(self.session, "graphics shader checkUniforms true")
        else:
            run(self.session, "graphics shader checkUniforms false")

    def _on_show_debug_menu_changed(self, visibility):
        if visibility.value:
            self.options_widget.show_tab(GraphicsSettingCategory.DEBUGGING)
        else:
            self.options_widget.hide_tab(GraphicsSettingCategory.DEBUGGING)
