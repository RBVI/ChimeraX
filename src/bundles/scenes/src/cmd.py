# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2025 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# You can also
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

from typing import Optional

from chimerax.core.commands import (
    register, CmdDesc, StringArg, FloatArg, SaveFolderNameArg, SaveFileNameArg,
    ListOf,
)

def register_commands(logger):
    register("scenes save", save_scene_desc, save_scene)
    register("scenes delete", delete_scene_desc, delete_scene)
    register("scenes restore", restore_scene_desc, restore_scene)
    register("scenes rename", rename_scene_desc, rename_scene)
    register("scenes list", list_scenes_desc, list_scenes)
    register("scenes export html", export_html_desc, export_html)
    register("scenes export storyboard", export_storyboard_desc, export_storyboard)


def save_scene(session, scene_name: Optional[str] = None) -> None:
    """Save the current scene as 'scene_name'."""
    session.scenes.save_scene(scene_name)


save_scene_desc = CmdDesc(
    optional=[("scene_name", StringArg)],
    synopsis="Save the current scene as 'scene_name'."
)


def delete_scene(session, scene_name: str):
    """Delete the scene 'scene_name'."""
    session.scenes.delete_scene(scene_name)


delete_scene_desc = CmdDesc(
    required=[("scene_name", StringArg)],
    synopsis="Delete scene 'scene_name'."
)


def restore_scene(session, scene_name):
    """Restore the scene named 'scene_name'."""
    session.scenes.restore_scene(scene_name)


restore_scene_desc = CmdDesc(
    required=[("scene_name", StringArg)],
    synopsis="Restore the scene named 'scene_name'."
)

def rename_scene(session, scene_name, new_scene_name):
    """Rename the scene named 'scene_name' to 'new_scene_name'."""
    session.scenes.rename_scene(scene_name, new_scene_name)


rename_scene_desc = CmdDesc(
    required=[("scene_name", StringArg), ("new_scene_name", StringArg)],
    synopsis="Rename the scene named 'scene_name'."
)


def list_scenes(session):
    """List all saved scenes."""
    for scene_name in session.scenes.scene_names:
        print(scene_name)


list_scenes_desc = CmdDesc(
    synopsis="List all saved scenes."
)


def export_html(session, scene_name, path):
    """Export 'scene_name' to a standalone HTML page with embedded glTF
    geometry. ``path`` may be either a .html file (a sidecar .glb is written
    alongside) or a directory (index.html + scene.glb are created inside)."""
    from .html_export import export_scene_html
    export_scene_html(session, scene_name, path)


export_html_desc = CmdDesc(
    required=[("scene_name", StringArg), ("path", SaveFileNameArg)],
    synopsis="Export a scene to a standalone HTML page (with sidecar glTF)."
)


def export_storyboard(session, path, scenes=None):
    """Export multiple scenes as a clickable storyboard HTML page. Output
    goes into the directory at ``path``: index.html + one .glb per scene.
    If ``scenes`` is omitted, all saved scenes are exported."""
    from .html_export import export_storyboard_html
    export_storyboard_html(session, path, scene_names=scenes)


export_storyboard_desc = CmdDesc(
    required=[("path", SaveFolderNameArg)],
    keyword=[("scenes", ListOf(StringArg))],
    synopsis="Export scenes as a clickable HTML storyboard."
)
