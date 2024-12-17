# vim: set expandtab shiftwidth=4 softtabstop=4:
import io

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
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

from chimerax.core.state import State
from chimerax.graphics.gsession import (ViewState, CameraState, LightingState, MaterialState, CameraClipPlaneState,
                                        SceneClipPlaneState)
from chimerax.geometry.psession import PlaceState
from chimerax.std_commands.view import NamedView
import copy
from abc import ABC, abstractmethod
from typing import Dict, Any
from chimerax.core.objects import all_objects
import inspect
from chimerax.core.models import Model


class Scene(State):
    """
    A Scene object is a snapshot of the current state of the session. Scenes save data from ViewState, NamedView,
    and model display and color data. ViewState uses implemented snapshot methods for itself and all nested objects (
    LightingState, MaterialState ect.) to store and restore data in a scene snapshot. NamedView is from
    std_commands.view and has interpolation methods for camera, clipping planes, and model positions. NamedView also
    is stored in scenes using snapshot methods. SceneColors and SceneVisibility are custom data storage containers
    that store {model: data} mappings for color and display data. Scenes can restore session state using the stored
    data. The class also has a static method to check if two scenes are interpolatable. Testing if scenes are
    interpolate-able involves checking that {model: data} mappings in NamedView, SceneColors, and SceneVisibility
    contain the same models. ViewState is a consistent instance across all sessions, so it is implied that ViewState
    is interpolatable always.
    """

    version = 0
    THUMBNAIL_SIZE = (128, 128)

    def __init__(self, session, name, *, scene_data=None):
        """
        Initialize a Scene object. If there is no snapshot scene data passed in then create a new scene from the session
        state. Otherwise, load the scene from the snapshot data.

        Args:
            session: The current session.
            scene_data (dict, optional): A dictionary of scene data to restore from snapshot method.
        """
        self.session = session
        self.name = name
        if scene_data is None:
            # Want a new scene
            self.init_from_session()
        else:
            # load a scene from snapshot
            self.thumbnail = scene_data['thumbnail']
            self.main_view_data = scene_data['main_view_data']
            self.named_view = NamedView.restore_snapshot(session, scene_data['named_view'])
            self.scene_models = scene_data['scene_models']
        return

    def init_from_session(self):
        self.thumbnail = self.take_thumbnail()
        # View State
        main_view = self.session.view
        view_state = self.session.snapshot_methods(main_view)
        self.main_view_data = view_state.take_snapshot(main_view, self.session, State.SCENE)
        # Session Models
        models = self.session.models.list()
        self.named_view = NamedView(self.session.view, self.session.view.center_of_rotation, models)
        self.scene_models = {}
        for model in all_objects(self.session).models:
            scene_implemented_cls = md_scene_implementation(model)
            if hasattr(scene_implemented_cls, 'take_snapshot'):
                self.scene_models[model] = scene_implemented_cls.take_snapshot(model, self.session,
                                                                               flags=State.SCENE)

    def take_thumbnail(self):
        """
        Take a thumbnail of the current session.

        Returns:
            str: The thumbnail image as a base64 encoded string.
        """
        image = self.session.main_view.image(*self.THUMBNAIL_SIZE)
        import io
        img_io = io.BytesIO()
        image.save(img_io, format='JPEG')
        image_bytes = img_io.getvalue()
        import codecs
        image_base64 = codecs.encode(image_bytes, 'base64').decode('utf-8')
        return image_base64

    def restore_scene(self):
        """
        Restore the session state with the data in this scene.
        """
        # Restore the main view
        main_view = self.session.view
        view_state = self.session.snapshot_methods(main_view)
        view_state.restore_scene(self.session, copy.deepcopy(self.main_view_data))
        # Restore model positions. Even though the view state contains camera and clip plane positions, on a restore
        # main view handles restoring the camera and clip plane positions.
        current_models = self.session.models.list()
        for model in current_models:
            if model in self.named_view.positions:
                model.positions = self.named_view.positions[model]
            if model in self.scene_models:
                model.restore_scene(self.scene_models[model])

    def models_removed(self, models: [str]):
        """
        Remove models and associated scene data. This is designed to be attached to a handler for the models removed
        trigger.

        Args:
            models (list of str): List of model identifiers to remove.
        """
        for model in models:
            if model in self.named_view.positions:
                del self.named_view.positions[model]
        return

    def get_name(self):
        return self.name

    def get_thumbnail(self):
        return self.thumbnail

    @staticmethod
    def restore_snapshot(session, data):
        if data['version'] != Scene.version:
            raise ValueError("Cannot restore Scene data with version %d" % data['version'])
        return Scene(session, data['name'], scene_data=data)

    def take_snapshot(self, session, flags):
        return {
            'version': self.version,
            'name': self.name,
            'thumbnail': self.thumbnail,
            'main_view_data': self.main_view_data,
            'named_view': self.named_view.take_snapshot(session, flags),
            'scene_models': self.scene_models
        }


def md_scene_implementation(model: Model):
    """
    Find the most derived class that implements restore_scene. If no class implements restore_scene, return Model.

    Args:
        model (Model): The model to find the most derived class that implements restore_scene for.

    Returns:
        Model: The most derived class that implements restore_scene.
    """
    for cls in inspect.getmro(type(model)):
        if 'restore_scene' in cls.__dict__:
            return cls
    return Model  # Default to Model if no other class implements restore_scene


class SceneRestoreable(ABC):

    @abstractmethod
    def take_scene(self) -> Dict[str, Any]:
        """
        Take a snapshot of the current state of a session object.

        Returns:
            Dict[str, Any]: A dictionary of attributes and their values.
        """
        pass

    @abstractmethod
    def restore_scene(self, scene_data: Dict[str, Any]):
        """
        Restore the model state from the given scene data.

        Args:
            scene_data (Dict[str, Any]): A dictionary of attributes and their values.
        """
        pass
