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

from chimerax.core.state import State
from chimerax.std_commands.view import NamedView
import copy
from abc import ABC, abstractmethod
from typing import Dict, Any
from chimerax.core.objects import all_objects
import inspect
from chimerax.core.models import Model


class Scene(State):
    """
    A Scene object is a snapshot of the current state of the session. Scenes allow a user to save and restore
    significant states of their session. Scenes are different from a Session save state because a Scene does not need to
    restore objects, it only restores the state of the objects which are assumed to already exist.

    When a new Scene is created from a Session, take_snapshot() with the State.SCENE flag is called on the ViewState and
    models that implement the Scene Interface. Scenes also save a NamedView object which stores and can interpolate
    camera and model positions which is relevant for animating Scenes.

    For a model to support the Scene Interface, the model must implement a take_snapshot() (Enforced by State inheritance)
    and restore_scene() method. When a Scene is created take_snapshot() will be called on the model with
    flags=State.SCENE. The model should return a dictionary of data that is needed to restore the model to the state it
    was in when the snapshot was taken. When the Scene is restored, restore_scene() will be called on the model with the
    data that was returned from take_snapshot(flags=State.SCENE).

    How to implement Scene support for a Model derived object:

    1) Make sure that take_snapshot(flags=State.SCENE) returns a dictionary of data that is appropriate for capturing
    Scene state.

    2) Implement restore_scene(data) where data is a pass by
    value dictionary that was returned from take_snapshot(flags=State.SCENE).

    Note:
    - It may be wise to make a call to your model's parent take_snapshot() and restore_scene() methods first in
    your implementations to allow generic model property states to be handled by generic model implementations.
    - Be aware that being able to call take_snapshot(flags=State.SCENE) does not mean that the model supports
    Scenes. The model MUST also implement restore_scene() to be considered Scene supported.
    - Model positions are saved by the NamedView object and do not need to be saved in Scene interface implementations.
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
        """
        Initialize from the current session state.
        """
        self.thumbnail = self.take_thumbnail()
        # View State does not inherit from State so we need to get the state managers take_snapshot.
        main_view = self.session.view
        view_state = self.session.snapshot_methods(main_view)
        # Make sure that the ViewState implements Scenes.
        if implements_scene(view_state):
            self.main_view_data = view_state.take_snapshot(main_view, self.session, State.SCENE)
        # Session Models
        models = self.session.models.list()
        # Create a NamedView object to store camera and model positions. NamedView's are built in to allow future support
        # for interpolating scenes.
        self.named_view = NamedView(self.session.view, self.session.view.center_of_rotation, models)
        # Attr scene_models stores model -> snapshot data mappings.
        self.scene_models = {}
        for model in all_objects(self.session).models:
            scene_implemented_cls = md_scene_implementation(model)
            if scene_implemented_cls is not None and hasattr(scene_implemented_cls, 'take_snapshot'):
                self.scene_models[model] = scene_implemented_cls.take_snapshot(model, self.session,
                                                                               flags=State.SCENE)

    def take_thumbnail(self):
        """
        Take a thumbnail of the current session.

        Returns:
            str: The thumbnail image as a base64 encoded JPEG string.
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
        Restore the session state with the data in this scene. All data is passed to restore_scene implementations
        as deep copies to prevent possibility of altering the scene data.
        """
        # Restore the main view
        main_view = self.session.view
        view_state = self.session.snapshot_methods(main_view)
        if implements_scene(view_state):
            view_state.restore_scene(self.session, copy.deepcopy(self.main_view_data))
        current_models = self.session.models.list()
        for model in current_models:
            # NamedView only handles restoring model positions. Camera and clip plane positions are restored with the
            # ViewState.
            if model in self.named_view.positions:
                model.positions = self.named_view.positions[model]
            if model in self.scene_models:
                model_data_copy = copy.deepcopy(self.scene_models[model])
                model.restore_scene(model_data_copy)

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

    def take_snapshot(self, session, flags):
        return {
            'version': self.version,
            'name': self.name,
            'thumbnail': self.thumbnail,
            'main_view_data': self.main_view_data,
            'named_view': self.named_view.take_snapshot(session, flags),
            'scene_models': self.scene_models
        }

    @staticmethod
    def restore_snapshot(session, data):
        if data['version'] != Scene.version:
            raise ValueError("Cannot restore Scene data with version %d" % data['version'])
        return Scene(session, data['name'], scene_data=data)


def md_scene_implementation(model: Model):
    """
    Find the most derived model subclass that implements restore_scene. If no class implements restore_scene, return
    None. This function is needed because Scenes are not enforced through inheritance so it is necessary to manually
    find the most derived class that implements restore_scene.

    Args:
        model (Model): The model to find the most derived class that implements restore_scene for.

    Returns:
        Model | None: The most derived class that implements restore_scene or None if no class up the inheritance tree
        implements restore_scene.
    """
    for cls in inspect.getmro(type(model)):
        if implements_scene(cls):
            return cls
    # Default to None if no class at or above param model in the Model inheritance tree implements restore_scene
    return None

def implements_scene(cls):
    return 'restore_scene' in cls.__dict__
