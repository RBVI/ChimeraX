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

from chimerax.core.state import State
from chimerax.std_commands.view import NamedView
import copy
from abc import ABC, abstractmethod
from typing import Dict, Any
from chimerax.core.objects import all_objects
import inspect
from chimerax.core.models import Model

class UnknownSessionVersion(ValueError):
    pass

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

    1) Ensure that take_snapshot(flags=State.SCENE) returns a dictionary of data that is appropriate for capturing
    Scene state.

    2) Implement restore_scene(data) where data is a pass by
    value dictionary that was returned from take_snapshot(flags=State.SCENE).

    Note:

    - Be aware that being able to call take_snapshot(flags=State.SCENE) does not mean that the model supports
    Scenes. take_snapshot() is a State method, not a Scene interface method. The model MUST also implement
    restore_scene() to be considered Scene supported.

    - It may be wise to make a call to your model's parent take_snapshot() and restore_scene() methods first in
    your implementations to allow generic model property states to be handled by generic model implementations.
    restore_scene() can be called from super() but it is recommended to use scene_super().take_snapshot(State.SCENE)
    instead of super().take_snapshot(State.SCENE) to ensure you are calling a Scene supported parent class.

    - Model positions are saved by the NamedView object and do not need to be saved in Scene interface implementations.

    How to implement Scene support for a static state managing class:

    1) Implement static take_snapshot(obj, session, flags=State.SCENE). obj is the object that the snapshot should be
    taken from, session is the current session, and flags is the State flag that is passed to take_snapshot. The flags
    argument will be State.SCENE when taking a scene snapshot. take_snapshot should return a dictionary of data that is
    needed to capture any scene relevant state data.

    2) Implement static restore_scene(obj, session, data). Obj is the object that the snapshot was taken from, session
    is the current session, and data is the dictionary that was returned from
    take_snapshot(obj, session, flags=State.SCENE). This method is expected to return an object that is assumed to have
    the scene state data applied to it.

    Note: Static state managing classes are not automatically to the scene snapshot system.
    """

    version = 1
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
            self.init_from_data(session, scene_data)

    def init_from_data(self, session, scene_data):
        self.thumbnail = scene_data['thumbnail']
        self.main_view_data = scene_data['main_view_data']
        self.named_view = NamedView.restore_snapshot(session, scene_data['named_view'])
        version = scene_data['version']
        if version == 0:
            # First version didn't retain whether the data came from the derived class or the
            # base Model class; guesstimate that scene implmentation status for the class
            # hasn't changed...
            revised_scene_models = {}
            for model, model_scene_data in scene_data['scene_models'].items():
                if model.restore_scene == Model.restore_scene:
                    revised_scene_models[model] = (False, model_scene_data)
                else:
                    revised_scene_models[model] = (True, model_scene_data)
            self.scene_models = revised_scene_models
        elif version == 1:
            self.scene_models = scene_data['scene_models']
        else:
            # Should not happen because of check in restore_snapshot() unless we forget
            # to add code for restoring a newer version here
            raise NotImplementedError("Support for version %d scenes not implemented" % version)

    def init_from_session(self):
        """
        Initialize from the current session state.
        """
        self.thumbnail = self.take_thumbnail()
        # View State does not inherit from State so we need to get the state managers take_snapshot.
        main_view = self.session.view
        view_state = self.session.snapshot_methods(main_view)
        # Check if the ViewState implements Scenes.
        #self.main_view_data = None
        if implements_scene(view_state):
            self.main_view_data = view_state.take_snapshot(main_view, self.session, State.SCENE)
        # Session Models
        models = self.session.models.list()
        # Create a NamedView object to store camera and model positions. NamedView's are built in to allow future support
        # for interpolating scenes.
        self.named_view = NamedView(self.session.view, self.session.view.center_of_rotation, models)
        # Attr scene_models stores model -> snapshot data mappings.
        self.scene_models = {}
        for model in self.session.models:
            if model.restore_scene == Model.restore_scene:
                model_scene_data = (False, Model.take_snapshot(model, self.session, flags=State.SCENE))
            else:
                model_scene_data = (True, model.__class__.take_snapshot(model, self.session, flags=State.SCENE))
            self.scene_models[model] = model_scene_data

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
            view_state.restore_scene(main_view, self.session, copy.deepcopy(self.main_view_data))
        current_models = self.session.models.list()
        for model in current_models:
            # NamedView only handles restoring model positions. Camera and clip plane positions are restored with the
            # ViewState.
            if model in self.named_view.positions:
                model.positions = self.named_view.positions[model]
            for model, scene_info in self.scene_models.items():
                restore_implemented, scene_data = scene_info
                if restore_implemented:
                    model.restore_scene(scene_data)
                else:
                    Model.restore_scene(model, scene_data)

    def models_removed(self, models: [str]):
        """
        Remove models and associated scene data. This is designed to be attached to a handler for the models removed
        trigger.

        Args:
            models (list of str): List of model identifiers to remove.
        """
        for model in models:
            if model in self.scene_models:
                del self.scene_models[model]
            if model in self.named_view.positions:
                del self.named_view.positions[model]

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
        if data['version'] >= Scene.version:
            raise ValueError("Cannot restore Scene data with version %d" % data['version'])
        return Scene(session, data['name'], scene_data=data)


def md_scene_implementation(obj):
    """
    Find the most derived class in the object's method resolution order that implements restore_scene. This can include
    the object itself. Finds the appropriate class to call take_snapshot(flags=State.SCENE) on when it is unknown
    if the passed object implements Scene support. take_snapshot() is inherited from State so it does not imply Scene
    support.

    Args:
        obj: The object to find the most derived class that implements restore_scene for.

    Returns:
        Object | None: The most derived class that implements restore_scene or None if no class up the inheritance tree
        implements restore_scene.
    """
    for cls in inspect.getmro(type(obj)):
        if implements_scene(cls):
            return cls
    # Default to None if no class at or above obj implements restore_scene
    return None

def scene_super(obj):
    """
    Find the most derived super class of an object that implements restore_scene. DOES NOT include object itself.
    If no superclass of the object implements restore_scene, return None. Use this function to replace super() when
    calling a scene implemented parent take_snapshot(). take_snapshot is inherited from State so it does not
    imply Scene support. super().take_snapshot() does not guarantee calling on a Scene implemented class.
    """
    for cls in inspect.getmro(type(obj))[1:]:
        if implements_scene(cls):
            return cls
    return None

def implements_scene(cls):
    return 'restore_scene' in cls.__dict__ and 'take_snapshot' in cls.__dict__
