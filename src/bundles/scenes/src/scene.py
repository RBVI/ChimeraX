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
            self.scene_restoreables = scene_data['scene_restorables']
        return

    def init_from_session(self):
        self.thumbnail = self.take_thumbnail()
        self.main_view_data = self.create_main_view_data()
        models = self.session.models.list()
        self.named_view = NamedView(self.session.view, self.session.view.center_of_rotation, models)
        self.scene_restoreables = {}
        for model in all_objects(self.session).models:
            scene_implemented_cls = md_scene_implementation(model)
            if hasattr(scene_implemented_cls, 'take_snapshot'):
                self.scene_restoreables[model] = scene_implemented_cls.take_snapshot(model, self.session, flags=State.SCENE)

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
        self.restore_main_view_data(self.main_view_data)
        # Restore model positions. Even though the view state contains camera and clip plane positions, on a restore
        # main view handles restoring the camera and clip plane positions.
        current_models = self.session.models.list()
        for model in current_models:
            if model in self.named_view.positions:
                model.positions = self.named_view.positions[model]
            if model in self.scene_restoreables:
                model.restore_scene(self.scene_restoreables[model])

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

    def create_main_view_data(self):
        """
        Build a data tree to represent the data in ViewState. This data is used for saving snapshot of the scene.
        This method first calls take_snapshot on the ViewState. By default, that snapshot will contain references to
        objects but not the actual object data itself. This method steps into each of the nested objects in the
        ViewState attributes and replaces the object references in the ViewState take snapshot with the raw data from
        each object. Most attributes of the sessions main view are handled by one State manager class, so they can be
        nested as raw data and converted back knowing that all the data is converted using one State manager. The
        exception is clip planes attr (a list of clip planes) which are handled by two different State manager
        classes. Therefore, the clip planes are stored as a dictionary with the key being the type of clip plane and
        the value being the raw data of the clip plane so that when it is time to restore the scene, the correct
        State manager can be used to restore the data entry.

        Returns:
            dict: A dictionary containing the main view data.
        """

        main_view = self.session.view

        view_state = self.session.snapshot_methods(main_view)
        data = view_state.take_snapshot(main_view, self.session, State.SCENE)

        # By default, ViewState take_snapshot uses class name references and uids to store data for object attrs stored
        # in the View. For the simplicity of Scenes we want to convert all the nested objects into raw data.
        v_camera = main_view.camera
        data['camera'] = CameraState.take_snapshot(v_camera, self.session, State.SCENE)
        c_position = v_camera.position
        data['camera']['position'] = PlaceState.take_snapshot(c_position, self.session, State.SCENE)

        v_lighting = main_view.lighting
        data['lighting'] = LightingState.take_snapshot(v_lighting, self.session, State.SCENE)

        v_material = main_view.material
        data['material'] = MaterialState.take_snapshot(v_material, self.session, State.SCENE)

        # 'clip_planes in data is an array of clip planes objects. The clip plane objects can be either scene or camera
        # clip planes. Need to convert them into raw data before storing them in the scene, but also need to keep track
        # of which state class the data was derived from, so it can be restored. Replace the 'clip_planes' key in data
        # with the raw data map.
        clip_planes = data['clip_planes']
        clip_planes_data = {}
        for clip_pane in clip_planes:
            cp_state_manager = self.session.snapshot_methods(clip_pane)
            if cp_state_manager == CameraClipPlaneState:
                clip_planes_data["camera"] = CameraClipPlaneState.take_snapshot(clip_pane, self.session, State.SCENE)
            if cp_state_manager == SceneClipPlaneState:
                clip_planes_data["scene"] = SceneClipPlaneState.take_snapshot(clip_pane, self.session, State.SCENE)

        data['clip_planes'] = clip_planes_data

        return data

    def restore_main_view_data(self, data):
        """
        Restore the main view data using ViewState's restore snapshot method to restore session state. ViewState
        restore_snapshot method expects certain nested values to be objects, not raw data. This method converts all
        the primitive data that represents nested objects formatted by the scene, back into the appropriate objects
        using the appropriate state managers.

        Args:
            data (dict): A dictionary containing the main view data.
        """

        # param:data is a pass by reference to a dict we are storing in our scene. We should not overwrite it
        restore_data = copy.deepcopy(data)

        restore_data['camera']['position'] = PlaceState.restore_snapshot(self.session,
                                                                         restore_data['camera']['position'])
        restore_data['camera'] = CameraState.restore_snapshot(self.session, restore_data['camera'])

        restore_data['lighting'] = LightingState.restore_snapshot(self.session, restore_data['lighting'])

        restore_data['material'] = MaterialState.restore_snapshot(self.session, restore_data['material'])

        # Restore the clip planes. The 'clip_planes' key in restore_data is an array of clip planes objects in snapshot
        # form. We need to convert them back into CameraClipPlane objects before restoring the main view data.
        clip_planes_data = restore_data['clip_planes']
        restored_clip_planes = []
        for clip_plane_type, clip_plane_data in clip_planes_data.items():
            if clip_plane_type == "camera":
                restored_clip_planes.append(CameraClipPlaneState.restore_snapshot(self.session, clip_plane_data))
            # TODO find a way to test scene clip planes
            if clip_plane_type == "scene":
                restored_clip_planes.append(SceneClipPlaneState.restore_snapshot(self.session, clip_plane_data))

        restore_data['clip_planes'] = restored_clip_planes

        # The ViewState by default skips resetting the camera because session.restore_options.get('restore camera')
        # is None. We set it to True, let the camera be restored, and then delete the option, so it reads None again in
        # case it is an important option for other parts of the code. We don't need to use the NamedView stored in the
        # scene to restore camera because the camera position is stored/restored using ViewState take and restore
        # snapshot.
        self.session.restore_options['restore camera'] = True
        ViewState.restore_snapshot(self.session, restore_data)
        del self.session.restore_options['restore camera']

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
            'scene_restorables': self.scene_restoreables
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
