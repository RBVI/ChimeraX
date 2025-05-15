"""
This module defines the Animation and Keyframe classes for managing animations in ChimeraX.

Classes:
    Animation: Manages the creation, deletion, saving, restoring, and interpolation of animations.
    Keyframe: Represents a keyframe in an animation, storing a scene name, time, and thumbnail.

Functions:
    format_time(time): Convert time in seconds to min:sec.__ format.
"""

from chimerax.core.state import StateManager, State
from chimerax.core.commands.motion import CallForNFrames
from chimerax.core.commands.run import run
from chimerax.core.triggerset import TriggerSet
from chimerax.core.errors import UserError
from .triggers import MGR_KF_ADDED, MGR_KF_DELETED, MGR_KF_EDITED, MGR_LENGTH_CHANGED, MGR_PREVIEWED, activate_trigger, \
    MGR_FRAME_PLAYED, MGR_RECORDING_START, MGR_RECORDING_STOP
from chimerax.movie.moviecmd import movie_encode
from chimerax.movie.moviecmd import movie_record

import io


def format_time(time):
    """
        Convert time in seconds to min:sec.__ format.

        Args:
            time (float): Time in seconds.

        Returns:
            str: Formatted time string.
        """
    """Convert time in seconds to min:sec.__ format."""
    minutes = int(time // 60)
    seconds = int(time % 60)
    fractional_seconds = round(time % 1, 2)
    return f"{minutes}:{seconds:02}.{int(fractional_seconds * 100):02}"


class Animation(StateManager):
    """
    Manages the creation, deletion, saving, restoring, and interpolation of animations.

    Attributes:
        MAX_LENGTH (int): Maximum length of the animation in seconds.
        version (int): Version of the Animation class.
        fps (int): Frames per second for the animation.
        DEFAULT_LENGTH (int): Default length of the animation in seconds.
    """

    MAX_LENGTH = 5 * 60  # 5 minutes
    version = 0
    fps = 144
    DEFAULT_LENGTH = 5  # in seconds

    def __init__(self, session, *, animation_data=None):
        """
        Initialize the Animation object.

        Args:
            session: The current session.
            animation_data (dict, optional): Data to restore the animation from a snapshot.
        """

        self.session = session
        self.logger = session.logger

        # list of steps to interpolate animation. Each step is a tuple of (scene_name1, scene_name2, %) interpolation
        # steps
        self._lerp_steps: [(str, str, int | float)] = []
        self._need_frames_update = True
        self._is_playing = False
        self._is_recording = False

        # dict representing arguments for the movie record command.
        self._record_data = None
        self._encode_data = None

        self.keyframes: [Keyframe] = []

        if animation_data is None:
            self.set_length(self.DEFAULT_LENGTH)
        else:
            for kf_data in animation_data['keyframes']:
                kf = Keyframe.restore_snapshot(session, kf_data)
                if kf is not None:
                    # None means the keyframe couldn't be restored because the scene doesn't exist
                    self.keyframes.append(kf)
                    activate_trigger(MGR_KF_ADDED, kf)
            self.set_length(animation_data['length'])

        self._call_for_n_frames = None

    def add_keyframe(self, keyframe_name: str, time: int | float | None = None):
        """
        Add a keyframe to the animation.

        Args:
            keyframe_name (str): Name of the keyframe.
            time (int | float, optional): Time of the keyframe in seconds.
        """

        # If there is no time param specified, then the keyframe should be created 1 second after the last keyframe or
        # 1 second after the start of the animation if there are no keyframes.
        if time is None:
            kf_time = self._last_kf_time() + 1
            if kf_time > self.length:
                self.set_length(kf_time)
        else:
            kf_time = time

        if self.keyframe_exists(keyframe_name):
            self.logger.warning(f"Can't create keyframe {keyframe_name} because it already exists.")
            return
        scenes = self.session.scenes.get_scene(keyframe_name)
        if scenes is None:
            self.logger.warning(f"Can't create keyframe for scene {keyframe_name} because it doesn't exist.")
            return
        if not self.validate_time(kf_time):
            return
        new_kf = Keyframe(self.session, keyframe_name, kf_time)
        self.keyframes.append(new_kf)
        self._sort_keyframes()
        self._need_frames_update = True
        self.logger.info(f"Created keyframe: {keyframe_name} at time: {format_time(kf_time)}")
        activate_trigger(MGR_KF_ADDED, new_kf)

    def edit_keyframe_time(self, keyframe_name, time):
        """
        Edit the time of an existing keyframe.

        Args:
            keyframe_name (str): Name of the keyframe.
            time (int | float): New time for the keyframe.
        """
        if not self.keyframe_exists(keyframe_name):
            self.logger.warning(f"Can't edit keyframe {keyframe_name} because it doesn't exist.")
            return
        kf: Keyframe = self.get_keyframe(keyframe_name)
        if kf.get_time() == time:
            self.logger.warning(f"{keyframe_name} is already at time {time}.")
            return
        # Extend the length of the animation if needed
        if time > self.length:
            if time > self.MAX_LENGTH:
                self.logger.warning(f"Can't edit keyframe {keyframe_name} because time {time} is over the "
                                            f"{self.MAX_LENGTH} second limit.")
                return
            self.set_length(time)
        if not self.validate_time(time):
            return
        kf.set_time(time)
        self._sort_keyframes()
        self._need_frames_update = True
        self.logger.info(f"Edited keyframe {keyframe_name} to time: {format_time(time)}")
        activate_trigger(MGR_KF_EDITED, kf)

    def delete_keyframe(self, keyframe_name):
        """
        Delete a keyframe from the animation.

        Args:
            keyframe_name (str): Name of the keyframe to delete.
        """
        if not self.keyframe_exists(keyframe_name):
            self.logger.warning(f"Can't delete keyframe {keyframe_name} because it doesn't exist.")
            return
        kf_to_delete = self.get_keyframe(keyframe_name)
        self.keyframes.remove(kf_to_delete)
        self._need_frames_update = True
        self.logger.info(f"Deleted keyframe {keyframe_name}")
        activate_trigger(MGR_KF_DELETED, kf_to_delete)

    def delete_all_keyframes(self):
        """
        Delete all keyframes from the animation.
        """
        if len(self.keyframes) < 1:
            self.logger.warning(f"There are no keyframes.")
            return
        while len(self.keyframes) > 0:
            kf = self.keyframes[-1]
            self.delete_keyframe(kf.get_name())
        self.logger.info(f"Deleted all keyframes")

    def insert_time(self, target_time, amount_for_insertion):
        """
        Insert time into the animation. All keyframes after the target time will be moved by the amount for insertion.

        Args:
            target_time (int | float): Time at which to insert the time.
            amount_for_insertion (int | float): Amount of time to insert.
        """
        if not self.validate_time(target_time, ignore_length=True):
            return
        if not isinstance(amount_for_insertion, (int, float)):
            self.logger.warning(f"Amount for insertion must be an integer or float.")
            return
        if amount_for_insertion < 0:
            self.logger.warning(f"Can't insert negative time.")
            return
        # If the target time is greater than the length of the animation, try to make the animation longer
        if target_time > self.length:
            if self.length + target_time + amount_for_insertion > self.MAX_LENGTH:
                self.logger.warning(f"Can't insert time because it would exceed the {self.MAX_LENGTH} second limit.")
                return
            # Extend the length of the animation to accommodate the target time
            self.set_length(target_time)
        else:
            if self.length + amount_for_insertion > self.MAX_LENGTH:
                self.logger.warning(f"Can't insert time because it would exceed the {self.MAX_LENGTH} second limit.")
                return
        self.set_length(self.length + amount_for_insertion)

        # Move all keyframes after the target time by the amount for insertion.
        # This keyframe adjusting loop has to run in reverse so to avoid a case when editing earliest to latest
        # keyframes, 2 keyframes are spaced equal to the insertion amount, which would result in them trying to hold
        # the same time value.
        for kf in reversed(self.keyframes):
            if kf.get_time() > target_time:
                kf.set_time(kf.get_time() + amount_for_insertion)
                self._need_frames_update = True
                activate_trigger(MGR_KF_EDITED, kf)

        self.logger.info(f"Inserted {amount_for_insertion} seconds at time {target_time}")

    def remove_time(self, target_time, amount_for_removal):
        """
        Remove time from the animation. All keyframes after the target time will be moved closer to the target time.

        Args:
            target_time (int | float): Time at which to remove the time.
            amount_for_removal (int | float): Amount of time to remove.
        """

        if not self.validate_time(target_time, ignore_keyframes=True):
            return
        if not isinstance(amount_for_removal, (int, float)):
            self.logger.warning(f"Amount for removal must be an integer or float.")
            return
        if amount_for_removal < 0:
            self.logger.warning(f"Can't remove negative time.")
            return
        if target_time > self.get_time_length() - amount_for_removal:
            self.logger.warning(f"Can't remove {format_time(amount_for_removal)} seconds because it would remove target"
                                f" time {format_time(target_time)}.")
            return
        frames_after_time_change = (self.get_time_length() - amount_for_removal) * self.get_frame_rate()
        if frames_after_time_change < 1:
            self.logger.warning(f"Can't remove {amount_for_removal} seconds because it would make the animation less "
                                f"than 1 frame long at {self.get_frame_rate()} fps.")
            return

        for kf in self.keyframes:
            if kf.get_time() > target_time:
                if target_time + amount_for_removal > kf.get_time():
                    self.logger.warning(f"Can't remove {amount_for_removal} because blocked by keyframe at "
                                        f"{format_time(kf.get_time())}.")
                    return

        # Move all keyframes after the target time by the amount for removal.
        for kf in self.keyframes:
            if kf.get_time() > target_time:
                kf.set_time(kf.get_time() - amount_for_removal)
                activate_trigger(MGR_KF_EDITED, kf)

        self.set_length(self.length - amount_for_removal)
        self.logger.info(f"Removed {amount_for_removal} seconds at time {format_time(target_time)}")

    def list_keyframes(self) -> list[str]:
        """
        List all keyframes in the animation.

        Returns:
            list[str]: List of keyframes with their times.
        """
        keyframe_list = []
        keyframe_list.append(f"Start: {format_time(0)}")
        for kf in self.keyframes:
            keyframe_list.append(f"{kf.get_name()}: {format_time(kf.get_time())}")
        keyframe_list.append(f"End: {format_time(self.length)}")
        return keyframe_list

    def preview(self, time):
        """
        Preview the animation at a specific time.

        Args:
            time (int | float): Time to preview the animation.
        """
        if not isinstance(time, (int, float)):
            self.logger.warning(f"Time must be an integer or float")
            return
        if time < 0 or time > self.length:
            self.logger.warning(f"Time must be between 0 and {self.length}")
            return

        # make sure the interpolation steps are up to date
        self._try_frame_refresh()

        step = round(self.fps * time)
        if step >= len(self._lerp_steps):
            step = len(self._lerp_steps) - 1
        scene1, scene2, fraction = self._lerp_steps[step]
        self.session.scenes.interpolate_scenes(scene1, scene2, fraction)
        self.logger.info(f"Previewing animation at time {format_time(time)}")
        activate_trigger(MGR_PREVIEWED, time)

    def play(self, start_time=0, reverse=False):
        """
        Play the animation from a specific start time.

        Args:
            start_time (int | float, optional): Time to start playing the animation.
            reverse (bool, optional): Whether to play the animation in reverse.
        """
        if self._is_playing:
            return

        if start_time < 0 or start_time > self.length:
            self.logger.warning(f"Start time must be between 0 and {self.length}")
            return

        self._try_frame_refresh()

        self.logger.status(f"Playing animation...")

        start_frame = max(0, min(round(self.fps * start_time), len(self._lerp_steps) - 1))

        # callback function for each frame
        def frame_cb(session, f):
            self._is_playing = True
            if reverse:
                frame_num = start_frame - f
                last_frame = 0
            else:
                frame_num = start_frame + f
                last_frame = len(self._lerp_steps) - 1
            # get the lerp step for this frame
            lerp_step = self._lerp_steps[frame_num]
            scene1, scene2, fraction = lerp_step
            self.session.scenes.interpolate_scenes(scene1, scene2, fraction)
            if frame_num == last_frame:
                self._is_playing = False
                self.logger.status(f"Finished playing animation.")
                self._try_end_recording()

            # The animation has exactly time * fps frames. That means there is either a 0:00 frame and not a
            # frame for the last time stamp or vice versa. Since we index 0 based, when we play reverse we end on frame
            # 0, get 0 / fps and the time emitted is 0 which is correct. When we play forwards and assume that frame 0
            # is the first frame, we need to add 1 to the frame number to get the correct time for the end otherwise
            # we get (length * fps - 1) / fps which is not the correct time. This causes a 5-second animation to end at
            # something like 4.99 seconds. This approach does mean the first frame played is not 0:00 seconds but rather
            # 1 / fps seconds. This is a tradeoff to ensure the end time is correct.
            time = (frame_num + 1) / self.fps if not reverse else frame_num / self.fps
            activate_trigger(MGR_FRAME_PLAYED, time)

        # Calculate how many frames need to be played between start_frame and the end of the animation. Take reverse
        # into account
        if reverse:
            num_frames_to_play = start_frame + 1  # from start_frame to 0 (inclusive)
        else:
            num_frames_to_play = len(self._lerp_steps) - start_frame  # from start_frame to the end
        self._call_for_n_frames = CallForNFrames(frame_cb, num_frames_to_play, self.session)

    def record(self, record_data=None, encode_data=None, reverse=False):
        """
        Start a recording for the animation using the chimerax.movie module.

        Args:
            record_data (dict, optional): Arguments for the movie record command.
            encode_data (dict, optional): Arguments for the movie encode command.
            reverse (bool, optional): Whether to play the animation in reverse.
        """
        if self._is_recording:
            self.logger.warning(f"Already recording an animation. Stop recording before starting a new one.")
            return
        self._record_data = record_data
        self._encode_data = encode_data
        # Add framerate to the encode data. The movie command takes this as a separate argument from the encode command
        # But we want the animation tool to track the framerate.
        self._encode_data['framerate'] = self.fps
        # Make sure the animation interpolation steps are generated before we start recording
        self._try_frame_refresh()
        # Stop a movie if one is already recording
        if hasattr(self.session, 'movie') and self.session.movie is not None:
            run(self.session, "movie reset", log=False)
        # If we want to ever show commands in the log this needs to be converted
        movie_record(self.session, **self._record_data)
        self._is_recording = True
        activate_trigger(MGR_RECORDING_START, None)
        self.play(reverse)

    def stop_playing(self, stop_recording=False):
        """
        Stop playing the animation.

        Args:
            stop_recording (bool, optional): Whether to stop recording as well.
        """
        if self._is_recording and not stop_recording:
            return
        if self._call_for_n_frames is not None:
            self._call_for_n_frames.done()
            self._is_playing = False
            self._call_for_n_frames = None
            self.logger.status(f"Stopped playing animation.")

            if stop_recording:
                self._try_end_recording()

    def _gen_lerp_steps(self):
        """
        Generate interpolation steps for the animation.
        """
        if len(self.keyframes) < 1:
            self.logger.warning(f"Can't generate lerp steps because there are no keyframes.")
            return

        # reset lerp steps
        self._lerp_steps = []

        self.logger.info(f"Generating interpolation steps for animation...")

        # tuple val to store previously iterated keyframe (keyframe name, time).
        prev_kf_name = None
        prev_kf_time = None
        # ittr all the keyframes
        for kf in self.keyframes:
            # calculate delta time between keyframes. If prev_kf is None, then delta t is keyframe time minus start of
            # animation time
            if prev_kf_time is None:
                d_time = kf.get_time() - 0
            else:
                d_time = kf.get_time() - prev_kf_time

            if prev_kf_name is None:
                # if prev_kf is None, then we are at the first keyframe. Assume the first keyframe is the state of
                # the animation between 0 and the first keyframe time, so we essentially make duplicate
                # frames from time 0 to the first keyframe
                kf_lerp_steps = self._gen_ntime_lerp_segment(kf.get_name(), kf.get_name(), d_time)
            else:
                kf_lerp_steps = self._gen_ntime_lerp_segment(prev_kf_name, kf.get_name(), d_time)

            # append the lerp steps connecting the two keyframes to the main lerp steps list
            self._lerp_steps.extend(kf_lerp_steps)

            # reset previous ittr keyframe vars
            prev_kf_name = kf.get_name()
            prev_kf_time = kf.get_time()

        # Still need to add the last keyframe to the end of the animation. Same deal as the 0:00 - first keyframe with
        # assuming the last keyframe is the state of the animation between the last keyframe and the end of the
        # animation

        # calculate delta time between last keyframe and end of animation time
        d_time = self.length - prev_kf_time
        # create lerp steps between last keyframe and end of animation. prev_kf will be the last keyframe bc of the loop
        kf_lerp_steps = self._gen_ntime_lerp_segment(prev_kf_name, prev_kf_name, d_time)
        # append the lerp steps connecting the last keyframe to the end of the animation to the main lerp steps list
        self._lerp_steps.extend(kf_lerp_steps)

        self.logger.info(f"Finished generating interpolation steps for animation.")

    def set_length(self, length):
        """
        Set the length of the animation.

        Args:
            length (int | float): New length of the animation in seconds.
        """
        if not isinstance(length, (int, float)):
            self.logger.warning(f"Length must be an integer or float")
            return
        if length < self._last_kf_time():
            run(self.session, "animations timeline", log=False)
            self.logger.warning(f"Length must be greater than {self._last_kf_time()}")
            return
        if length > self.MAX_LENGTH:
            self.logger.warning(f"Length must be less than {self.MAX_LENGTH} seconds")
            return
        if length * self.fps < 1:
            self.logger.warning(f"Length {length} is less than 1 frame long at {self.get_frame_rate()} fps.")
            return

        self.length = length
        # make sure to update the interpolation steps after time is adjusted
        self._need_frames_update = True
        self.logger.info(f"Updated animation length to {format_time(self.length)}")
        activate_trigger(MGR_LENGTH_CHANGED, self.length)

    def _gen_ntime_lerp_segment(self, kf1, kf2, d_time):
        """
        Generate linear interpolation steps between two keyframes.

        Args:
            kf1 (str): Name of the first keyframe.
            kf2 (str): Name of the second keyframe.
            d_time (int | float): Time difference between the keyframes.

        Returns:
            list[tuple]: List of interpolation steps.
        """
        # calculate number of steps/frames between keyframes using delta time and fps. Must be whole number
        n_frames = round(d_time * self.fps)

        # create an array of % decimals that linearly range [0.0, 1.0) in n_frames steps
        fractions = [i / (n_frames - 1) for i in range(n_frames)] if n_frames > 1 else [0]

        # return an array of tuples of (kf1, kf2, fraction) for each fraction in fractions
        return [(kf1, kf2, f) for f in fractions]

    def _try_end_recording(self):
        """
        End the recording and encode the movie if the animation is currently recording.
        """
        if self._is_recording:
            run(self.session, "movie stop", log=False)
            # If this command ever wants to be seen in the log would have to unpack the encode_data dict and pass it
            try:
                movie_encode(self.session, **self._encode_data)
            except UserError as e:
                # This user error can happen if the output is invalid or the encoding fails. We still need to run our
                # code to stop recording and reset the recording flag and fire our trigger.
                self.logger.error(str(e))
            self._is_recording = False
            activate_trigger(MGR_RECORDING_STOP, None)

    def _sort_keyframes(self):
        """
        Sort keyframes by time.
        """
        self.keyframes.sort(key=lambda kf: kf.get_time())

    def _try_frame_refresh(self):
        """
        Refresh the interpolation steps if needed.
        """
        if self._need_frames_update:
            self._gen_lerp_steps()
            self._need_frames_update = False

    def _last_kf_time(self):
        """
        Get the time of the last keyframe.

        Returns:
            int | float: Time of the last keyframe.
        """
        if len(self.keyframes) < 1:
            return 0
        return self.keyframes[-1].get_time()

    def validate_time(self, time, ignore_length=False, ignore_keyframes=False):
        """
        Validate time for a keyframe.

        Args:
            time (int | float): Time to validate.
            ignore_length (bool, optional): Whether to ignore the length of the animation.
            ignore_keyframes (bool, optional): Whether to ignore existing keyframes.

        Returns:
            bool: True if the time is valid, False otherwise.
        """
        if not isinstance(time, (int, float)):
            self.logger.warning(f"Time must be an integer or float")
            return False
        if not ignore_length and not self.time_in_range(time):
            self.logger.warning(f"Time must be between 0 and {self.length}")
            return False
        if not ignore_keyframes and time in [kf.get_time() for kf in self.keyframes]:
            self.logger.warning(f"Time {time} is already taken by a different keyframe.")
            return False
        return True

    def time_in_range(self, time):
        """
        Check if a time is within the range of the animation.

        Args:
            time (int | float): Time to check.

        Returns:
            bool: True if the time is within range, False otherwise.
        """
        return 0 <= time <= self.length

    def get_time_length(self):
        """
        Get the second length of the animation.

        Returns:
            int | float: Length of the animation in seconds.
        """
        return self.length

    def keyframe_exists(self, keyframe_name):
        """
        Check if a keyframe exists.

        Args:
            keyframe_name (str): Name of the keyframe.

        Returns:
            bool: True if the keyframe exists, False otherwise.
        """
        return any(kf.get_name() == keyframe_name for kf in self.keyframes)

    def get_keyframe(self, keyframe_name):
        return next((kf for kf in self.keyframes if kf.get_name() == keyframe_name), None)

    def get_keyframes(self):
        """
        Get all keyframes in the animation.

        Returns:
            list[Keyframe]: List of keyframes.
        """
        return self.keyframes

    def get_num_keyframes(self):
        """
        Get the number of keyframes in the animation.

        Returns:
            int: Number of keyframes.
        """
        return len(self.keyframes)

    def get_frame_rate(self):
        """
        Get the frame rate of the animation.

        Returns:
            int: Frame rate in frames per second.
        """
        return self.fps

    def reset_state(self, session):
        """
        Reset the state of the animation. Empty the interpolation steps, delete all keyframes, and set the length to the
        default length.

        Args:
            session: The current session.
        """
        self._lerp_steps: [(str, str, int | float)] = []
        self._need_frames_update = True
        if len(self.keyframes) > 0:
            self.delete_all_keyframes()
        self.set_length(self.DEFAULT_LENGTH) # in seconds

    def take_snapshot(self, session, flags):
        return {
            'version': self.version,
            'keyframes': [kf.take_snapshot(session, flags) for kf in self.keyframes],
            'length': self.length,
        }

    @staticmethod
    def restore_snapshot(session, data):
        if Animation.version != data['version']:
            raise ValueError(f"Can't restore snapshot version {data['version']} to version {Animation.version}")
        return Animation(session, animation_data=data)


class Keyframe(State):
    """
    Represents a keyframe in an animation, storing a scene name, time, and thumbnail.

    Attributes:
        version (int): Version of the Keyframe class.
        thumbnail_size (tuple): Size of the thumbnail image.
    """

    version = 0
    thumbnail_size = (200, 200)

    def __init__(self, session, name, time, thumbnail=None):
        """
        Initialize the Keyframe object.

        Args:
            session: The current session.
            name (str): Name of the keyframe.
            time (int | float): Time of the keyframe in seconds.
            thumbnail (bytes, optional): Thumbnail image data.
        """

        self.session = session
        # Store a reference to the scene manager for easy access
        self.scene_mgr = session.scenes

        # only create the scene if it doesn't exist already. This will prevent scenes from being created extra times
        # when restoring snapshots.
        if not self.scene_mgr.get_scene(name):
            raise ValueError(f"Can't create keyframe {name} because a matching name scene doesn't exist.")

        self.name = name
        self.time = time
        if thumbnail is None:
            self.thumbnail = self.take_thumbnail()
        else:
            self.thumbnail = thumbnail

    def take_thumbnail(self):
        """
        Take a thumbnail and return the bytes array for a JPEG image.

        Returns:
            bytes: Thumbnail image data.
        """
        pil_image = self.session.view.image(*self.thumbnail_size)
        byte_stream = io.BytesIO()
        pil_image.save(byte_stream, format='JPEG')
        return byte_stream.getvalue()

    def get_time(self):
        """
        Get the time of the keyframe.

        Returns:
            int | float: Time of the keyframe in seconds.
        """
        return self.time

    def get_name(self):
        """
        Get the name of the keyframe.

        Returns:
            str: Name of the keyframe.
        """
        return self.name

    def get_thumbnail(self):
        """
        Get the thumbnail image data.

        Returns:
            bytes: Thumbnail image data.
        """
        return self.thumbnail

    def set_time(self, time):
        """
        Set the time of the keyframe.

        Args:
            time (int | float): New time in seconds for the keyframe.
        """
        self.time = time

    def take_snapshot(self, session, flags):
        return {
            'name': self.name,
            'time': self.time,
            'thumbnail': self.thumbnail,
            'version': self.version,
        }

    @staticmethod
    def restore_snapshot(session, data):
        if Keyframe.version != data['version']:
            raise ValueError(f"Can't restore snapshot version {data['version']} to version {Keyframe.version}")

        scene_mgr = session.scenes
        if not scene_mgr.get_scene(data['name']):
            session.logger.error(f"Can't restore keyframe {data['name']} because the scene doesn't exist.")
            return None
        return Keyframe(session, data['name'], data['time'], data['thumbnail'])
