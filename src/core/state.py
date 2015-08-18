# vi: set expandtab shiftwidth=4 softtabstop=4:
import abc

class RestoreError(RuntimeError):
    """Raised when session file has a problem being restored"""
    pass

class State(metaclass=abc.ABCMeta):
    """Session state API for classes that support saving session state

    Session state consists only of "simple" types, i.e.,
    those that are supported by the :py:mod:`.serialize` module.

    Since scenes are snapshots of the current session state,
    the State API is reused for scenes.

    References to objects should use the session's unique identifier
    for the object.  The object's class needs to be registered with
    :py:func:`register_unique_class`.
    """
    #: state type
    SCENE = 0x1
    #: state type
    SESSION = 0x2
    ALL = SCENE | SESSION

    #: state take phase
    SAVE_PHASE = 'save data'
    #: state take phase
    CLEANUP_PHASE = 'cleanup temporary data structures'
    #: state restoration phase
    CREATE_PHASE = 'create objects'
    #: state restoration phase
    RESOLVE_PHASE = 'resolve object references'

    #: common exception for needing a newer version of the application
    NeedNewerError = RestoreError(
        "Need newer version of application to restore session")

    @abc.abstractmethod
    def take_snapshot(self, phase, session, flags):
        """Return snapshot of current state, [version, data], of instance.

        The semantics of the data is unknown to the caller.
        Returns None if should be skipped."""
        if phase != self.SAVE_PHASE:
            return
        version = 0
        data = {}
        return [version, data]

    @abc.abstractmethod
    def restore_snapshot(self, phase, session, version, data):
        """Restore data snapshot into instance.

        Restoration is done in two phases: CREATE_PHASE and RESOLVE_PHASE.  The
        first phase should restore all of the data.  The
        second phase should restore references to other objects (data is None).
        The session instance is used to convert unique ids into instances.
        """
        if version != 0 or len(data) > 0:
            raise RestoreError("Unexpected version or data")

    @abc.abstractmethod
    def reset_state(self):
        """Reset state to data-less state"""
        pass

    # possible animation API
    # include here to emphasize that state aware code
    # needs to support animation
    # def restore_frame(self, phase, frame, timeline, transition):
    #    # frame would be the frame number
    #    # timeline would be sequence of (start frame, scene)
    #    # transition would be method to get to given frame that might need
    #    #   look at several scenes
    #    pass


class ParentState(State):
    """Mixin for classes that manage other state instances.

    This class makes the assumptions that the state instances
    are kept as values in a dictionary, and that all of the instances
    are known.
    """

    VERSION = 0
    SKIP = 'skip'
    _child_attr_name = None  # replace in subclass

    def take_snapshot(self, phase, session, flags):
        """Return snapshot of current state"""
        child_dict = getattr(self, self._child_attr_name)
        if phase == self.CLEANUP_PHASE:
            for child in child_dict.values():
                child.take_snapshot(session, phase, flags)
            return
        if phase != self.SAVE_PHASE:
            return
        my_data = []
        for name, child in child_dict.items():
            snapshot = child.take_snapshot(session, phase, flags)
            if snapshot is None:
                continue
            version, data = snapshot
            my_data.append([tag, version, data])
        return [self.VERSION, data]

    def restore_snapshot(self, phase, session, version, data):
        """Restore data snapshot into instance"""
        # TODO: handle previous versions
        if version != self.VERSION or not data:
            raise RestoreError("Unexpected version or data")
        child_dict = getattr(self, self._child_attr_name)
        for name, child_version, child_data in data:
            if name in child_dict:
                child = child_dict[name]
            else:
                child = self.missing_child(name)
                if child is self.SKIP:
                    continue
            child.restore_snapshot(phase, session, child_version, child_data)

    def reset_state(self):
        """Reset state to data-less state"""
        child_dict = getattr(self, self._child_attr_name)
        for child in child_dict.values():
            child.reset_state()

    def missing_child(self, name):
        # TODO: warn about missing child (return self.SKIP)
        #       or create missing child to fill in and return child
        return self.SKIP
