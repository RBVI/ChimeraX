# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
tasks: Task creation and monitoring
===================================

This module defines classes for tasks and their state manager.

Tasks are threads of execution that continue independently from
the main UI thread.  They can be used for web services or local
computations that potentially take a while and should be done
in the background to keep the UI interactive.

Attributes
----------
ADD_TASK : str
    Name of trigger that is fired when a new task
    registers with the state manager.

REMOVE_TASK : str
    Name of trigger that is fired when an existing task
    deregisters with the state manager.

Notes
-----

The :py:class:`Tasks` instance is a singleton per session and may be
referenced as ``session.tasks``.  It is the state manager for
'Task' instances.

A :py:class:`Task` instance represents one thread of execution
and should be registered with 'session.tasks' when
instantiated and deregistered when terminated.

Session-specific triggers are fired when tasks
are registered and deregistered.  To add and remove
:py:class:`Task` handlers, use ``session.trigger.add_handler``
and ``session.trigger.delete_handler``.
"""

import abc
from .session import State, RestoreError

ADD_TASK = 'add task'
REMOVE_TASK = 'remove task'
UPDATE_TASK = 'update task'
END_TASK = 'end task'

# Possible task state
PENDING = "pending"           # Initialized but not running
RUNNING = "running"           # Running
TERMINATING = "terminating"   # Termination requested
TERMINATED = "terminated"   # Termination requested
FINISHED = "finished"       # Finished


class Task(State):
    """Base class for instances of tasks.

    Classes for tasks should inherit from :py:class:`Task` and override methods
    to implement task-specific functionality.  In particularly, methods
    from session :py:class:`~chimera.core.session.State` should be defined
    so that saving and restoring of scenes and sessions work properly,
    and the :py:meth:`run` method should be overriden
    to provide actual functionality.

    Attributes
    ----------
    id : readonly int
        ``id`` is a unique identifier among Task instances
        registered with the session state manager.
    state : readonly str
        ``state`` is one of ``PENDING``, ``RUNNING``, ``TERMINATING``
        ``TERMINATED``, and ``FINISHED``.
    SESSION_ENDURING : bool, class-level optional
        If True, then task survives across sessions.
    SESSION_SKIP : bool, class-level optional
        If True, then task is not saved in sessions.
    """

    SESSION_ENDURING = False
    SESSION_SKIP = False

    def __init__(self, session, id=None, **kw):
        """Initialize a Task.

        Parameters
        ----------
        session : instance of :py:class:`~chimera.core.session.Session`
            Session in which this task was created.

        """
        self.id = id
        import weakref
        self._session = weakref.ref(session)
        self._thread = None
        self._terminate = None
        self.state = PENDING
        session.tasks.add(self)

    @property
    def session(self):
        """Read-only property for session that contains this task."""
        return self._session()

    def display_name(self):
        """Name to display to user for this task.

        This method should be overridden to return a task-specific name.

        """
        return self.__class__.__name__

    def _update_state(self, state):
        self.session.tasks.update_state(self, state)

    def terminate(self):
        """Terminate this task.

        This method should be overridden to clean up
        task data structures.  This base method should be
        called as the last step of task deletion.

        """
        self.session.tasks.remove(self)
        if self._terminate is not None:
            self._terminate.set()
        self._update_state(TERMINATING)

    def terminating(self):
        """Return whether user has requested termination of this task.

        """
        if self._terminate is None:
            return False
        return self._terminate.isSet()

    def terminated(self):
        """Return whether task has finished.

        """
        return self.state in [TERMINATED, FINISHED]

    def start(self, *args, **kw):
        """Start task running.

        This method calls the instance 'start' method in a thread.

        """
        if self.state != PENDING:
            raise RuntimeError("starting task multiple times")
        import threading
        self._terminate = threading.Event()
        self._thread = threading.Thread(target=self._run_thread,
                                        daemon=True, args=args, kwargs=kw)
        self._thread.start()
        self._update_state(RUNNING)

    def _cleanup(self):
        """Clean up after thread has ended.

        This is usually called from the :py:class:`Tasks` instance after a task
        declares itself finished.

        """
        self._thread = None
        self._terminate = None

    def _run_thread(self, *args, **kw):
        try:
            self.run(*args, **kw)
        finally:
            if self.terminating():
                self._update_state(TERMINATED)
            else:
                self._update_state(FINISHED)
        self.session.ui.thread_safe(self.on_finish)

    @abc.abstractmethod
    def run(self, *args, **kw):
        """Run the task.

        This method must be overridden to implement actual functionality.
        :py:meth:`terminating` should be checked regularly to see whether
        user has requested termination.

        """
        raise RuntimeError("base class \"run\" method called.")

    def on_finish(self):
        """Callback method executed after task thread terminates.

        This callback is executed in the UI thread after the
        :py:meth:`run` method returns.  By default, it does nothing.

        """
        pass


class Job(Task):
    """
    'Job' is a long-running task.

    A 'Job' instance is a Chimera task that invokes
    and monitors a long-running process.  Job execution
    is modeled as process launch followed by multiple checks
    for process termination.

    'Job' is implemented by overriding the :py:meth:`run` method to
    launch and monitor the background process.  Subclasses
    should override the 'launch' and 'monitor' methods to
    implement actual functionality.

    Classes deriving from 'Job' indirectly inherits from
    :py:class:`Task` and should override methods to implement
    task-specific functionality.  In particularly, methods
    from session :py:class:`~chimera.core.session.State`
    should be defined so that saving
    and restoring of scenes and sessions work properly.

    Notes
    -----
    :py:meth:`start` is still the main entry point for callers
    to 'Job' instances, not :py:meth:`run`.

    """
    CHECK_INTERVALS = [5, 5, 10, 15, 25, 40, 65, 105, 170, 275]

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._timing_step = 0

    def run(self, *args, **kw):
        """Launch and monitor a background process.

        This method is run in the task thread (not the UI
        thread.  ``run`` calls the abstract methods :py:meth:`launch`,
        :py:meth:`running` and :py:meth:`monitor` to initiate and check status
        of the background process.  Timing of the checks
        are handled by the :py:meth:`next_check` method, which may
        be overridden to provide custom timing.

        """
        import time
        self.launch(*args, **kw)
        while self.running():
            if self.terminating():
                break
            time.sleep(self.next_check())
            self.monitor()

    def next_check(self):
        t = self._timing_step
        self._timing_step += 1
        try:
            # Some predetermined intervals
            return self.CHECK_INTERVALS[t]
        except IndexError:
            # Or five minutes
            return 300

    @abc.abstractmethod
    def launch(self, *args, **kw):
        """Launch the background process.

        """
        raise RuntimeError("base class \"launch\" method called.")

    @abc.abstractmethod
    def running(self):
        """Check if job is running.

        """
        raise RuntimeError("base class \"running\" method called.")

    @abc.abstractmethod
    def monitor(self):
        """Check the status of the background process.

        The task should be marked as terminated (using
        'update_state') when the background process is done

        """
        raise RuntimeError("base class \"monitor\" method called.")

    @abc.abstractmethod
    def exited_normally(self):
        """Return whether job terminated normally.

        Returns
        -------
        status : bool
            True if normal termination, False otherwise.

        """
        raise RuntimeError("base class \"exited_normally\" method called.")


class JobError(RuntimeError):
    """Generic job error."""
    pass


class JobLaunchError(JobError):
    """Exception thrown when job launch fails."""
    pass


class JobMonitorError(JobError):
    """Exception thrown when job status check fails."""
    pass


class Tasks(State):
    """A per-session state manager for tasks.

    :py:class:`Tasks` instances are per-session singletons that track
    tasks in the session, as well as managing saving and restoring
    task states for scenes and sessions.
    """
    VERSION = 1     # snapshot version

    def __init__(self, session):
        """Initialize per-session state manager for tasks.

        Parameters
        ----------
        session : instance of :py:class:`~chimera.core.session.Session`
            Session for which this state manager was created.

        """
        import weakref
        self._session = weakref.ref(session)
        session.triggers.add_trigger(ADD_TASK)
        session.triggers.add_trigger(REMOVE_TASK)
        session.triggers.add_trigger(UPDATE_TASK)
        session.triggers.add_trigger(END_TASK)
        self._tasks = {}
        import itertools
        self._id_counter = itertools.count(1)

    def take_snapshot(self, phase, session, flags):
        """Save state of running tasks.

        Overrides :py:class:`~chimera.core.session.State` default method
        to save state of all registered running tasks.

        Parameters
        ----------
        session : instance of :py:class:`~chimera.core.session.Session`
            Session for which state is being saved.
            Should match the ``session`` argument given to ``__init__``.
        flags : int
            Flags indicating whether snapshot is being taken to
            save scene or session.  See :py:mod:`chimera.core.session` for
            more details.

        """
        if phase == self.SAVE_PHASE:
            data = {}
            for tid, t in self._tasks.items():
                assert(isinstance(t, Task))
                if t.state == RUNNING and not t.SESSION_SKIP:
                    data[tid] = [session.unique_id(t),
                                 t.take_snapshot(session, phase, flags)]
            return [self.VERSION, data]
        elif phase == self.CLEANUP_PHASE:
            for tid, t in self._tasks.items():
                if t.state == RUNNING and not t.SESSION_SKIP:
                    t.take_snapshot(session, phase, flags)

    def restore_snapshot(self, phase, session, version, data):
        """Restore state of running tasks.

        Overrides :py:class:`~chimera.core.session.State` default method to
        restore state of all registered running tasks.

        Parameters
        ----------
        phase : str
            Restoration phase.  See :py:mod:`chimera.core.session` for more
            details.
        session : instance of :py:class:`~chimera.core.session.Session`
            Session for which state is being saved.
            Should match the ``session`` argument given to ``__init__``.
        version : any
            Version of state manager that saved the data.
            Used for determining how to parse the ``data`` argument.
        data : any
            Data saved by state manager during :py:meth:`take_snapshot`.

        """
        if version != self.VERSION:
            raise RestoreError("Unexpected version")

        session = self._session()   # resolve back reference
        for tid, [uid, [task_version, task_data]] in data.items():
            if phase == self.CREATE_PHASE:
                try:
                    cls = session.class_of_unique_id(uid, Task)
                except KeyError:
                    class_name = session.class_name_of_unique_id(uid)
                    session.log.warning("Unable to restore task %s (%s)"
                                        % (tid, class_name))
                    continue
                task = cls(session, id=tid)
                session.restore_unique_id(task, uid)
            else:
                task = session.unique_obj(uid)
            task.restore_snapshot(phase, session, task_version, task_data)

    def reset_state(self):
        """Reset state manager to default state.

        Overrides :py:class:`~chimera.core.session.State` default method
        to reset to default state.  Since the default state has no running
        tasks, all registered tasks are terminated.

        """
        task_list = list(self._tasks.values())
        for tid, t in self._tasks.items():
            if t.SESSION_SKIP or t.SESSION_ENDURING:
                continue
            task.terminate()
            # ?assert(tid not in self._tasks)
            if tid in self._tasks:
                del self._tasks[tid]

    def list(self):
        """Return list of tasks.

        Returns
        -------
        list
            List of :py:class:`Task` instances.

        """
        return list(self._tasks.values())

    def add(self, task):
        """Register task with state manager.

        Parameters
        ----------
        task : :py:class:`Task` instances
            A newly created task.

        """
        session = self._session()   # resolve back reference
        if task.id is None:
            task.id = next(self._id_counter)
        self._tasks[task.id] = task
        session.triggers.activate_trigger(ADD_TASK, task)

    def remove(self, task):
        """Deregister task with state manager.

        Parameters
        ----------
        task_list : list of :py:class:`Task` instances
            List of registered tasks.

        """
        session = self._session()   # resolve back reference
        tid = task.id
        if tid is None:
            # Not registered in a session
            return
        task.id = None
        try:
            del self._tasks[tid]
        except KeyError:
            # Maybe we had reset and there were still old
            # tasks finishing up
            pass
        session.triggers.activate_trigger(REMOVE_TASK, task)

    def update_state(self, task, new_state):
        """Update the state for the given task.

        Parameters
        ----------
        task : :py:class:`Task` instance
            Task whose state just changed
        new_state : str
            New state of the task (one of ``PENDING``, ``RUNNING``,
            ``TERMINATING`` or ``FINISHED``).

        """
        task.state = new_state
        session = self._session()   # resolve back reference
        if task.terminated():
            task._cleanup()
            session.triggers.activate_trigger(END_TASK, task)
        else:
            session.triggers.activate_trigger(UPDATE_TASK, task)

    def find_by_id(self, tid):
        """Return a :py:class:`Task` instance with the matching identifier.

        Parameters
        ----------
        tid : int
            Unique per-session identifier for a registered task.

        """
        return self._tasks.get(tid, None)

    def find_by_class(self, cls):
        """Return a list of tasks of the given class.

        All tasks that match ``cls`` as defined by :py:func:`isinstance`
        are returned.

        Parameters
        ----------
        cls : class object
            Class object used to match task instances.

        """
        return [task for task in self._tasks.values() if isinstance(task, cls)]
