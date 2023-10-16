# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

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
and ``session.trigger.remove_handler``.
"""

import abc
import datetime
import itertools
import sys
import threading
import time
import weakref

from enum import StrEnum
from typing import Optional, Union

from .state import State, StateManager

# If any of the *STATE_VERSIONs change, then increase the (maximum) core session
# number in setup.py.in
TASKS_STATE_VERSION = 1

ADD_TASK = 'add task'
REMOVE_TASK = 'remove task'
UPDATE_TASK = 'update task'
END_TASK = 'end task'

task_triggers = [ADD_TASK, REMOVE_TASK, UPDATE_TASK, END_TASK]

# Possible task states
class TaskState(StrEnum):
    PENDING = "pending"         # Initialized but not running
    RUNNING = "running"         # Running
    TERMINATING = "terminating" # Termination requested
    TERMINATED = "terminated"   # Termination requested
    FINISHED = "finished"       # Finished
    # Webservices states
    STARTED = "started"
    FAILED = "failed"
    DELETED = "deleted"
    CANCELED = "canceled"
    # Unknown state?
    UNDEFINED = "undefined"     # Undefined

    @classmethod
    def from_str(cls, value):
        ret = getattr(cls, value, None)
        ret = ret or getattr(cls, value.upper(), None)
        ret = ret or getattr(cls, value.title(), None)
        if not ret:
            raise NotImplementedError("Unknown TaskState: %s" % value)
        return ret


PENDING = TaskState.PENDING
RUNNING = TaskState.RUNNING
TERMINATING = TaskState.TERMINATING
TERMINATED = TaskState.TERMINATED
FINISHED = TaskState.FINISHED

class Task(State):
    """Base class for instances of tasks.

    Classes for tasks should inherit from :py:class:`Task` and override methods
    to implement task-specific functionality.  In particularly, methods
    from session :py:class:`~chimerax.core.session.State` should be defined
    so that saving and restoring of scenes and sessions work properly,
    and the :py:meth:`run` method should be overriden
    to provide actual functionality.

    Attributes
    ----------
    id : readonly int
        ``id`` is a unique identifier among Task instances
        registered with the session state manager.
    state : readonly TaskState
        ``state`` is one of ``PENDING``, ``RUNNING``, ``TERMINATING``
        ``TERMINATED``, and ``FINISHED``.
    SESSION_ENDURING : bool, class-level optional
        If True, then task survives across sessions.
    SESSION_SAVE : bool, class-level optional
        If True, then task is saved in sessions.
    """

    SESSION_ENDURING = False
    SESSION_SAVE = False

    def __init__(self, session, id: int = None):
        """Initialize a Task.

        Parameters
        ----------
        session : instance of :py:class:`~chimerax.core.session.Session`
            Session in which this task was created.

        """
        self.id = id
        self._session: weakref.ref['Session'] = weakref.ref(session)
        self._thread: threading.Thread = None
        self._terminate = None
        self._state: TaskState = TaskState.PENDING
        self.start_time: Optional[datetime.datetime] = None
        self.end_time: Optional[datetime.datetime] = None
        if session:
            session.tasks.add(self)

    @property
    def session(self):
        """Read-only property for session that contains this task."""
        return self._session()

    @property
    def runtime(self):
        if self.state == TaskState.PENDING:
            return datetime.timedelta()
        if self.state not in [
            TaskState.TERMINATED, TaskState.FINISHED, TaskState.UNDEFINED
            , TaskState.FAILED, TaskState.DELETED, TaskState.CANCELED
        ]:
            return datetime.datetime.now() - self.start_time
        else:
            return self.end_time - self.start_time

    # TODO: @session_trigger(UPDATE_TASK, self)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: Union[str, TaskState]):
        if isinstance(state, str):
            state = TaskState.from_str(state)
        self._state = state
        if self.terminated():
            self._cleanup()
            self.session.triggers.activate_trigger(END_TASK, self)
        else:
            self.session.triggers.activate_trigger(UPDATE_TASK, self)

    # TODO: @session_trigger(END_TASK, self)
    def terminate(self):
        """Terminate this task.

        This method should be overridden to clean up
        task data structures.  This base method should be
        called as the last step of task deletion.

        """
        self.session.tasks.remove(self)
        self.end_time = datetime.datetime.now()
        if self._terminate is not None:
            self._terminate.set()
        self.state = TaskState.TERMINATING

    def terminating(self):
        """Return whether user has requested termination of this task.

        """
        if self._terminate is None:
            return False
        return self._terminate.is_set()

    def terminated(self):
        """Return whether task has finished.

        """
        return self.state in [TaskState.TERMINATED, TaskState.FINISHED]

    def start(self, *args, **kw):
        """Start task running.

        If a keyword arguments 'blocking' is present and true,
        this method calls the instance 'run' method in the
        current thread; otherwise, it calls the instance
        'run' method in a separate thread.  The 'blocking'
        keyword is passed through, so the 'run' methods in
        derived classes will see it.

        """
        if self.state != TaskState.PENDING:
            raise RuntimeError("starting task multiple times")
        blocking = kw.get("blocking", False) # since _run_thread will pop() it
        self._thread = threading.Thread(target=self._run_function,
                                        daemon=True, args=(self.run, *args), kwargs=kw)
        self._thread.start()
        self.start_time = datetime.datetime.now()
        self.state = TaskState.RUNNING
        self._terminate = threading.Event()
        if blocking:
            self._thread.join()
            self.state = TaskState.FINISHED
            # the non-blocking code path also has an on_finish()
            # call that executes asynchronously
            if self.launched_successfully:
                self.session.ui.thread_safe(self.on_finish)

    def restore(self, *args, **kw):
        """Like start, but for restoring a task from a snapshot."""
        blocking = kw.get("blocking", False) # since _run_thread will pop() it
        self._thread = threading.Thread(target=self._run_function,
                                        daemon=True, args=(self._relaunch, *args), kwargs=kw)
        self._thread.start()
        self.start_time = datetime.datetime.now()
        self.state = TaskState.RUNNING
        self._terminate = threading.Event()
        if blocking:
            self._thread.join()
            self.state = TaskState.FINISHED
            # the non-blocking code path also has an on_finish()
            # call that executes asynchronously
            if self.launched_successfully:
                self.session.ui.thread_safe(self.on_finish)

    def _cleanup(self):
        """Clean up after thread has ended.

        This is usually called from the :py:class:`Tasks` instance after a task
        declares itself finished.

        """
        self._thread = None
        self._terminate = None

    def _run_function(self, func: callable, *args, **kw):
        blocking = kw.pop("blocking", False)
        func(*args, **kw)
        if self.terminating():
            self.state = TaskState.TERMINATED
        else:
            self.state = TaskState.FINISHED
        if not blocking and self.launched_successfully and not self.state in [
            TaskState.CANCELED, TaskState.DELETED, TaskState.FAILED, TaskState.TERMINATED
        ]:
            # the blocking code path also has an on_finish() call that executes immediately
            self.session.ui.thread_safe(self.on_finish)

    def exited_normally(self) -> bool:
        return True

    def launched_successfully(self) -> bool:
        return True

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

    def __str__(self):
        return ("ChimeraX Task, ID %s" % self.id)

    def thread_safe_status(self, message: str):
        if self.session:
            status = self.session.logger.status
            tsafe = self.session.ui.thread_safe
            tsafe(status, message)

    def thread_safe_log(self, message: str):
        if self.session:
            status = self.session.logger.info
            tsafe = self.session.ui.thread_safe
            tsafe(status, message)

    def thread_safe_warning(self, message: str):
        if self.session:
            status = self.session.logger.warning
            tsafe = self.session.ui.thread_safe
            tsafe(status, message)

    def thread_safe_error(self, message: str):
        if self.session:
            status = self.session.logger.error
            tsafe = self.session.ui.thread_safe
            tsafe(status, message)


    def from_snapshot(self, session, data):
        pass

    def take_snapshot(self, session, flags) -> dict[any, any]:
        data = {
            "id": self.id
            # msgpack is schizophrenic about enums and can't
            # decide whether it can or can't serialize them
            # so we'll just use strings
            , "state": str(self.state)
            , "start_time": self.start_time
            , "end_time": self.end_time
        }
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        pass

class Job(Task):
    """
    'Job' is a long-running task.

    A 'Job' instance is a ChimeraX task that invokes
    and monitors a long-running process.  Job execution
    is modeled as process launch followed by multiple checks
    for process termination.

    'Job' implements a minimal run function which checks for
    termination, waits for a time step, and then calls a
    monitor method. To implement functionality, the 'run'
    method must be overriden. At the end of the run method,
    subclasses can call `super().run()` to hook into this
    functionality.

    Any status updating logic should be implemented by overriding
    the 'monitor' method.

    Finally, next_check can be overridden to provide alternative
    timetables for updating the status of tasks.

    Classes deriving from 'Job' indirectly inherit from
    :py:class:`Task` and should override methods to implement
    task-specific functionality.  In particularly, methods
    from session :py:class:`~chimerax.core.session.State`
    should be defined so that saving
    and restoring of scenes and sessions work properly.

    Notes
    -----
    :py:meth:`start` is still the main entry point for callers
    to 'Job' instances, not :py:meth:`run`.

    """
    local_timing_intervals = [
        5, 5, 10, 15, 25, 40, 65, 105, 170, 275, 300, 350, 400, 450, 500
        , 550, 600, 650, 700, 750, 800
    ]
    def __init__(self, session):
        super().__init__(session)
        self._local_timing_step = 0

    def run(self):
        while self.running():
            if self.terminating():
                break
            time.sleep(self.next_check())
            if self.running():
                self.monitor()

    def next_check(self):
        t = self._local_timing_step
        t += 1
        try:
            return self.local_timing_intervals[t]
        except IndexError:
            # 5 minutes
            return 300

    @abc.abstractmethod
    def running(self):
        """Check if job is running.

        """
        raise RuntimeError("base class \"running\" method called.")

    def monitor(self):
        """Check the status of the background process.

        The task should be marked as terminated
        when the background process is done

        """
        pass

    def __str__(self):
        return ("ChimeraX Job, ID %s" % self.id)


class JobError(RuntimeError):
    """Generic job error."""
    pass


class JobLaunchError(JobError):
    """Exception thrown when job launch fails."""
    pass


class JobMonitorError(JobError):
    """Exception thrown when job status check fails."""
    pass


class Tasks(StateManager):
    """A per-session state manager for tasks.

    :py:class:`Tasks` instances are per-session singletons that track
    tasks in the session, as well as managing saving and restoring
    task states for scenes and sessions.
    """

    def __init__(self, session, ids_start_from = 1):
        """Initialize per-session state manager for tasks.

        Parameters
        ----------
        session : instance of :py:class:`~chimerax.core.session.Session`
            Session for which this state manager was created.

        """
        self._session = weakref.ref(session)
        self._id_counter = itertools.count(ids_start_from)
        self._tasks = {}

    def __len__(self) -> int:
        "Return the number of registered tasks."
        return len(self._tasks)

    def __contains__(self, item) -> bool:
        return item in self._tasks

    def __iter__(self):
        return iter(self._tasks)

    def __getitem__(self, key):
        return self._tasks[key]

    def keys(self):
        return self._tasks.keys()

    def items(self):
        return self._tasks.items()

    def values(self):
        return self._tasks.values()

    def list(self):
        """Return list of tasks.

        Returns
        -------
        list
            List of :py:class:`Task` instances.

        """
        return list(self._tasks.values())

    # session.tasks.add(self) should == session.tasks[None] = self
    def add(self, task):
        self.__setitem__(None, task)

    # TODO: @session_trigger(ADD_TASK, task)
    def __setitem__(self, key, task):
        if key in self:
            raise ValueError("Attempted to record task ID already in task list")
        if key is None:
            id = next(self._id_counter)
            task.id = id
            dict.__setitem__(self._tasks, id, task)
        else:
            dict.__setitem__(self._tasks, key, task)
        if self.session:
            self.session.triggers.activate_trigger(ADD_TASK, task)

    # TODO: @session_trigger(REMOVE_TASK, task)
    def __delitem__(self, key):
        """Deregister task with state manager.

        Parameters
        ----------
        task_list : list of :py:class:`Task` instances
            List of registered tasks.

        """
        task = self._tasks[key]
        try:
            del self._tasks[key]
        except KeyError:
            # Maybe we had reset and there were still old
            # tasks finishing up
            pass
        self.session.triggers.activate_trigger(REMOVE_TASK, task)
        del task

    def remove(self, task: Task) -> None:
        self.__delitem__(task.id)

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

    @property
    def session(self):
        """Read-only property for session that contains this task."""
        return self._session()

    def take_snapshot(self, session, flags):
        """Save state of running tasks.

        Overrides :py:class:`~chimerax.core.session.State` default method
        to save state of all registered running tasks.

        Parameters
        ----------
        session : instance of :py:class:`~chimerax.core.session.Session`
            Session for which state is being saved.
            Should match the ``session`` argument given to ``__init__``.
        flags : int
            Flags indicating whether snapshot is being taken to
            save scene or session.  See :py:mod:`chimerax.core.session` for
            more details.

        """
        tasks = {}
        for tid, task in self._tasks.items():
            assert(isinstance(task, Task))
            if task.SESSION_SAVE:
                tasks[tid] = task
            # tasks[tid] = task.take_snapshot(session, flags)
        data = {'tasks': tasks,
                'version': TASKS_STATE_VERSION,
                'counter': next(self._id_counter) - 1}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        """Restore state of running tasks.

        Overrides :py:class:`~chimerax.core.session.State` default method to
        restore state of all registered running tasks.

        Parameters
        ----------
        session : instance of :py:class:`~chimerax.core.session.Session`
            Session for which state is being saved.
            Should match the ``session`` argument given to ``__init__``.
        data : any
            Data saved by state manager during :py:meth:`take_snapshot`.

        """
        t = session.tasks
        for tid, task in data['tasks'].items():
            t._tasks[tid] = task
        # TODO: t._id_counter?
        return t

    def reset_state(self, session):
        """Reset state manager to default state.

        Overrides :py:class:`~chimerax.core.session.State` default method
        to reset to default state.  Since the default state has no running
        tasks, all registered tasks are terminated.

        """
        tids = list(self._tasks.keys())
        for tid in tids:
            try:
                task = self._tasks[tid]
            except KeyError:
                continue
            if task.SESSION_ENDURING or not task.SESSION_SAVE:
                continue
            task.terminate()
            try:
                del self._tasks[tid]
            except KeyError:
                # In case terminating the task removed it from task list
                pass
