# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
HTTP queueing -- avoids flooding servers by queueing requests and only running so many at a time.

Non-blocking usage:
    You get the (singleton) HTTP queue with httpq.get(session).  You then get slots to handle your requests
with calls to queue.new_slot(http_server_name).  With each slot, you request a callback to your workhorse
function with slot.request(workhorse_func, arg1, arg1, ...).  When your workhorse function is called it
will be executing in a thread, so ensure that calls that may change the GUI (e.g. logger calls) or that
make further queueing requests are called via session.ui.thread_safe(func, arg1, arg2, ..., kw1=v1, ...).
Once a slot's function runs, the slot can be reused via another slot.request(...) call [again, not made
in a thread].

Blocking usage:
    Here you typically create your own private HTTPQueue instance so that you don't wait for jobs that you
did not create yourself.  Usage is basically the same as for the public HTTPQueue, except that after you
issue all your new_slot requests, you call the queue's wait() method to wait for them to complete.
"""

class HTTPQueue:
    '''Handles all queueing'''

    def __init__(self, session, *, thread_max=10):
        self.server_map = {}
        self.session = session
        self.thread_max = thread_max
        from threading import Lock
        self.wait_lock = Lock()

    def new_slot(self, server_name):
        if not self.server_map:
            self.wait_lock.acquire()
        try:
            server = self.server_map[server_name]
        except KeyError:
            server = self.server_map[server_name] = HTTPQueueServer(self, server_name,
                thread_max=self.thread_max)
        return server.make_slot()

    def wait(self):
        self.wait_lock.acquire()
        self.wait_lock.release()

    def _delete_server(self, server_name):
        del self.server_map[server_name]
        if not self.server_map:
            self.wait_lock.release()

class HTTPQueueServer:
    '''Handles all requests to a particular host'''

    def __init__(self, queue, server_name, *, thread_max=10):
        self.queue = queue
        self.server_name = server_name
        self.thread_max = thread_max
        self.slots = set()
        self.requests = []
        self.running = set()

    def make_slot(self):
        slot = HTTPQueueSlot(self)
        self.slots.add(slot)
        return slot

    def delete_slot(self):
        self.slots.remove(slot)
        if not self.slots:
            self.queue._delete_server(self.server_name)

    def new_request(self, slot):
        self.requests.append(slot)
        self._start_request()

    def _start_request(self):
        while (self.thread_max is None or len(self.running) < self.thread_max) and self.requests:
            slot = self.requests.pop(0)
            self.running.add(slot)
            from threading import Thread
            Thread(target=slot.run, daemon=False).start()

    def _request_done(self, slot):
        self.running.remove(slot)
        slot._request_finished()
        self._start_request()

class HTTPQueueSlot:
    '''Class for making a single request to the server.  Can be reused for a series of sequential requests'''

    def __init__(self, server):
        self.server = server
        self.request_data = None

    def request(self, func, *args):
        if self.request_data is not None:
            raise RuntimeError("Cannot make new request to slot until previous is finished")
        self.request_data = (func, args)
        self.server.new_request(self)

    def run(self):
        session = self.server.queue.session
        func, args = self.request_data
        try:
            func(*args)
        except Exception:
            def exception_reporting(ses, exc_info):
                ses.logger.report_exception(preface="Error generating/processing HTTP request",
                    exc_info=exc_info)
            import sys
            session.ui.thread_safe(exception_reporting, session, sys.exc_info())
        session.ui.thread_safe(self.server._request_done, self)

    def _request_finished(self):
        self.request_data = None

    def finished(self):
        self.server.delete_slot(self)

def get(session):
    if not hasattr(session, '_http_queue'):
        session._http_queue = HTTPQueue(session)
    return session._http_queue
