# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

import threading
class WorkThread(threading.Thread):
    """Run a function in a thread."""
    def __init__(self, function, in_queue = None, out_queue = None):
        threading.Thread.__init__(self)
        self.function = function

        from queue import Queue
        self.in_queue = Queue() if in_queue is None else in_queue
        self.out_queue = Queue() if out_queue is None else out_queue

    def run(self):
        import queue
        while True:
            try:
                s = self.in_queue.get_nowait()
            except queue.Empty:
                break
            try:
                r = self.function(*s)
            except Exception as e:
                r = e
            self.out_queue.put(r)
            self.in_queue.task_done()

# List of return values does not match args ordering.
def apply_to_list(func, args, nthread = None):

    if nthread is None:
        from multiprocessing import cpu_count
        nthread = cpu_count()//2

    nthread = min(len(args), nthread)

    if nthread <= 1:
        return [func(*a) for a in args]

    # Create input and output queues.
    from queue import Queue
    in_queue = Queue()
    for a in args:
        in_queue.put(a)
    out_queue = Queue()

    # Create threads.
    # Threads will exit when there is no more input.
    for i in range(nthread):
        t = WorkThread(func, in_queue, out_queue)
        t.daemon = True
        t.start()

    # Wait until input queue is empty.
    in_queue.join()

    results = []
    while not out_queue.empty():
        r = out_queue.get()
        if isinstance(r, Exception):
            raise r
        results.append(r)

    return results
