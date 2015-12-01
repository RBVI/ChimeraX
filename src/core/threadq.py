import threading
class WorkThread(threading.Thread):
    """Compute a molecular surface"""
    def __init__(self, function, in_queue, out_queue):
        threading.Thread.__init__(self)
        self.function = function
        self.in_queue = in_queue
        self.out_queue = out_queue

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

    # Make print statements in threads not attempt display in GUI thread
    # in log since this will cause crash.
    redirect_stdout_stderr(True)

    # Create threads.
    for i in range(nthread):
        t = WorkThread(func, in_queue, out_queue)
        t.daemon = True
        t.start()

    # Wait until input queue is empty.
    in_queue.join()

    redirect_stdout_stderr(False)

    results = []
    while not out_queue.empty():
        r = out_queue.get()
        if isinstance(r, Exception):
            raise r
        results.append(r)

    return results

# Print statements in threads cannot display immediately in the GUI thread
# without causing a crash. Keep the output and display after threads complete.
def redirect_stdout_stderr(redirect):
    import sys
    if redirect:
        class Output:
            def __init__(self):
                from queue import Queue
                self._queue = Queue()
            def write(self, s):
                self._queue.put(s)
            def flush(self):
                return
            def __str__(self):
                q = self._queue
                lines = ''.join(tuple(q.queue))
                return lines

        sys._keep_stdout = sys.stdout
        sys.stdout = out = Output()
        sys._keep_stderr = sys.stderr
        sys.stderr = out
    else:
        out = sys.stdout
        sys.stdout = sys._keep_stdout
        sys.stderr = sys._keep_stderr
        print(out)
            
        
