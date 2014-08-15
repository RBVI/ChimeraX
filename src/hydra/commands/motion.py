class call_for_n_frames:
    
    def __init__(self, func, n, session):
        self.func = func
        self.n = n
        self.session = session
        self.frame = 0
        session.view.add_new_frame_callback(self.call)
        if not hasattr(session, 'motion_in_progress'):
            session.motion_in_progress = set()
        session.motion_in_progress.add(self)
    def call(self):
        f = self.frame
        if f >= self.n:
            self.done()
        else:
            self.func()
            self.frame = f+1
    def done(self):
        s = self.session
        s.view.remove_new_frame_callback(self.call)
        s.motion_in_progress.remove(self)

def freeze_command(cmdname, args, session):

    from .parse import parse_arguments
    req_args = opt_args = kw_args = ()
    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    freeze(**kw)

def freeze(session):

    if hasattr(session, 'motion_in_progress'):
        for mip in tuple(session.motion_in_progress):
            mip.done()

def motion_in_progress(session):

    return len(getattr(session, 'motion_in_progress', ())) > 0
