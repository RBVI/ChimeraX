# Cycle display among several models.

class Cycle_Model_Display:
    
    def __init__(self, models, frames_per_model = 1, frames = None, session = None):

        self.models = models
        self.frames_per_model = frames_per_model
        self.frames = frames
        self.session = session
        self.frame = None

    def start(self):

        self.frame = 0

        mlist = self.models
        for m in mlist:
            m.display = False
        if mlist:
            mlist[0].display = True

        v = self.session.view
        v.add_new_frame_callback(self.next_frame)

    def stop(self):

        if not self.frame is None:
            v = self.session.view
            v.remove_new_frame_callback(self.next_frame)
            self.frame = None

    def next_frame(self):

        f,nf = self.frame+1, self.frames
        if not nf is None and f >= nf:
            self.stop()
            return

        mlist = self.models
        nm = len(mlist)
        mi = (f//self.frames_per_model) % nm
        m = mlist[mi]
        m.display = True
        mprev = mlist[(mi + nm-1) % nm]
        mprev.display = False

        self.frame += 1

def cycle_command(cmdname, args, session):

    from ..ui.commands import models_arg, int_arg, no_arg, parse_arguments
    req_args = ()
    opt_args = (('models', models_arg),)
    kw_args = (('wait', int_arg),
               ('frames', int_arg),
               ('stop', no_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    cycle(**kw)

def cycle(models = None, wait = 1, frames = None, stop = False, session = None):
    if models is None:
        models = session.model_list()
    if len(models) == 0:
        return

    if stop:
        if hasattr(session, 'cycle_models'):
            for c in session.cycle_models:
                c.stop()
            session.cycle_models = []
    else:
        c = Cycle_Model_Display(models, wait, frames, session)
        c.start()
        if hasattr(session, 'cycle_models'):
            session.cycle_models.append(c)
        else:
            session.cycle_models = [c]
    
