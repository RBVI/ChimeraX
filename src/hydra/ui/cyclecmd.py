# Cycle display among several models.

class Cycle_Model_Display:
    
    def __init__(self, models, frames_per_model = 1, frames = None, bounce = False, session = None):

        self.models = list(models)
        self.frames_per_model = frames_per_model
        self.frames = frames
        self.bounce = bounce
        self.session = session
        self.frame = None
        self.mlast = None

    def start(self):

        self.frame = 0

        mlist = self.models
        for m in mlist[1:]:
            m.display = False
        if mlist:
            m = mlist[0]
            m.display = True
            self.mlast = m

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
        i = f//self.frames_per_model
        if self.bounce:
            il = i % (2*nm-2)
            mi = (2*nm - 2 - il) if il >= nm else il
        else:
            mi = i % nm
        m = mlist[mi]

        mlast = self.mlast
        if not mlast is m:
            m.display = True		            # Display this model
            if not mlast is None:
                mlast.display = False	            # Undisplay last model
            self.mlast = m

        self.frame += 1

def cycle_command(cmdname, args, session):

    from ..ui.commands import models_arg, int_arg, no_arg, parse_arguments
    req_args = ()
    opt_args = (('models', models_arg),)
    kw_args = (('wait', int_arg),
               ('frames', int_arg),
               ('stop', no_arg),
               ('bounce', no_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    cycle(**kw)

def cycle(models = None, wait = 1, frames = None, stop = False, bounce = False, session = None):
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
        c = Cycle_Model_Display(models, wait, frames, bounce, session)
        c.start()
        if hasattr(session, 'cycle_models'):
            session.cycle_models.append(c)
        else:
            session.cycle_models = [c]
    
