# vi: set expandtab shiftwidth=4 softtabstop=4:
class CallForNFrames:
    # CallForNFrames acts like a function that keeps track of per-frame
    # functions.  But those functions need state, so that state is
    # encapsulated in instances of this class.
    #
    # Instances are automatically added to the given session 'Attribute'.

    Infinite = -1
    Attribute = 'motion_in_progress'    # session attribute

    def __init__(self, func, n, session):
        self.func = func
        self.n = n
        self.session = session
        self.frame = 0
        self.handler = session.triggers.add_handler('new frame', self)
        if not hasattr(session, self.Attribute):
            setattr(session, self.Attribute, set([self]))
        else:
            getattr(session, self.Attribute).add(self)

    def __call__(self, *_):
        f = self.frame
        if self.n != self.Infinite and f >= self.n:
            self.done()
        else:
            self.func(self.session, f)
            self.frame = f + 1

    def done(self):
        s = self.session
        s.triggers.delete_handler(self.handler)
        getattr(s, self.Attribute).remove(self)


def motion_in_progress(session):
    # Return True if there are non-infinite motions
    if not hasattr(session, CallForNFrames.Attribute):
        return False
    has_finite_motion = False
    for m in getattr(session, CallForNFrames.Attribute):
        if m.n == CallForNFrames.Infinite:
            return False
        has_finite_motion = True
    return has_finite_motion
