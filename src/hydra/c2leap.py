from . import Leap

class LeapListener(Leap.Listener):
    def __init__(self, viewer):
        Leap.Listener.__init__(self)
        self.viewer = viewer
        self.last_time = None
        self.last_positions = None      # Two finger tip positions, chopstick mode
        self.last_hand = None           # Used by position mode
#        self.mode = 'chopsticks'        # chopsticks or position or velocity
        self.mode = 'position'
#        self.mode = 'velocity'

        self.max_delay = 0.1            # Maximum delay between events, seconds
        self.rotation_speed = 3         # models rotate by physical rotation angle
        				#  times this factor 
        self.min_rotation = 10          # degrees
        self.translation_speed = 0.01   # fraction of screen width model motion per
        				#  millimeter of physical motion
        self.min_translation = 15       # millimeters

    def on_init(self, controller):
        print ("Initialized leap")

    def on_connect(self, controller):
        print ("Connected leap")
        if not controller.has_focus:
            print('Application does not have Leap focus')

    def on_focus_gained(self, controller):
        print ("App got Leap focus")

    def on_focus_lost(self, controller):
        print ("App lost Leap focus")

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print ("Disconnected leap")

    def on_exit(self, controller):
        print ("Exited leap")

    def on_frame(self, controller):
        m = self.mode
        if m == 'chopsticks':
            self.chopsticks(controller)
        elif m == 'position':
            self.position(controller)
        elif m == 'velocity':
            self.velocity(controller)

    def chopsticks(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()
        hands = frame.hands
        if len(hands) != 2:
            return

        f1, f2 = hands[0].fingers, hands[1].fingers
        if len(f1) != 1 or len(f2) != 1:
            return

        lp = self.last_positions
        p1, p2 = f1[0].tip_position, f2[0].tip_position
        self.last_positions = (p1,p2)

        from time import time
        t = time()
        lt = self.last_time
        self.last_time = t
        if lt is None or t > lt + self.max_delay or lp is None:
            return

        sep, dsep, dmid, rangle, raxis = chopstick_motion(lp, (p1,p2))

        zmag, tmag, rmag = abs(dsep), norm(dmid), abs(sep*rangle)
        mmax = max(zmag, tmag, rmag)

        v = self.viewer
        s = max(v.window_size)*v.pixel_size()
        if zmag == mmax:
            # Zoom
            f = dsep/sep
            v.translate(0, 0, f*s)
        elif tmag == mmax:
            # Translate (coordinates in millimeters)
            f = s*self.translation_speed
            tx,ty,tz = keep_within_frustum((f*dmid[0], f*dmid[1], 0), v)
            v.translate(tx,ty,tz)
        elif rmag == mmax:
            # Rotation
            from math import pi
            v.rotate(raxis, self.rotation_speed*rangle*180/pi)

        self.last_time = t
        self.last_positions = (p1,p2)

    def position(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()
        hands = frame.hands
        if len(hands) != 1:
            return

        hand = hands[0]
        c = hand.palm_position
        n = hand.palm_normal
        d = hand.direction

        lh = self.last_hand
        self.last_hand = (c,n,d)

        # Ignore events if time lag is too long.
        from time import time
        t = time()
        lt = self.last_time
        self.last_time = t
        if lt is None or t > lt + self.max_delay or lh is None:
            return

        if len(hand.fingers) == 0:
            return      # Stop motion when user makes fist

        lc,ln,ld = lh

        # Hand translation
        t = subtract_vec(c,lc)
        tmag = norm(t)

        # Hand rotation from orthonormal frames (normal, finger dir, thumb dir)
        lf = (vec_xyz(ln),vec_xyz(ld),cross_product_vec(ln,ld))
        f = (vec_xyz(n),vec_xyz(d),cross_product_vec(n,d))
        axis, angle = frame_rotation(lf, f)
        rmag = angle*100

        mmax = max(tmag, rmag)

        v = self.viewer
        s = max(v.window_size)*v.pixel_size()
        if tmag == mmax:
            # Translate (coordinates in millimeters)
            f = s*self.translation_speed
            tx,ty,tz = keep_within_frustum((f*t[0], f*t[1], f*t[2]), v)
            v.translate(tx,ty,tz)
        elif rmag == mmax:
            # Rotation
            from math import pi
            v.rotate(axis, self.rotation_speed*angle*180/pi)

    def velocity(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()
        hands = frame.hands
        if len(hands) != 1:
            return

        # Ignore events if time lag is too long.
        from time import time
        t = time()
        lt = self.last_time
        self.last_time = t
        if lt is None or t > lt + self.max_delay:
            return

        hand = hands[0]
        if len(hand.fingers) == 0:
            return      # Stop motion when user makes fist

        c = hand.palm_position
        n = hand.palm_normal
        d = hand.direction

        # Hand translation
        t = subtract(vec_xyz(c), (0,200,0))     # Y coord is not centered on 0
        tmag = norm(t)
        
        # Hand rotation from orthonormal frames (normal, finger dir, thumb dir)
        nf = ((0,-1,0),(0,0,-1),(1,0,0))
        f = (vec_xyz(n),vec_xyz(d),cross_product_vec(n,d))
        axis, angle = frame_rotation(nf, f)
        rmag = angle*100

        mmax = max(tmag, rmag)

        v = self.viewer
        s = max(v.window_size)*v.pixel_size()
        if tmag == mmax:
            # Translate (coordinates in millimeters)
            if tmag >= self.min_translation:
                f = 0.03*s*self.translation_speed*(1.0-self.min_translation/tmag)**2
                tx,ty,tz = keep_within_frustum((f*t[0], f*t[1], f*t[2]), v)
                v.translate(tx,ty,tz)
        elif rmag == mmax:
            # Rotation
            from math import pi
            if angle >= self.min_rotation*pi/180:
                ra = 0.1*self.rotation_speed*angle*180/pi*(1.0 - self.min_rotation/(angle*180/pi))**2
                v.rotate(axis, ra)

def chopstick_motion(lp, p):
    from math import sqrt, asin

    l1, l2 = lp
    p1, p2 = p
    dp = subtract_vec(p2, p1)
    sep = norm(dp)
    dl = subtract_vec(l2, l1)
    lsep = norm(dl)
    dsep = sep - lsep

    mid = (0.5*(p1.x+p2.x), 0.5*(p1.y+p2.y), 0.5*(p1.z+p2.z))
    lmid = (0.5*(l1.x+l2.x), 0.5*(l1.y+l2.y), 0.5*(l1.z+l2.z))
    dmid = subtract(mid, lmid)

    raxis = cross_product(dl,dp)
    r = norm(raxis)
    if r > 0:
        raxis = tuple(c/r for c in raxis)
    dpdl = sep*lsep
    rangle = asin(r/dpdl) if dpdl > 0 and r <= dpdl else 0       # rotation angle

    return sep, dsep, dmid, rangle, raxis

def keep_within_frustum(t, v):

    c = v.center_of_rotation
    cvi = v.camera_view_inverse
    from .. import matrix
    p1 = matrix.apply_matrix(cvi, c)
    p2 = add(p1, t)
    from math import pi, sin, cos
    a = 0.5*v.field_of_view*pi/180
    ca, sa = cos(a), sin(a)
    ww,wh = v.window_size
    aspect = float(wh)/ww
    # Normals point into frustum.
    plane_normals = ((-ca,0,-sa),        # Right face
                     (ca,0,-sa),       # Left face
                     (0,-ca,-aspect*sa), # Top face
                     (0,ca,-aspect*sa)) # Bottom face
    fmin = None
    for p in plane_normals:
        ip1, ip2 = inner_product(p1, p), inner_product(p2, p)
        if ip1 >= 0 and ip2 < 0:
            f = ip1 / (ip1 - ip2)
            if fmin is None or f < fmin:
                fmin = f
        elif ip1 < 0 and ip2 < ip1:
            fmin = 0    # Outside frustum and moving further away
    if fmin is None:
        return t

    return tuple(x*fmin for x in t)

def vec_xyz(u):
    return (u.x,u.y,u.z)

def add(u,v):
    return (u[0]+v[0], u[1]+v[1], u[2]+v[2])

def subtract_vec(u,v):
    return (u.x-v.x, u.y-v.y, u.z-v.z)

def subtract(u,v):
    return (u[0]-v[0], u[1]-v[1], u[2]-v[2])

def cross_product(u,v):
    (ux,uy,uz),(vx,vy,vz) = u,v
    return (uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx)

def cross_product_vec(u,v):
    return cross_product(vec_xyz(u), vec_xyz(v))

def norm(v):
    from math import sqrt
    return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

def inner_product(u, v):
    return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]

# frame has x, y and z axis coordinates as rows.
def frame_rotation(lf, f):
    # Rotation matrix is transpose of frame vectors
    from numpy import transpose, dot
    r = dot(transpose(f), lf)
    return R_to_axis_angle(r)

def R_to_axis_angle(matrix):
    """Convert the rotation matrix into the axis-angle notation.

    Conversion equations
    ====================

    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::

        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)

    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    from numpy import array, zeros, hypot, float64
    from math import atan2
    # Axes.
    axis = zeros(3, float64)
    matrix = array(matrix)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = hypot(axis[0], hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = atan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis, theta

def leap_listener(viewer):
    v = viewer
    if not hasattr(v, 'leap_listener'):
        # Create a leap listener and controller
        v.leap_listener = l = LeapListener(v)
        v.leap_controller = c = Leap.Controller()
        # Have the listener receive events from the controller
        if not c.add_listener(l):
            print('Adding leap listener failed')
# Docs say it is not connected immediately.  Listener will get on_connect callback.
#        if not c.is_connected:
#            print('Leap controller is not connected to device')
        
    return v.leap_listener

def leap_mode(mode, viewer):
    l = leap_listener(viewer)
    l.mode = mode

def report_leap_focus(viewer):
    v = viewer
    if hasattr(v, 'leap_controller'):
        from ..gui import show_status
        msg = 'App has leap focus' if v.leap_controller.has_focus else 'App does not have Leap focus'
        show_status(msg)

def quit_leap(viewer):
    v = viewer
    if hasattr(v, 'leap_listener'):
        v.leap_controller.remove_listener(v.leap_listener)
