#
# Turn command to rotate models.
#
def turn_command(cmdname, args, session):

    from .commands import float_arg, int_arg, axis_arg, parse_arguments
    req_args = (('axis', axis_arg),
                ('angle', float_arg),)
    opt_args = (('frames', int_arg),)
    kw_args = ()
    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    turn(**kw)

def turn(axis, angle, frames = 1, session = None):

    v = session.view
    c = v.camera
    cv = c.view()
    saxis = cv.apply_without_translation(axis)  # Convert axis from camera to scene coordinates
    center = v.center_of_rotation
    from ..geometry.place import rotation
    r = rotation(saxis, -angle, center)
    if frames == 1:
        c.set_view(r*cv)
    else:
        def rotate(r=r,c=c):
            c.set_view(r*c.view())
        call_for_n_frames(rotate, frames, session)

class call_for_n_frames:
    
    def __init__(self, func, n, session):
        self.func = func
        self.n = n
        self.session = session
        self.frame = 0
        session.view.add_new_frame_callback(self.call)
    def call(self):
        f = self.frame
        if f >= self.n:
            self.done()
        else:
            self.func()
            self.frame = f+1
    def done(self):
        v = self.session.view
        v.remove_new_frame_callback(self.call)

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

def turn_old(axis, angle=1.5, frames=None, coordinateSystem=None, models=None,
	 center=None, precessionTilt=None):
	param = turnParam(axis, angle, frames, coordinateSystem, models,
			  center, precessionTilt)
	param['removalHandler'] = _removeMotionHandler
	if frames is None:
		_movement(None, param, None)
	else:
		_addMotionHandler(_movement, param)

def turnParam(axis, angle=1.5, frames=None, coordinateSystem=None, models=None,
	      center=None, precessionTilt=None):
	v = _axis(axis)
	param = {'command':'turn',
		 'xform':chimera.Xform.rotation(v, angle),
		 'frames':frames,
		 'coordinateSystem':coordinateSystem,
		 'models':models,
		 'center':center}
	if not precessionTilt is None:
		import Matrix
		x,y,z = Matrix.orthonormal_frame(v.data())
		from math import sin, cos, pi
		a = precessionTilt*pi/180
		pa = Matrix.linear_combination(cos(a), z, -sin(a), y)
		param['precessionAxis'] = chimera.Vector(*pa)
		param['precessionStep'] = -angle
	return param

def doRoll(cmdName, args):
	# 'turn' also calls this
	argWords = args.split()

	kw = {'axis':'y'}
	t = _parseKeywordArg(argWords, 'precessionTilt', cmdName, float)
	if not t is None:
		kw['precessionTilt'] = t

	_parseMovementKeywordArgs(argWords, 0, kw, cmdName)

	n = len(argWords)
	if n >= 2:
		kw['angle'] = _parseFloat(argWords[1], cmdName, 'angle')
	if n >= 3:
		kw['frames'] = _parseInt(argWords[2], cmdName, 'frames')
	if n > 3:
		raise MidasError('wait_frames argument not implemented')
	f = getattr(Midas, cmdName)
	f(**kw)


# Parse center, models, axis and coordinateSystem keyword arguments.
def _parseMovementKeywordArgs(argWords, axisArgIndex, kw, cmdName):
	_parseCoordinateSystemArg(argWords, kw, cmdName)
	a = 0
	ccs = acs = None
	while a < len(argWords):
		w = argWords[a].lower()
		if 'models'.startswith(w):
			if a+1 >= len(argWords):
				raise MidasError('%s missing models argument' %
						 cmdName)
			kw['models'] = modelsFromSpec(argWords[a+1], 'models')
			del argWords[a:a+2]
		elif w.startswith('ce') and 'center'.startswith(w):
			if a+1 >= len(argWords):
				raise MidasError('%s missing center argument'
						 % cmdName)
			kw['center'], ccs = parseCenterArg(argWords[a+1],
							   cmdName)
			del argWords[a:a+2]
		else:
			a += 1

	# Parse axis argument
	if len(argWords) > axisArgIndex:
		axis, axisPoint, acs = _parseAxis(argWords[axisArgIndex],
						  cmdName)
		kw['axis'] = axis
		if not 'center' in kw and axisPoint:
			# Use axis point if no center specified.
			kw['center'] = axisPoint
			ccs = acs

	# If no coordinate system specified use axis or center coord system.
	cs = (ccs or acs)
	if not 'coordinateSystem' in kw and cs: 
		kw['coordinateSystem'] = cs
		xf = cs.xform.inverse()
		if 'center' in kw and not ccs:
			kw['center'] = xf.apply(kw['center'])
		if 'axis' in kw and not acs:
			kw['axis'] = xf.apply(kw['axis'])

	# Convert axis and center to requested coordinate system.
	if 'coordinateSystem' in kw:
		xf = kw['coordinateSystem'].xform.inverse()
		if 'center' in kw and ccs:
			kw['center'] = xf.apply(ccs.xform.apply(kw['center']))
		if 'axis' in kw and acs:
			kw['axis'] = xf.apply(acs.xform.apply(kw['axis']))

def _parseCoordinateSystemArg(argWords, kw, cmdName):
	a = 0
	while a < len(argWords):
		w = argWords[a].lower()
		if 'coordinatesystem'.startswith(w):
			if a+1 >= len(argWords):
				raise MidasError('%s missing coordinateSystem argument' % cmdName)
			os = openStateFromSpec(argWords[a+1],
					       'coordinateSystem')
			kw['coordinateSystem'] = os
			del argWords[a:a+2]
		else:
			a += 1

def _parseKeywordArg(argWords, kwName, cmdName, type=None, multiple=False):
	args = []
	a = 0
	while a < len(argWords):
		w = argWords[a].lower()
		if kwName.lower().startswith(w):
			if a+1 >= len(argWords):
				raise MidasError('%s missing %s argument' %
						 (cmdName, kwName))
			arg = argWords[a+1]
			del argWords[a:a+2]
			if not type is None:
				try:
					arg = type(arg)
				except:
					raise MidasError('%s illegal %s value "%s"' % (cmdName, kwName, arg))
			if multiple:
				args.append(arg)
			else:
				return arg
		else:
			a += 1
	return args if multiple else None

#
# Internal functions that support motion commands
#
_motionHandlers = []
def _addMotionHandler(func, param):
	if not isinstance(param['frames'], int):
		raise MidasError("'frames' must be an integer")
	h = chimera.triggers.addHandler('new frame', func, param)
	param['handler'] = h
	_motionHandlers.append(param)

def _removeMotionHandler(param):
	try:
		_motionHandlers.remove(param)
		chimera.triggers.deleteHandler('new frame', param['handler'])
		param['handler'] = None  # break cycle
	except KeyError:
		pass

def _clearMotionHandlers():
	global _motionHandlers
	for param in _motionHandlers:
		chimera.triggers.deleteHandler('new frame', param['handler'])
		param['handler'] = None  # break cycle
	_motionHandlers = []

def _motionRemaining():
	frames = 0
	for param in _motionHandlers:
		f = param['frames']
		if f > 0 and f > frames:
			frames = f
	import Fly
	frames = max(frames, Fly.frames_left())
	return frames

def _tickMotionHandler(param):
	if param['frames'] > 0:
		param['frames'] = param['frames'] - 1
		if param['frames'] == 0:
			if param.has_key('removalHandler'):
				param['removalHandler'](param)
			else:
				_removeMotionHandler(param)
	elif not chimera.openModels.list():
		if param.has_key('removalHandler'):
			param['removalHandler'](param)
		else:
			_removeMotionHandler(param)

def _movement(trigger, param, triggerData):
	if trigger:
		_tickMotionHandler(param)
	if not _moveModels(param['xform'],
			   param.get('coordinateSystem', None),
			   param.get('models', None),
			   param.get('center', None),
			   param.get('precessionAxis', None),
			   param.get('precessionStep', None)):
		if param.has_key('removalHandler'):
			param['removalHandler'](param)
		else:
			_removeMotionHandler(param)

def _moveModels(xf, coordsys_open_state, models, center,
		precessionAxis = None, precessionStep = None):
	if not precessionAxis is None:
		from chimera import Xform
		xfp = Xform.identity()
		xfp.premultiply(xf)
		pa = xf.apply(precessionAxis)
		xfp.premultiply(Xform.rotation(precessionAxis, precessionStep))
		precessionAxis.x = pa.x
		precessionAxis.y = pa.y
		precessionAxis.z = pa.z
		xf = xfp
	if coordsys_open_state:
		if coordsys_open_state.__destroyed__:
			return False
		cxf = coordsys_open_state.xform
		from chimera import Xform
		rxf = Xform.translation(-cxf.getTranslation())
		rxf.multiply(cxf)
		axf = Xform()
		for x in (rxf, xf, rxf.inverse()):
			axf.multiply(x)
		# Reorthogonalize for stability when coordinate frame rotating
		axf.makeOrthogonal(True)
		xf = axf
		if center:
			center = cxf.apply(center)
	from chimera import openModels as om
	if models is None and center is None:
		om.applyXform(xf)
	else:
		if models is None:
			models = [m for m in om.list() if m.openState.active]
		for os in set([m.openState for m in models
			       if not m.__destroyed__]):
			_moveOpenState(os, xf, center)
	return True

def _moveOpenState(os, xf, center):
	# Adjust transform to act about center of rotation.
	from chimera import openModels as om, Xform
	if center:
		c = center
	elif om.cofrMethod == om.Independent:
		c = os.xform.apply(os.cofr)
	else:
		c = om.cofr
	cxf = Xform.translation(c.x, c.y, c.z)
	cxf.multiply(xf)
	cxf.multiply(Xform.translation(-c.x, -c.y, -c.z))
	os.globalXform(cxf)
