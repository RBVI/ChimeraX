def _debug_print(s):
	from time import ctime
	with open("/tmp/chimera2_debug.log", "a") as f:
		print("backend", ctime(), s, file=f)
_debug_print("backend script started")

#
# Main chimera2 code
#

main_view = None
client_data = []

def init_chimera2():
	# initialize chimera2 internals
	#   -- setup graphics to generate JSON
	#   -- register all commands
	#
	# Supported tags include:
	#	"llgr"	-- for llgr JSON format data
	#	...
	global main_view
	from chimera2 import scene
	scene.set_glsl_version('webgl')
	import llgr
	llgr.set_output('json')
	from chimera2 import io, commands
	io.initialize_formats()
	commands.register()
	# TODO: set HOME to home directory of authenticated user, so ~/ works

	# Augment JSON converter to support
	# TODO: figure how to automate this
	from webapp_server import register_json_converter
	register_json_converter(llgr.AttributeInfo, llgr.AttributeInfo.json)
	register_json_converter(llgr.Enum, lambda x: x.value)
	import numpy
	register_json_converter(numpy.ndarray, list)

	# register data handlers
	from chimera2.trackchanges import track
	track.add_handler(scene.View, update_client_data)

	#
	track.block() # block so main_view is assigned before tracking is done
	main_view = scene.View()
	scene.reset()
	track.release()

def update_client_data(views):
	_debug_print("update_client_data: %s in? %s" % (main_view, views.modified))
	if main_view not in views.modified:
		return
	scene_info = {
		'bbox': [
			list(main_view.bbox.llb),
			list(main_view.bbox.urf)
		],
	}
	if main_view.OPEN_MODELS_CHANGE in views.reasons:
		client_data.append(['open_models',
				[(m.id, m.name) for m in main_view.models]])
	if main_view.GRAPHICS_CHANGE in views.reasons:
		pass
	if (main_view.CAMERA_CHANGE in views.reasons
	or main_view.FOV_CHANGE in views.reasons):
		pass
	camera = main_view.camera
	if camera is not None:
		# camera is created/modified as part of scene rendering
		scene_info.update({
			'eye': camera.eye,
			'at': camera.at,
			'up': camera.up,
			'fov': main_view.fov,
			'viewport': main_view.viewport,	# DEBUGGING
		})
	if main_view.VIEWPORT_CHANGE in views.reasons:
		pass
	client_data.append(['scene', scene_info])
	# TODO: should only be needed if GRAPHICS_CHANGE
	info = main_view.render(as_data=True, skip_camera_matrices=True)
	if info:
		client_data.append(['llgr', info])

#
# Main program for per-session backend
#
from webapp_server import Server
class Backend(Server):

	def __init__(self):
		_debug_print("init Server")
		Server.__init__(self)
		import sys
		_debug_print("argv %s" % sys.argv)
		self.name = sys.argv[0]
		self.session_dir = sys.argv[1]
		self.session_name = sys.argv[2]
		self.log = open("/tmp/chimera2_backend.log", "w")
		self.set_log(self.log)
		# register handlers for web client data
		self.register_handler("client_state", self._client_state_handler)
		self.register_handler("command", self._command_handler)
		init_chimera2()

	def _client_state_handler(self, value):
		# process "client_state" data from web client
		# value is a dictionary
		_debug_print("client state handler: %s: %s" % (type(value), value))
		if 'width' in value:
			main_view.viewport = (value['width'], value['height'])
		answer = {
			"status": True		# Success!
		}
		return answer

	def _command_handler(self, value):
		_debug_print("command handler: %s: %s" % (type(value), value))
		answer = {
			"status": True,		# Success!
			"command": str(value),
		}
		from chimera2 import cli
		try:
			from chimera2.trackchanges import track
			cmd = cli.Command(value, final=True)
			cmd.error_check()
			answer["command"] = cmd.current_text
			result = []
			try:
				track.block()
				info = cmd.execute()
			finally:
				track.release()
			if isinstance(info, str):
				result.append(["info", info])
			if client_data:
				result.extend(client_data)
				client_data.clear()
			answer["client_data"] = result
		except cli.UserError as e:
			answer["status"] = False
			answer["error"] = str(e)
		except Exception:
			import traceback
			answer["status"] = False
			answer["error"] = traceback.format_exc()
		return answer

_debug_print("__name__ %s" % __name__)
if __name__ == "__main__":
	try:
		Backend().run()
	except BaseException:
		import traceback
		with open("/tmp/chimera2_debug.log", "a") as f:
			traceback.print_exc(file=f)
	else:
		with open("/tmp/chimera2_debug.log", "a") as f:
			print("run returned", file=f)
