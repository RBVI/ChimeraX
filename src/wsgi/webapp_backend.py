def _debug_print(s):
	from time import ctime
	with open("/tmp/chimera2_debug.log", "a") as f:
		print("backend", ctime(), s, file=f)
_debug_print("backend script started")

#
# Main chimera2 code
#

from chimera2 import cmds

client_state = {}

def init_chimera2():
	# initialize chimera2 internals
	#   -- setup graphics to generate JSON
	#   -- register all commands
	#
	# Individual commands return a list of [tag, tag_data] pairs.
	# Supported tags include:
	#	"llgr"	-- for llgr JSON format data
	#	...
	from chimera2 import scene
	scene.set_glsl_version('webgl')
	import llgr
	llgr.set_output('json')
	import chimera2.io
	chimera2.io.initialize_formats()
	cmds.register('exit', cmd_exit)
	cmds.register('open', cmd_open)
	import chimera2.lighting.cmd # autoregisters commands
	# TODO: set HOME to home directory of authenticated user, so ~/ works

	from webapp_server import register_json_converter
	register_json_converter(llgr.AttributeInfo, llgr.AttributeInfo.json)
	_debug_print(str(dir(llgr)))
	register_json_converter(llgr.Enum, lambda x: x.value)
	import numpy
	register_json_converter(numpy.ndarray, list)

def cmd_exit(status: int=0):
	raise SystemExit(status)

def cmd_open(filename):
	from chimera2 import scene
	scene.reset()
	from chimera2 import io
	try:
		info = io.open(filename)
		result = [['info', info]] if info else []
	except OSError as e:
		raise cmds.UserError(str(e))
	llgr_info = scene.render(as_data=True)
	scene_info = {
		'bbox': [list(scene.bbox.llb), list(scene.bbox.urf)],
	}
	if scene.camera is not None:
		# camera is created/modified as part of scene rendering
		scene_info.update({
			'eye': scene.camera.eye,
			'at': scene.camera.at,
			'up': scene.camera.up,
		})
	return [
		['scene', scene_info ],
		['llgr', llgr_info ],
	] + result

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
		self.register_handler("client_state", self._client_state_handler)
		self.register_handler("command", self._command_handler)
		init_chimera2()

	def _client_state_handler(self, value):
		# value is a dictionary
		_debug_print("client state handler: %s: %s" % (type(value), value))
		client_state.update(value)
		if 'width' in value:
			from chimera2 import scene
			scene.set_viewport(value['width'], value['height'])
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
		try:
			cmd = cmds.Command(value, autocomplete=True)
			cmd.error_check()
			answer["command"] = cmd.current_text
			result = cmd.execute(error_check=False)
			if isinstance(result, str):
				result = [["info", result]]
			# TODO: check dirty bits and add changes to result,
			#	ie: graphics and other state, so result is
			#	always an info string
			answer["client_data"] = result
		except cmds.UserError as e:
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
