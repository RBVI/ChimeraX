def _debug_print(s):
	from time import ctime
	with open("/tmp/chimera2_debug.log", "a") as f:
		print("backend", ctime(), s, file=f)
_debug_print("backend script started")

def init_chimera2():
	# TODO: put in separate file
	import llgr
	llgr.set_output('json')
	import chimera2.io
	chimera2.io.initialize_formats()
	from chimera2 import cmds
	cmds.register('open', cmd_open)
	# TODO: set HOME to home directory of authenticated user, so ~/ works

def cmd_open(filename):
	from chimera2 import scene
	scene.reset()
	from chimera2 import io
	try:
		io.open(filename)
	except OSError as e:
		return ["error", str(e)]
	from math import radians
	from chimera2.math3d import Identity
	return ['json', scene.render((0, 0, 200, 200), radians(30), Identity(), as_string=True)]

def process_command(text):
	from chimera2 import cmds
	try:
		cmd = cmds.Command(text, autocomplete=True)
		return cmd.execute()
	except cmds.UserError as e:
		return ["error", str(e)]
	except Exception:
		import traceback
		return ["error", traceback.format_exc()]

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
		self.register_handler("command", self._command_handler)
		init_chimera2()

	def _command_handler(self, value):
		_debug_print("command handler: %s: %s" % (type(value), value))
		answer = dict()
		answer["status"] = True		# Success!
		answer["stdout"] = str(value)
		answer["client_data"] = process_command(value)
		if answer["client_data"][0] == "error":
			answer["status"] = False
		return answer

_debug_print("__name__ %s" % __name__)
if __name__ == "__main__":
	try:
		Backend().run()
	except:
		import traceback
		with open("/tmp/chimera2_debug.log", "a") as f:
			traceback.print_exc(file=f)
	else:
		with open("/tmp/chimera2_debug.log", "a") as f:
			print("run returned", file=f)
