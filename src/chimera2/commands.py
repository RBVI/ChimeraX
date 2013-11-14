"""
commands -- Default set of commands
===================================

This module implements the default set of cli commands
"""

from . import cli

def cmd_exit():
	raise SystemExit(0)

def cmd_open(filename):
	try:
		from chimera2 import io
		return io.open(filename)
	except OSError as e:
		raise cli.UserError(e)

def cmd_close(model_id):
	from . import open_models
	open_models.close(model_id)

def cmd_stop(ignore=None):
	return 'use "exit" instead of "stop"'

def register():
	cli.register('exit', (), cmd_exit)
	cli.register('open', ([('filename', cli.string_arg)],), cmd_open)
	cli.register('close', ([('model_id', cli.model_id_arg)],), cmd_close)
	cli.register('stop', ([], [('ignore', cli.rest_of_line)]), cmd_stop)
	def lighting_cmds():
		import chimera2.lighting.cmd as cmd
		cmd.register()
	cli.delay_registration('lighting', lighting_cmds)
