"""
commands -- Default set of commands
===================================

This module implements the default set of cli commands
"""

from . import cli

def cmd_close(model_id):
	from . import open_models
	try:
		return open_models.close(model_id)
	except ValueError as e:
		raise cli.UserError(e)

def cmd_exit():
	raise SystemExit(0)

def cmd_list():
	from . import open_models
	models = open_models.list()
	if len(models) == 0:
		return "No open models."
	info = "Open models:"
	if len(models) > 1:
		info += ", ".join(str(m.id) for m in models[:-1]) + " and"
	info += " %s" % models[-1].id
	return info

def cmd_open(filename):
	try:
		from chimera2 import io
		return io.open(filename)
	except OSError as e:
		raise cli.UserError(e)

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
	cli.register('list', (), cmd_list)
