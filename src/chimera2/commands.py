"""
commands -- Default set of commands
===================================

This module implements a default set of cli commands.
After importing this module, :py:func:`register`
must be called to get the commands recognized by the command line interface
(:py:mod:`chimera2.cli`).
Since Python code should call the wrapped functionality directly,
all of the other functions are private.
"""

from . import cli

def _cmd_close(model_id):
	from . import open_models
	try:
		return open_models.close(model_id)
	except ValueError as e:
		raise cli.UserError(e)

def _cmd_exit():
	raise SystemExit(0)

def _cmd_list():
	from . import open_models
	models = open_models.list()
	if len(models) == 0:
		return "No open models."
	info = "Open models:"
	if len(models) > 1:
		info += ", ".join(str(m.id) for m in models[:-1]) + " and"
	info += " %s" % models[-1].id
	return info

def _cmd_open(filename):
	try:
		from chimera2 import io
		return io.open(filename)
	except OSError as e:
		raise cli.UserError(e)

def _cmd_stop(ignore=None):
	raise cli.UserError('use "exit" instead of "stop"')

def register():
	"""Register common cli commands"""
	cli.register('exit', (), _cmd_exit)
	cli.register('open', ([('filename', cli.string_arg)],), _cmd_open)
	cli.register('close', ([('model_id', cli.model_id_arg)],), _cmd_close)
	cli.register('stop', ([], [('ignore', cli.rest_of_line)]), _cmd_stop)
	def lighting_cmds():
		import chimera2.lighting.cmd as cmd
		cmd.register()
	cli.delay_registration('lighting', lighting_cmds)
	cli.register('list', (), _cmd_list)
