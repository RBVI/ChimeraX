ORIGINAL IMPLEMENTATION
FUNCTIONALITY MOVED INTO __init__.py

"""shed - Chimera2 Tool Shed

The Tool Shed provides an interface for querying available and
out-of-date packages, and for updating, installing and uninstalling
packages while handling inter-package dependencies.

The '''shed''' interface uses '''distlib''' heavily and many of the
objects, both returned and expected, are instances of '''distlib'''
classes.  For example, '''Distribution''' instances from '''distlib'''
are used to represent both available and installed packages; the
'''distlib''' '''Locator''' class is used for finding '''Distribution'''s.
"""

from __future__ import print_function
_ChimeraCore = "shedtest.ChimeraCore"
_ChimeraTools = "shedtest.ChimeraTools"
_ChimeraToolboxes = "shedtest.ChimeraToolboxes"
_ChimeraToolboxPrefix = "shedtest.toolbox"

#
# Init function
#
_chimera_user_base = None
_chimera_user_sitepackages = None
def init():
	"""Initialize and add user Chimera '''site-packages'''
	to '''sys.path''' so that per-users tools may be found."""
	global _chimera_user_base, _chimera_user_sitepackages
	if _chimera_user_base is not None:
		return
	import version	# Chimera version
	version_name = "chimera%d_%d" % (version.get_major_version(),
						version.get_minor_version())
	import site
	userbase = site.getuserbase()
	import os.path
	_chimera_user_base = os.path.join(userbase, version_name)
	_chimera_user_sitepackages = os.path.join(_chimera_user_base,
							"site-packages")
	#DEBUG("_chimera_user_base", _chimera_user_base)
	#DEBUG("_chimera_user_sitepackages", _chimera_user_sitepackages)
	import os
	os.makedirs(_chimera_user_sitepackages, exist_ok=True)
	import site
	site.addsitedir(_chimera_user_sitepackages)
def get_chimera_user_base():
	"""Return path to per-user Chimera directory."""
	return _chimera_user_base
def get_chimera_user_sitepackages():
	"""Return path to per-user Chimera '''site-packages''' directory."""
	return _chimera_user_sitepackages

#
# Chimera locator
#
_repo_locator = None
def repo_get_locator():
	"""Return '''Locator''' for repository."""
	if _repo_locator:
		return _repo_locator
	else:
		return repo_set_locator("http://localhost:8080")
def repo_set_locator(url):
	"""Set Locator for repository."""
	global _repo_locator
	from chimera_locator import ChimeraLocator
	_repo_locator = ChimeraLocator(url)
	return _repo_locator

_repo_distributions = None
def repo_distributions():
	"""Return '''Distribution'''s available from repository."""
	global _repo_locator
	if _repo_distributions is None:
		_repo_distributions = repo_get_locator().get_distributions()
	return _repo_distributions

def _repo_reset():
	global _repo_locator, _repo_distributions
	_repo_locator = None
	_repo_distributions = None

#
# Installed distribution path and locator
#
_inst_path = None
def installed_path():
	"""Return '''DistributionPath''' for installed packages."""
	global _inst_path
	if _inst_path is None:
		from distlib.database import DistributionPath
		_inst_path = DistributionPath()
	return _inst_path

_inst_locator = None
def installed_locator():
	"""Return '''Locator''' for installed packages."""
	global _inst_locator
	if _inst_locator is None:
		from distlib.locators import DistPathLocator
		_inst_locator = DistPathLocator(installed_path())
	return _inst_locator

_inst_chimera_core = None
_inst_tools = None
_inst_toolboxes = None
_inst_core = None
def _installed_init():
	global _inst_chimera_core, _inst_core, _inst_tools, _inst_toolboxes
	# Filter the distribution list to get around
	# bug where some installations have bad metadata
	all_distributions = []
	for d in installed_path().get_distributions():
		try:
			d.run_requires
		except:
			continue
		else:
			all_distributions.append(d)
	# Find only the Chimera distribution packages
	l = installed_locator()
	core = l.locate(_ChimeraCore)
	if core is None:
		_inst_core = set()
		_inst_tools = set()
		_inst_toolboxes = set()
		#raise RuntimeError("cannot find distribution \"%s\"" % _ChimeraCore)
		DEBUG("warning: \"%s\" distribution not found" % _ChimeraCore)
		return
	from distlib.database import make_graph
	dg = make_graph(all_distributions)
	known_dists = set([ core ])
	_inst_chimera_core = core
	_inst_core = set([core])
	_inst_tools = set()
	_inst_toolboxes = set()
	for d, label in dg.adjacency_list[core]:
		_inst_core.add(d)
	check_list = [ core ]
	while check_list:
		for d in dg.reverse_list[check_list.pop()]:
			if d in known_dists:
				continue
			known_dists.add(d)
			check_list.append(d)
			name = d.name
			if name == _ChimeraTools or name == _ChimeraToolboxes:
				continue
			if name.startswith(_ChimeraToolboxPrefix):
				_inst_toolboxes.add(d)
			else:
				_inst_tools.add(d)
def installed_chimera_core():
	"""Return '''Distribution''' instance for installed Chimera core package."""
	global _inst_chimera_core
	if _inst_core is None:		# Check _inst_core because _inst_chimera_core
		_installed_init()	# ... may be legitimately be None
	return _inst_chimera_core
def installed_core():
	"""Return '''Distribution''' instances for installed Chimera core subpackages."""
	global _inst_core
	if _inst_core is None:
		_installed_init()
	return _inst_core
def installed_tools():
	"""Return '''Distribution''' instances for installed Chimera tool packages."""
	global _inst_tools
	if _inst_tools is None:
		_installed_init()
	return _inst_tools
def installed_toolboxes():
	"""Return '''Distribution''' instances for installed Chimera toolbox packages."""
	global _inst_toolboxes
	if _inst_toolboxes is None:
		_installed_init()
	return _inst_toolboxes

def _installed_reset():
	global _inst_path, _inst_locator
	_inst_path = None
	_inst_locator = None
	global _inst_chimera_core, _inst_core, _inst_tools, _inst_toolboxes
	_inst_chimera_core = None
	_inst_core = None
	_inst_tools = None
	_inst_toolboxes = None

#
# Utility functions
#
show = print
def set_show(w):
	"""Set the function to use to display message to user."""
	global show
	old = show
	show = w
	return old

def _no_op(*args, **kw):
	return
DEBUG = _no_op
DEBUG = print

_download_location = None
def download_location():
	"""Return download directory path."""
	global _download_location
	if _download_location is None:
		import tempfile
		_download_location = tempfile.mkdtemp()
		import atexit
		atexit.register(clean_download_location, _download_location)
	return _download_location
def set_download_location(dl):
	"""Set download directory path (directory must exist)."""
	global _download_location
	_download_location = dl
def clean_download_location(location=None):
	"""Remove download directory."""
	if location is None:
		global _download_location
		location = download_location
	if location is not None:
		import shutil
		shutil.rmtree(location, ignore_errors=True)

#
# Internal functions
#
def _remove_distribution(d):
	DEBUG("removing %s (%s)" % (d.name, d.version))
	from distlib.database import InstalledDistribution
	if not isinstance(d, InstalledDistribution):
		raise RuntimeError("trying to remove uninstalled distribution: %s (%s)"
								% (d.name, d.version))
	# HACK ALERT: since there is no API for uninstalling a distribution
	# (as of distlib 0.1.9), here's my hack:
	#   assume that d.list_installed_files() returns paths relative to
	#     undocumented dirname(d.path)
	#   remove all listed installed files while keeping track of
	#     directories from which we removed files
	#   try removing the directories, longest first (this will remove
	#     children directories before parents)
	import os.path
	basedir = os.path.dirname(d.path)
	dircache = set()
	try:
		for path, hash, size in d.list_installed_files():
			p = os.path.join(basedir, path)
			DEBUG("remove", p)
			os.remove(p)
			dircache.add(os.path.dirname(p))
	except OSError as e:
		show("Error:", e)
		return basedir
	try:
		# Do not try to remove the base directory (probably
		# "site-packages somewhere)
		dircache.remove(basedir)
	except KeyError:
		pass
	for d in reversed(sorted(dircache, key=len)):
		try:
			DEBUG("rmdir", d)
			os.rmdir(d)
		except OSError as e:
			# If directory not empty, just ignore
			DEBUG("failed", e)
			pass
	return basedir

def _depends_on(graph, da, db):
	# Returns whether distribution "da" depends on "db"
	# "graph" is a distlib.depgraph.DependencyGraph instance
	# Do depth-first search
	for depa, label in graph.adjacency_list[da]:
		if depa is db or _depends_on(graph, depa, db):
			return True
	return False

#
# Command loop using Chimera2 cli interface
#
_looping = False
from chimera2 import cli

def command_loop(prompt=">"):
	"""Run an command loop using stdin and stdout."""
	import sys
	command = cli.Command()
	global _looping
	_looping = True
	from distlib import DistlibException
	while _looping:
		print(prompt, end=" ")
		sys.stdout.flush()
		line = sys.stdin.readline()
		if not line:
			_looping = False
			print()
			break
		command.parse_text(line.strip(), final=True)
		try:
			command.execute()
		except cli.UserError as e:
			print(command.current_text, file=sys.stderr)
			rest = command.current_text[command.amount_parsed:]
			spaces = len(rest) - len(rest.lstrip())
			error_at = command.amount_parsed + spaces
			print("%s^" % ('.' * error_at), file=sys.stderr)
			print("Error:", str(e), file=sys.stderr)
		except (RuntimeError, DistlibException) as e:
			print("Error:", str(e), file=sys.stderr)
		sys.stderr.flush()

#
# Command callback functions
#   Set global variable _looping to False to terminate loop
#   Raise RuntimeError on errors
#

#
# Command: "exit"
#
@cli.register("exit", cli.CmdInfo())
def exit():
	"""Terminate command loop."""
	global _looping
	_looping = False

#
# Command: "repo"
#

# Example of using enum with numerical values with text labels
T_ALL = 0
T_CORE = 1
T_TOOL = 2
T_TOOLBOX = 3
Target = cli.Enum_of([ T_ALL, T_CORE, T_TOOL, T_TOOLBOX ],
			[ "all", "core", "tool", "toolbox" ],
				name="An installation target class")

@cli.register("repo", cli.CmdInfo(optional=(("target", Target),
					("names", cli.List_of(cli.string_arg)))))
def repo(target=T_ALL, names=[None]):
	"""Query repository for available packages.
	
	''target'' may be ''all'', ''core'', ''tool'' or ''toolbox''.
	''names'' is a list of packages names of interest (strings)."""

	if target in [ T_ALL, T_CORE ]:
		for name in names:
			_repo_target("Core", _ChimeraCore, name)
	if target in [ T_ALL, T_TOOL ]:
		for name in names:
			_repo_target("Tool", _ChimeraTools, name)
	if target in [ T_ALL, T_TOOLBOX ]:
		for name in names:
			_repo_target("Toolbox", _ChimeraToolboxes, name)

def _repo_target(label, container, name):
	l = repo_get_locator()
	d = l.locate(container)
	if d is None:
		raise RuntimeError("cannot find distribution \"%s\"" % container)
	if name is None:
		show("%s container:" % label)
		_repo_distribution_show(container, d)
		show("%s distributions:" % label)
		for name in d.run_requires:
			sd = l.locate(name)
			_repo_distribution_show(name, sd)
	else:
		sd = l.locate(name)
		show("%s distribution \"%s\":" % (label, name))
		_repo_distribution_show(name, sd)

def _repo_distribution_show(name, d):
	if d is None:
		show("Package \"%s\" is missing" % name)
		return
	show("%s:" % d.name_and_version)
	show("\tSummary: %s" % d.metadata.summary)
	show("\tDownload URL: %s" % d.source_url)

#
# Command: "list"
#
@cli.register("list", cli.CmdInfo(optional=(("target", Target),
					("names", cli.List_of(cli.string_arg))),
					keyword=(("full", cli.bool_arg),)))
def list_(target=T_ALL, names=[None], full=False):
	"""Query for installed packages.
	
	''target'' may be ''all'', ''core'', ''tool'' or ''toolbox''.
	''names'' is a list of packages names of interest (strings)."""

	if target in [ T_ALL, T_CORE ]:
		for name in names:
			_list_target("Core", installed_core(), name, full)
	if target in [ T_ALL, T_TOOL ]:
		for name in names:
			_list_target("Tool", installed_tools(), name, full)
	if target in [ T_ALL, T_TOOLBOX ]:
		for name in names:
			_list_target("Toolbox", installed_toolboxes(), name, full)

def _list_target(label, dists, name, full):
	if name is None:
		show("%s distributions:" % label)
		for d in dists:
			_list_distribution_show(d, full)
	else:
		count = 0
		for d in dists:
			if d.name == name:
				_list_distribution_show(name, d, full)
				count += 1
		if count == 0:
			raise RuntimeError("Distribution \"%s\" not found" % name)

def _list_distribution_show(d, full):
	show("%s:" % d.name_and_version)
	show("\tSummary: %s" % d.metadata.summary)
	show("\tDownload URL: %s" % d.source_url)
	if full:
		for path, hash, size in d.list_installed_files():
			show("\t\t%s" % path)

#
# Command: "ood" (out of date)
#
@cli.register("ood", cli.CmdInfo())
def ood():
	"""Check for out-of-date packages."""

	from distlib.version import NormalizedVersion as Version
	l = repo_get_locator()
	available = {}
	for name in l.get_distribution_names():
		version = max([ Version(v) for v in l.get_project(name).keys() ])
		available[name] = version
	l = installed_locator()
	for name in sorted(available.keys()):
		versions = [ Version(v) for v in l.get_project(name).keys() ]
		if not versions:
			# In repo but not installed
			show("%s\tuninstalled (%s available)" % (name, aversion))
			continue
		aversion = available[name]
		iversion = max(versions)
		if aversion == iversion:
			show("%s\tup to date (%s)" % (name, aversion))
		elif aversion < iversion:
			show("%s\tinstalled is newer! (%s > %s)" % (name, iversion, aversion))
		else:
			show("%s\tnew version available (%s < %s)" % (name, iversion, aversion))

#
# Command: "install"
#
@cli.register("install", cli.CmdInfo(optional=(("target", Target),
						("names", cli.List_of(cli.string_arg))),
					keyword=(("always", cli.bool_arg),
						("system", cli.bool_arg))))
def install(target=T_ALL, names=None, always=False, system=False):
	"""Install or update packages.

	''target'' may be ''all'', ''core'', ''tool'' or ''toolbox''.
	''names'' is a list of package names of interest, including version
	specifications as specified in PEP 426.
	''always'' is a boolean indicating whether update should proceed if
	it would result in packages with unsatisfied dependencies."""

	want_update = []
	need_update = []
	if target == T_ALL:
		if names is not None:
			show("named distribution not supported with \"all\" target")
			return
		_install_core(None, want_update)
	elif target == T_CORE:
		if names is not None:
			show("Chimera core can only be updated as a single unit")
			return
		_install_core(None, want_update)
	elif target == T_TOOLBOX:
		if names is None:
			names = [ None ]
		for name in names:
			_install_toolbox(name, want_update)
	elif target == T_TOOL:
		if names is None:
			names = [ None ]
		for name in names:
			_install_tool(name, want_update)
	else:
		show("unknown install target: \"%s\"" % target)
	_install_cascade(want_update, need_update)
	incompatible = _install_check_incompatible(need_update)
	if incompatible and not always:
		return
	_install_wheels(need_update, system)
	reset(repo=False)

def _install_core(name, want):
	if name is None:
		d = _install_distribution(_ChimeraCore)
		if d:
			want.append(d)
	else:
		d = _install_distribution(name)
		if d:
			want.append(d)

def _install_distribution(name):
	repo_dist = repo_get_locator().locate(name)
	if repo_dist is None:
		raise RuntimeError("cannot find new distribution of \"%s\"" % name)
	inst_dist = installed_locator().locate(name)
	if inst_dist is None:
		return repo_dist
	else:
		from distlib.version import NormalizedVersion as Version
		repo_version = Version(repo_dist.version)
		inst_version = Version(inst_dist.version)
		if inst_version < repo_version:
			return repo_dist
		elif inst_version > repo_version:
			show("Warning: installed \"%s\" is newer than latest: %s > %s"
				% inst_dist.version, repo_dist.version)
	return None

def _install_toolbox(name, want):
	if name is None:
		for d in installed_toolboxes():
			_install_toolbox(d.name, want)
	else:
		d = _install_distribution(name)
		if d:
			want.append(d)

def _install_tool(name, want):
	if name is None:
		for d in installed_tools():
			_install_tool(d.name, want)
	else:
		d = _install_distribution(name)
		if d:
			want.append(d)

def _install_cascade(want, need):
	seen = set()
	check = set(want)
	while check:
		d = check.pop()
		seen.add(d)
		need.append(d)
		for req in d.run_requires:
			nd = _install_distribution(req)
			if nd and nd not in seen:
				check.add(nd)

def _install_check_incompatible(need):
	all = dict([ (d.name, d)
		for d in installed_core() | installed_tools() | installed_toolboxes() ])
	all.update([ (d.name, d) for d in need ])
	from distlib.database import make_graph
	graph = make_graph(all.values())
	if graph.missing:
		for d, req_list in graph.missing:
			if len(req_list) == 1:
				s = repr(req_list[0])
			else:
				s = " and ".join(", ".join([ repr(r) for r in req_list[:-1] ]),
							repr(req_list[-1]))
			show("Warning: \"%s\" needs %s" % (d.name, s))
		return True
	else:
		return False

def _install_wheels(need, system):
	# TODO: if a package is being updated, it should be installed in
	# the same location as before

	# Find all packages that should be deleted
	all = dict([ (d.name, d)
		for d in installed_core() | installed_tools() | installed_toolboxes() ])
	from distlib.database import make_graph
	import itertools
	graph = make_graph(itertools.chain(all.values(), need))
	l = need[:]	# what we started with
	ordered = []	# ordered by least dependency
	depend = {}	# dependency relationship cache
	while l:
		for d in l:
			for d2 in l:
				if d2 is d:
					continue
				try:
					dep = depend[(d, d2)]
				except KeyError:
					dep = _depends_on(graph, d, d2)
					depend[(d, d2)] = dep
				if dep:
					break
			else:
				ordered.append(d)
				l.remove(d)
				break
		else:
			# This can only happen if there is circular dependencies
			# in which case we just process the distributions in
			# given order since its no worse than anything else
			ordered.extend(l)
			break
	remove_list = []
	check = set()
	for d in ordered:
		if d in remove_list:
			continue
		try:
			rd = all[d.name]
		except KeyError:
			pass
		else:
			remove_list.append(rd)
			al = graph.adjacency_list[rd]
			if al:
				check.update([ sd for sd, l in al ])
	# Repeatedly go through the list of distributions to see whether
	# they can be removed.  It must be iterative.  Suppose A and B need
	# to be removed; C depends on A; D depends on B and C; if we check D
	# first, it will not be removable since C is not marked for removal
	# yet; but a second pass will show that D is removable.  Iteration
	# ends when no new packages are marked as removable.
	while check:
		any_deletion = False
		new_check = set()
		for d in check:
			for pd in graph.reverse_list[d]:
				if pd not in remove_list:
					new_check.add(d)
					break
			else:
				any_deletion = True
				remove_list.append(d)
				for sd, l in graph.adjacency_list[d]:
					if sd not in remove_list and sd not in check:
						new_check.add(sd)
		if not any_deletion:
			break
		check = new_check
	removed_location = {}
	for d in remove_list:
		removed_location[d.name] = _remove_distribution(d)

	dl = download_location()
	default_paths = _install_make_paths(system)
	from distlib.scripts import ScriptMaker
	maker = ScriptMaker(None, None)
	import os.path
	try:
		from urllib.request import urlretrieve, URLError
	except ImportError:
		from urllib import urlretrieve, URLError
	from distlib.wheel import Wheel
	from distlib import DistlibException
	for d in need:
		try:
			old_location = removed_location[d.name]
		except KeyError:
			paths = default_paths
		else:
			paths = _install_make_paths(system, old_location)
		url = d.source_url
		filename = url.split('/')[-1]
		dloc = os.path.join(dl, filename)
		if not os.path.isfile(dloc):
			try:
				filename, headers = urlretrieve(url, dloc)
			except URLError as e:
				show("Warning: cannot fetch %s: %s" % (url, str(e)))
				continue
		w = Wheel(dloc)
		try:
			w.verify()
		except DistlibExecption as e:
			show("Warning: cannot verify %s: %s" % (d.name, str(e)))
			continue
		show("installing %s (%s)" % (w.name, w.version))
		w.install(paths, maker)

def _install_make_paths(system, sitepackages=None):
	import site, sys, os.path
	if system:
		base = sys.prefix
	else:
		base = _chimera_user_base
	if sitepackages is None:
		if system:
			sitepackages = site.getsitepackages()[-1]
		else:
			sitepackages = _chimera_user_sitepackages
	paths = {
		"prefix": sys.prefix,
		"purelib": sitepackages,
		"platlib": sitepackages,
		"headers": os.path.join(base, "include"),
		"scripts": os.path.join(base, "bin"),
		"data": os.path.join(base, "lib"),
	}
	return paths

#
# Command: "uninstall"
#
@cli.register("uninstall", cli.CmdInfo(optional=(("names", cli.List_of(cli.string_arg)),),
					keyword=(("always", cli.bool_arg),
							("unused", cli.bool_arg))))
def uninstall(names=None, always=False, unused=True):
	"""Remove packages.

	''names'' is a list of package names of interest, including version
	specifications as specified in PEP 426.
	''always'' is a boolean indicating whether update should proceed if
	it would result in packages with unsatisfied dependencies."""

	if names is None:
		raise RuntimeError("no distribution specified for removal")
	remove_set = set()
	l = installed_locator()
	for name in names:
		d = l.locate(name)
		if d is None:
			show("Error: distribution \"%s\" not found" % name)
			raise RuntimeError("Distribution \"%s\" not found" % name)
		remove_set.add(d)

	from distlib.database import make_graph
	all = installed_core() | installed_tools() | installed_toolboxes()
	graph = make_graph(all)

	stranded = set()
	for d in remove_set:
		for pd in graph.reverse_list[d]:
			if pd not in remove_set:
				# Bad case.  pd _will not_ be removed but
				# is depending on d which _will_ be removed
				stranded.add(pd)
	if stranded:
		msg = ("following distributions will have missing requirements: %s"
				% ", ".join([ d.name for d in stranded ]))
		if not always:
			raise RuntimeError(msg)
		else:
			show("Warning:", msg)

	if unused:
		added = True
		while added:
			added = False
			for d in all:
				if d in remove_set:
					# already scheduled for removal
					# leave it alone
					continue
				rl = graph.reverse_list[d]
				if not rl:
					# not referenced at all
					# leave it alone
					continue
				for rd in rl:
					if rd not in remove_set:
						# used by not-removed distribution
						# leave it alone
						break
				else:
					# only referred to by removed distributions
					# get rid of this one too
					remove_set.add(d)
					import sys
					added = True

	for d in remove_set:
		_remove_distribution(d)
	reset(repo=False)

#
# Command: "reset"
#
@cli.register("reset", cli.CmdInfo(keyword=(("repo", cli.bool_arg),
						("installed", cli.bool_arg))))
def reset(repo=True, installed=True):
	"""Reset of repository and installed package lists so that they
	will be reloaded on next access."""

	if repo:
		_repo_reset()
	if installed:
		_installed_reset()

#
# Command: "test"
#
@cli.register("import", cli.CmdInfo(required=(("name", cli.string_arg),)))
def import_(name):
	"""Try importing module or package '''name'''."""

	import importlib
	try:
		m = importlib.import_module(name)
	except ImportError as e:
		show("Warning: import of \"%s\" failed: %s" % (name, str(e)))
	else:
		DEBUG("\"%s\" successfully imported" % name)

if __name__ == "__main__":
	init()
	command_loop()
