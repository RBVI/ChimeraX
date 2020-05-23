# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def register_remote_control_command(command_name, logger):

    from chimerax.core.commands import CmdDesc, register, BoolArg, StringArg, IntArg, FloatArg
    desc = CmdDesc(
        required = [('enable', BoolArg)],
        keyword = [('address', StringArg),
                   ('port', IntArg),
                   ('timeout', FloatArg)],
        synopsis = 'Allow other processes to send XMLRPC commands to ChimeraX')
    register(command_name, desc, remote_control, logger=logger)

def remote_control(session, enable, address = '127.0.0.1', port = 42184, timeout = 0.1):
    '''
    Start XMLRPC server so Phenix can ask ChimeraX to show models and maps.
    '''
    if enable:
        s = getattr(session, '_xmlrpc_server', None)
        if s is None:
            s = ChimeraxXMLRPCServer(session, address, port, timeout)
            session._xmlrpc_server = s
    else:
        s = getattr(session, '_xmlrpc_server', None)
        if s:
            del session._xmlrpc_server
            s.close()


#
# XML remote procedure call server run by ChimeraX to allow other apps such as
# the Phenix x-ray model building program to use ChimeraX for visualizing results.
#
from xmlrpc.server import SimpleXMLRPCServer
class ChimeraxXMLRPCServer(SimpleXMLRPCServer):

  def __init__ (self, session, address = '127.0.0.1', port = 42184, timeout = 10):

      self.session = session	# ChimeraX Session object

      # start XML-RPC server
      SimpleXMLRPCServer.__init__(self, (address, port), logRequests=0)
      self.socket.settimeout(timeout)
      session.logger.info("xmlrpc server running at %s on port %d" % (address, port))
      
      # Handle RPC calls once every frame draw
      self._handler = session.triggers.add_handler('new frame', self.process_requests)

  def close(self):
      self.session.triggers.remove_handler(self._handler)
      self._handler = None
      self.socket.close()
# Hangs
#      self.shutdown()
      
  def _dispatch (self, method, params) :

      func = getattr(self, method, None)
      if func is None:
          raise NameError('Remote procedure call method name "%s" unknown' % method)
        
      if not hasattr(func, "__call__"):
          raise NameError('Remote procedure call method "%s" is not callable' % method)

      try :
          result = func(*params)
      except Exception as e :
          traceback_str = "\n".join(traceback.format_tb(sys.exc_info()[2]))
          raise RuntimeError('Remote procedure call "%s" generated the following error:\n%s'
                             % (method, traceback_str))

      if result is None:
          result = -1	# RPC cannot return None.

      return result

  def process_requests(self, *args):
      self.handle_request()

  #---------------------------------------------------------------------
  # XML-RPC methods
  #
  def run_command(self, command):
    from chimerax.core.commands import run
    try:
      run(self.session, command)
    except Exceptions as e:
      self.session.info(str(e))
      return False
    return True
    
  def load_phenix_refine_temp_files (self, tmp_dir, run_name) :

    if tmpdir is None or not os.path.isdir(tmp_dir) or run_name is None:
      return False

    from os.path import join, isfile
    pdb_tmp = join(tmp_dir, "tmp.refine.pdb")
    if not isfile(pdb_tmp):
      log = self.session.logger
      log.info("PDB output file not found: %s" % pdb_tmp)
      return False

    # map_tmp = join(tmp_dir, "tmp.refine_maps.mtz")
    map_tmp = join(tmp_dir, "tmp.refine_2mFo-DFc.ccp4")
    if not isfile(map_tmp):
      log = self.session.logger
      log.info("Map output file not found: %s" % map_tmp)
      return False

    m = self.models
    if m:
      self.session.models.close(m)
      self.models = []
    
    from chimerax.core.commands.open import open
    pdb_models = open(self.session, pdb_tmp)
    map_models = open(self.session, map_tmp)
    self.models = pdb_models + map_models

    return True
