# -----------------------------------------------------------------------------
# Save fits in session files.
#
def save_fit_list_state(fit_list, file):

  s = Fit_List_State()
  s.state_from_fit_list(fit_list)
  
  from SessionUtil import objecttree
  t = objecttree.instance_tree_to_basic_tree(s)

  file.write('\n')
  file.write('def restore_fit_list():\n')
  file.write(' fit_list_state = \\\n')
  objecttree.write_basic_tree(t, file, indent = '   ')
  file.write('\n')
  file.write(' try:\n')
  file.write('  from FitMap import session\n')
  file.write('  session.restore_fit_list_state(fit_list_state)\n')
  file.write(' except:\n')
  file.write("  reportRestoreError('Error restoring fit list')\n")
  file.write('\n')
  file.write('registerAfterModelsCB(restore_fit_list)\n')
  file.write('\n')
  
# -----------------------------------------------------------------------------
#
def restore_fit_list_state(fit_list_state):

  classes = (
    Fit_List_State,
    Fit_State,
    )
  name_to_class = {}
  for c in classes:
    name_to_class[c.__name__] = c

  from SessionUtil import objecttree
  s = objecttree.basic_tree_to_instance_tree(fit_list_state, name_to_class)
  import fitlist
  d = fitlist.show_fit_list_dialog()
  s.restore_state(d)
  report_lookup_failures()
  
# -----------------------------------------------------------------------------
#
class Fit_List_State:

  version = 1
  
  state_attributes = ('fits',
                      'smooth_motion',
                      'smooth_steps',
                      'show_clash',
		      'version',
                      )

  # ---------------------------------------------------------------------------
  #
  def state_from_fit_list(self, fit_list):

    d = fit_list
    self.smooth_motion = d.smooth_motion.get()
    self.smooth_steps = d.smooth_steps.get()
    self.show_clash = d.show_clash.get()
    self.fits = [Fit_State(f) for f in d.list_fits]

  # ---------------------------------------------------------------------------
  #
  def restore_state(self, fit_list):

    d = fit_list
    d.smooth_motion.set(self.smooth_motion, invoke_callbacks = False)
    d.smooth_steps.set(self.smooth_steps, invoke_callbacks = False)
    d.show_clash.set(self.show_clash, invoke_callbacks = False)
    d.list_fits = [fs.create_object() for fs in self.fits]
    d.refill_list()

# -----------------------------------------------------------------------------
#
class Fit_State:

  version = 1
  
  state_attributes = ('models',
                      'transforms',
                      'volume',
                      'stats',
		      'version',
                      )

  def __init__(self, fit = None):

    if fit:
      self.state_from_fit(fit)
    
  # ---------------------------------------------------------------------------
  #
  def state_from_fit(self, fit):

    f = fit
    from SimpleSession import sessionID
    mids = [model_id(m) for m in f.models]
    self.models = [mid for mid in mids if not mid is None]
    v = f.volume
    self.volume = model_id(v)
    self.transforms = f.transforms
    self.stats = dict([(k,v) for k,v in f.stats.items()
                       if isinstance(v, (int, float, bool, str))])

  # ---------------------------------------------------------------------------
  #
  def create_object(self):

    models = [id_to_model(m) for m in self.models]
    volume = id_to_model(self.volume)
    
    from search import Fit
    f = Fit(models, self.transforms, volume, self.stats)

    return f

# -----------------------------------------------------------------------------
# Return session id or if not available (e.g. volume data) model id, subid
# and name.
#
def model_id(model):

  if model is None or model.__destroyed__:
    return None

  from SimpleSession import sessionID
  try:
    id = sessionID(model)
  except:
    id = (model.id, model.subid, model.name)
  return id

# -----------------------------------------------------------------------------
#
def id_to_model(id):

  model = None
  if isinstance(id, tuple) and len(id) == 3:
    from SimpleSession import modelMap
    mlist = [m for m in modelMap.get(id[:2], []) if m.name == id[2]]
    if len(mlist) == 1:
      model = mlist[0]
    else:
      global id_to_model_failed
      id_to_model_failed.add(id)
  elif not id is None:
    from SimpleSession import idLookup
    model = idLookup(id)

  return model

# -----------------------------------------------------------------------------
#
id_to_model_failed = set()
def report_lookup_failures():

  if id_to_model_failed:
    ids = '\n'.join(['\tid %d, subid %d, name "%s"' % id for id in id_to_model_failed])
    from chimera.replyobj import info
    info('Session restore failed to find fit models:\n%s\n' % ids)
