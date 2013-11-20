# -----------------------------------------------------------------------------
#
def cell_size(ss_size, ss_name, size, name, parent_widget):

  allowed_ranges = map(lambda s, ss: ((s+ss)/(ss+1), s/ss), size, ss_size)

  nosize = filter(lambda r: r[1] < r[0], allowed_ranges)
  ones = (allowed_ranges == [(1,1), (1,1), (1,1)])
  if nosize or ones:
    import tkMessageBox
    ss_size_text = '(%d,%d,%d)' % ss_size
    size_text = '(%d,%d,%d)' % size
    msg = ('Data %s of size %s is not a valid size' % (ss_name, ss_size_text) +
           ' for subsamples of %s of size %s' % (name, size_text))
    tkMessageBox.showwarning('Invalid Subsample Data', msg)
    return None
  
  ambiguous = filter(lambda r: r[1] > r[0], allowed_ranges)
  if ambiguous:
    return query_for_cell_size(ss_size, ss_name, size, name, allowed_ranges,
			       parent_widget)

  csize = map(lambda r: r[0], allowed_ranges)
  return tuple(csize)

# -----------------------------------------------------------------------------
#
def query_for_cell_size(ss_size, ss_name, size, name, allowed_step_ranges,
			parent_widget):

  csd = Cell_Size_Dialog(ss_size, ss_name, size, name, allowed_step_ranges)
  csd.run(parent_widget)
  csize = csd.cell_size()

  return csize

# -----------------------------------------------------------------------------
#
from chimera.baseDialog import ModalDialog

class Cell_Size_Dialog(ModalDialog):

  title = 'Subsample Cell Size'
  name = 'subsample cell size'
  buttons = ('Ok',)

  def __init__(self, ss_size, ss_name, size, name, allowed_step_ranges):

    self.subsample_size = ss_size
    self.subsample_name = ss_name
    self.full_data_size = size
    self.full_data_name = name
    self.allowed_step_ranges = allowed_step_ranges
    ModalDialog.__init__(self)

  # ---------------------------------------------------------------------------
  #
  def fillInUI(self, parent):

    subsample_size_text = '(%d,%d,%d)' % self.subsample_size
    size_text = '(%d,%d,%d)' % self.full_data_size
    step_ranges = map(lambda r: '%d-%d' % r, self.allowed_step_ranges)
    allowed_steps = '%s %s %s' % tuple(step_ranges)
    msg = ('There are multiple possible subsample cell sizes for\n' + 
           '%s %s as subsamples of %s %s.\n' % (self.subsample_name,
                                                subsample_size_text,
                                                self.full_data_name,
                                                size_text) +
           'Possibile cell sizes are %s.' % allowed_steps)

    import Tkinter
    lbl = Tkinter.Label(parent, text = msg, justify = 'left')
    lbl.grid(row = 0, column = 0, sticky = 'w')

    csize = '%d %d %d' % tuple(map(lambda r: r[1], self.allowed_step_ranges))
    from CGLtk import Hybrid
    cs = Hybrid.Entry(parent, 'Cell size ', 15, csize)
    cs.frame.grid(row = 1, column = 0, sticky = 'w')
    self.cell_size_var = cs.variable

  # ---------------------------------------------------------------------------
  #
  def Ok(self):

    self.Cancel()

  # ---------------------------------------------------------------------------
  #
  def cell_size(self):

    fields = self.cell_size_var.get().split()
    if len(fields) != 3:
      return None

    try:
      csize = map(int, fields)
    except ValueError:
      return None

    if filter(lambda s: s <= 0, csize):
      return None

    return tuple(csize)
    
# -----------------------------------------------------------------------------
#
from chimera import dialogs
dialogs.register(Cell_Size_Dialog.name, Cell_Size_Dialog, replace = 1)
