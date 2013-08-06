# -----------------------------------------------------------------------------
# Volume cache memory use dialog.
#
from chimera.baseDialog import ModelessDialog

# -----------------------------------------------------------------------------
#
class Memory_Use_Dialog(ModelessDialog):

  title = 'Volume Memory Manager'
  name = 'volume memory manager'
  buttons = ('Update', 'Close',)
  help = 'ContributedSoftware/volumeviewer/memory.html'
  
  def fillInUI(self, parent):

    from CGLtk import Hybrid

    parent.columnconfigure(0, weight = 1)
    row = 0

    vl = Hybrid.Scrollable_List(parent, 'Memory use', 5)
    vl.listbox['font'] = 'Courier'	# fixed width font so columns line up
    vl.heading['font'] = 'Courier'
    self.object_listbox = vl.listbox
    self.object_list_heading = vl.heading
    vl.frame.grid(row = row, column = 0, sticky = 'news')
    parent.rowconfigure(row, weight = 1)
    row = row + 1

    self.update_use()

  # ---------------------------------------------------------------------------
  # Override ModelessDialog method.
  #
  def enter(self):

    self.update_use()
    ModelessDialog.enter(self)

  # ---------------------------------------------------------------------------
  #
  def Update(self):

    self.update_use()

  # ---------------------------------------------------------------------------
  #
  def update_use(self):

    listbox = self.object_listbox
    listbox.delete('0', 'end')

    mb = float(2**20)

    dcache = self.data_cache()
    if dcache == None:
      dlist = []
      limit = 0
    else:
      dlist = dcache.data.values()
      dlist.sort(lambda d1, d2: -cmp(d1.size, d2.size))
      import sys
      for d in dlist:
	refs = sys.getrefcount(d.value) - 2
	line = '%8.1f %6d     %s' % (d.size / mb, refs , d.description)
	listbox.insert('end', line)
      limit = dcache.size

    used = reduce(lambda t, d: t + d.size, dlist, 0)
    heading = ('%d objects using %.0f of %.0f Mb' %
	       (len(dlist), used/mb, limit/mb))
    heading = heading + '\nSize (Mb)  In use   Description'
    self.object_list_heading['text'] = heading

  # ---------------------------------------------------------------------------
  #
  def data_cache(self):

    from VolumeData import data_cache
    return data_cache

# -----------------------------------------------------------------------------
#
def show_memory_use_dialog():

  from chimera import dialogs
  return dialogs.display(Memory_Use_Dialog.name)
    
# -----------------------------------------------------------------------------
#
from chimera import dialogs
dialogs.register(Memory_Use_Dialog.name, Memory_Use_Dialog, replace = 1)
