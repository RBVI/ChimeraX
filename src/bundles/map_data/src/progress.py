# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Report progress reading or writing a file, plane by plane.
#
class ProgressReporter:

  def __init__(self, operation,
               ijk_size = None,
               element_size = None,
               report_interval = 0.2,           # seconds
               log = None):

    self._log = log

    self.allow_cancel = True
    self.cancel = False
    self.cancel_file = None     # File to close on cancel.
    
    self.operation = operation
    self.report_interval = report_interval
    from time import time
    self.next_time = time() + report_interval
    self.status_shown = False

    self.format = None
    self.ksize = None
    self.array_size(ijk_size, element_size)

  # ---------------------------------------------------------------------------
  #
  def message(self, text):
    log = self._log
    if log:
      log.status(text)

  # ---------------------------------------------------------------------------
  #
  def error(self, text):
    log = self._log
    if log:
      log.error(text)
    else:
      import sys
      sys.stderr.write(text)
      
  # ---------------------------------------------------------------------------
  #
  def show_status(self, text):

#    from chimera import nogui
#    if self.allow_cancel and not nogui:
#      self.message(text, showNow = False)
#      self.message('  cancel', color = 'blue', append = True,
#                   clickCallback = self.cancel_cb)
#    else:
    self.status_shown = True
    self.message(text)

  # ---------------------------------------------------------------------------
  #
  def cancel_cb(self):

    self.cancel = True

  # ---------------------------------------------------------------------------
  #
  def cancel_check(self):

    if self.allow_cancel and self.cancel:
      self.done()
      if self.cancel_file:
        self.cancel_file.close()
      from chimera import CancelOperation
      raise CancelOperation('Cancelled '  + self.operation)
    
  # ---------------------------------------------------------------------------
  #
  def fraction(self, f):

    if self.time_for_status():
      self.cancel_check()
      self.show_status(self.format % (100*f))

  # ---------------------------------------------------------------------------
  #
  def plane(self, k):

    if self.time_for_status():
      self.cancel_check()
      pct = (100.0 * k) / self.ksize
      self.show_status(self.format % pct)

    if k == self.ksize - 1:
      self.done()

  # ---------------------------------------------------------------------------
  #
  def time_for_status(self):

    import time
    t = time.time()
    if t < self.next_time:
      return False

    self.next_time = t + self.report_interval
    return True

  # ---------------------------------------------------------------------------
  #
  def array_size(self, ijk_size, element_size):

    if ijk_size is None:
      self.ksize = None
      self.format = self.message_format(None)
    else:
      isz, jsz, ksz = ijk_size
      self.ksize = ksz
      if not element_size is None:
        bytes = float(isz) * jsz * ksz * element_size
        self.format = self.message_format(bytes)

  # ---------------------------------------------------------------------------
  #
  def text_file_size(self, bytes):

    if bytes > 0:
      self.format = self.message_format(bytes)

  # ---------------------------------------------------------------------------
  #
  def message_format(self, bytes):

    if bytes is None:
      asize = ''
    elif bytes >= 2**30:
      asize = '%.1f Gb' % (float(bytes)/2**30)
    elif bytes >= 10 * 2**20:
      asize = '%.0f Mb' % (float(bytes)/2**20)
    elif bytes >= 2**20:
      asize = '%.1f Mb' % (float(bytes)/2**20)
    else:
      asize = '%.0f Kb' % (float(bytes)/2**10)
    format = '%s %s ' % (self.operation, asize) + '%.0f%%'
    return format
    
  # ---------------------------------------------------------------------------
  #
  def close_on_cancel(self, file):

    self.cancel_file = file
    
  # ---------------------------------------------------------------------------
  #
  def done(self):

    self.allow_cancel = False
    if self.status_shown:
      status = 'Cancelled' if self.cancel else 'Done'
      self.show_status('%s %s' % (status, self.operation))
