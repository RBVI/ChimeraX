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
# Maintain a cache of data objects using a limited amount of memory.
# The least recently accessed data is released first.
#

# -----------------------------------------------------------------------------
#
class Data_Cache:

  def __init__(self, size):

    self.size = size
    self.used = 0
    self.time = 1
    self.data = {}
    self.groups = {}

  # ---------------------------------------------------------------------------
  #
  def cache_data(self, key, value, size, description, groups = []):

    self.remove_key(key)
    d = Cached_Data(key, value, size, description,
                    self.time_stamp(), groups)
    self.data[key] = d

    for g in groups:
      gtable = self.groups
      if not g in gtable:
        gtable[g] = []
      gtable[g].append(d)

    self.used = self.used + size
    self.reduce_use()

  # ---------------------------------------------------------------------------
  #
  def lookup_data(self, key):

    data = self.data
    if key in data:
      d = data[key]
      d.last_access = self.time_stamp()
      v = d.value
    else:
      v = None
    self.reduce_use()
    return v

  # ---------------------------------------------------------------------------
  #
  def remove_key(self, key):

    data = self.data
    if key in data:
      self.remove_data(data[key])
    self.reduce_use()

  # ---------------------------------------------------------------------------
  #
  def group_keys_and_data(self, group):

    groups = self.groups
    if not group in groups:
      return []

    kd = [(d.key, d.value) for d in groups[group]]
    return kd

  # ---------------------------------------------------------------------------
  #
  def resize(self, size):

    self.size = size
    self.reduce_use()

  # ---------------------------------------------------------------------------
  #
  def reduce_use(self):

    if self.used <= self.size:
      return

    data = self.data
    dlist = list(data.values())
    dlist.sort(key = lambda d: d.last_access)
    import sys
    for d in dlist:
      if sys.getrefcount(d.value) == 2:
        self.remove_data(d)
        if self.used <= self.size:
          break

  # ---------------------------------------------------------------------------
  #
  def remove_data(self, d):

    del self.data[d.key]
    self.used = self.used - d.size
    d.value = None

    for g in d.groups:
      dlist = self.groups[g]
      dlist.remove(d)
      if len(dlist) == 0:
        del self.groups[g]

  # ---------------------------------------------------------------------------
  #
  def time_stamp(self):

    t = self.time
    self.time = t + 1
    return t

# -----------------------------------------------------------------------------
#
class Cached_Data:

  def __init__(self, key, value, size, description, time_stamp, groups):

    self.key = key
    self.value = value
    self.size = size
    self.description = description
    self.last_access = time_stamp
    self.groups = groups
