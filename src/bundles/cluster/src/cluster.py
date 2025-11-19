# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# This is an implementation of the NMRCLUST algorithm

debug = False

class NMRClust:

	def __init__(self, distance_matrix):
		# distance_matrix should be a map whose key
		# is a two-tuple of indices and whose value
		# is the distance
		from heapq import heappush
		self.dm = distance_matrix
		self.size = distance_matrix.size
		self.clusters = set([])
		cluster = []
		for i in range(self.size):
			c = Cluster(0, i, 1, 0.0)
			cluster.append(c)
			self.clusters.add(c)
		self.distances = []
		self.matrix = {}
		for i in range(self.size):
			for j in range(i + 1, self.size):
				d = distance_matrix.get(i, j)
				k = (cluster[i], cluster[j])
				heappush(self.distances, (d, k))
				self.matrix[k] = d
		step = 0
		self.av_sp = []
		while len(self.clusters) > 1:
			step += 1
			self._reduce(step)
			self._add_avg_spread()
		for c in self.clusters:
			self.root = c
			break
		if self.av_sp:
			# Only cut if there are more than one cluster
			self._cut()

	def _reduce(self, step):
		# Merge two clusters into one and compute the average spread
		from heapq import heappush, heappop
		while True:
			d, (c0, c1) = heappop(self.distances)
			if c0 not in self.clusters or c1 not in self.clusters:
				# One or both clusters already merged
				continue
			self.clusters.remove(c0)
			self.clusters.remove(c1)
			nc = Cluster(step, [c0, c1], c0.size + c1.size, self._spread(c0, c1))
			for c in self.clusters:
				d = self._distance(c, nc)
				k = (c, nc)
				heappush(self.distances, (d, k))
				self.matrix[k] = d
			self.clusters.add(nc)
			break

	def _cluster_distance(self, c0, c1):
		try:
			return self.matrix[(c1, c0)]
		except KeyError:
			return self.matrix[(c0, c1)]

	def _spread(self, c0, c1):
		# Compute the spread when combining two clusters
		w0 = (c0.size * (c0.size - 1) / 2)
		w1 = (c1.size * (c1.size - 1) / 2)
		d = self._cluster_distance(c0, c1)
		w01 = c0.size * c1.size
		spread = ((w0 * c0.spread + w1 * c1.spread + w01 * d)
				/ (w0 + w1 + w01))
		return spread

	def _distance(self, c, nc):
		# Compute the average distance between clusters
		c0, c1 = nc.value
		d0 = self._cluster_distance(c, c0)
		d1 = self._cluster_distance(c, c1)
		w0 = c0.size * c.size
		w1 = c1.size * c.size
		d = (w0 * d0 + w1 * d1) / (w0 + w1)
		return d

	def _add_avg_spread(self):
		cnum = 0
		spread = 0.0
		for c in self.clusters:
			if c.size > 1:
				# Only count if not outlier (cluster of one)
				cnum += 1
				spread += c.spread
		self.av_sp.append(spread / cnum)

	def _cut(self):
		max_av_sp = max(self.av_sp)
		min_av_sp = min(self.av_sp)
		coeff = (self.size - 2) / (max_av_sp - min_av_sp)
		av_sp_norm = []
		penalty = []
		step = 0
		cut_step = None
		min_penalty = None
		for av_sp in self.av_sp:
			step += 1
			nclusi = self.size - step
			avspni = coeff * (av_sp - min_av_sp) + 1
			av_sp_norm.append(avspni)
			p = avspni + nclusi
			penalty.append(p)
			if cut_step is None or p < min_penalty:
				cut_step = step
				min_penalty = p
		self.clusters = set([])
		self._prune(cut_step, self.root)

	def _prune(self, cut_step, c):
		if c.step <= cut_step or c.is_leaf():
			self.clusters.add(c)
		else:
			for c in c.value:
				self._prune(cut_step, c)

	def __str__(self):
		member_list = []
		for c in self.clusters:
			mList = [ str(v) for v in c.members() ]
			member_list.append("{%s}" % ','.join(mList))
		return "[%s]" % ','.join(member_list)

	def representative(self, c):
		members = c.members()
		if len(members) == 1:
			return members[0]
		from distmat import DistanceMatrix
		dm = DistanceMatrix(len(members))
		for i in range(len(members)):
			mi = members[i]
			for j in range(i + 1, len(members)):
				mj = members[j]
				dm.set(i, j, self.dm.get(mi, mj))
		rep = dm.representative()
		return members[rep]


class Cluster:

	def __init__(self, step, value, size, spread):
		self.step = step
		self.value = value
		self.size = size
		self.spread = spread

	def __str__(self):
		if self.is_leaf():
			return str(self.value)
		else:
			return "(%s)" % ','.join([ str(c) for c in self.value ])

	def is_leaf(self):
		return self.size == 1

	def members(self):
		if self.is_leaf():
			return [ self.value ]
		m = []
		for c in self.value:
			m += c.members()
		return m
