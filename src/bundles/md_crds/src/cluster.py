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

class ClusterError(ValueError):
    pass

from chimerax.core.state import State
class Clustering(State):
    def __init__(self, frames, representative):
        self.frames = frames
        self.representative = representative

    @staticmethod
    def restore_snapshot(session, data):
        return Clustering(data['frames'], data['representative'])

    def take_snapshot(self, session, flags):
        return {
            'frames': self.frames,
            'representative': self.representative,
        }

def cluster(structure, atoms, frame_nums, *, test_abort=None, status=None):
    num_frames = len(frame_nums)
    if status:
        status("Fetching %d coordinate arrays" % num_frames)
    from chimerax.core.errors import CancelOperation
    crd_arrays = {}
    from time import time
    t0 = time()
    from chimerax.cluster import DistanceMatrix, NMRClust
    with structure.suppress_coordset_change_notifications():
        for i, fn in enumerate(frame_nums):
            structure.active_coordset_id = fn
            crd_arrays[fn] = atoms.coords
            if test_abort and test_abort():
                raise CancelOperation("clustering aborted")
            if status:
                if i == num_frames - 1:
                    status("Fetched %d coordinate arrays" % num_frames)
                elif i % 100 == 99:
                    elapsed = time() - t0
                    per_sec = elapsed / (i+1)
                    remaining = per_sec * (num_frames - (i+1))
                    status("Fetched %d of %d coordinate arrays; About %.1f minutes remaining" % (i+1,
                        num_frames, remaining / 60.0))
        t0 = time()
        total_RMSDs = round(num_frames * (num_frames - 1) / 2)
        if status:
            status("Computing %d RMSDs" % total_RMSDs)
        full_DM = DistanceMatrix(num_frames)
        same_as = {}
        from math import sqrt
        import numpy
        from chimerax.geometry import align_points
        for i, frame1 in enumerate(frame_nums):
            crds1 = crd_arrays[frame1]
            for j, frame2 in enumerate(frame_nums[i+1:]):
                crds2 = crd_arrays[frame2]
                xform, rmsd = align_points(crds1, crds2)
                full_DM.set(i, i+j+1, rmsd)
                if rmsd == 0.0 and frame2 not in same_as:
                    same_as[frame2] = frame1
            if test_abort and test_abort():
                raise CancelOperation("clustering aborted")
            if status:
                num_computed = total_RMSDs - ((num_frames - (i+1)) * (num_frames - (i+2))) / 2
                if num_computed == total_RMSDs:
                    status("Computed %d RMSDs" % total_RMSDs)
                else:
                    elapsed = time() - t0
                    per_sec = elapsed / num_computed
                    remaining = per_sec * (total_RMSDs - num_computed)
                    if remaining < 50:
                        time_est = "%d seconds" % round(remaining)
                    else:
                        time_est = "%.1f minutes" % remaining / 60.0
                    status("Computed %d of %d RMSDs; About %s remaining"
                        % (num_computed, total_RMSDs, time_est))
    if status:
        status("Generating clusters")
    if not same_as:
        dm = full_DM
        reduced_frame_nums = frame_nums
        index_map = range(len(frame_nums))
    elif len(same_as) == num_frames - 1:
        raise ClusterError("All frames to cluster are identical!")
    else:
        dm = DistanceMatrix(num_frames - len(same_as))
        reduced_frame_nums = []
        index_map = []
        for i, fn in enumerate(frame_nums):
            if fn in same_as:
                continue
            reduced_frame_nums.append(fn)
            index_map.append(i)
        for i in range(len(reduced_frame_nums)):
            map_i = index_map[i]
            for j in range(i+1, len(reduced_frame_nums)):
                map_j = index_map[j]
                dm.set(i, j, full_DM.get(map_i, map_j))
    dist_clust = NMRClust(dm)
    # transform result into more easily usable form
    expand_same = {}
    for frame2, frame1 in same_as.items():
        expand_same.setdefault(frame1, []).append(frame2)
    clusterings = []
    for cluster in dist_clust.clusters:
        fns = []
        for index in cluster.members():
            fn = reduced_frame_nums[index]
            fns.append(fn)
            if fn in expand_same:
                fns.extend(expand_same[fn])
        clusterings.append(Clustering(sorted(fns), reduced_frame_nums[dist_clust.representative(cluster)]))

    if status:
        status("Generated clusters")
    return clusterings

def save_clusterings(clusterings, save_file):
    with open(save_file, 'w') as outf:
        for i, clustering in enumerate(clusterings):
            print("Cluster %d: %d frames (%s); representative frame: %d" % (i+1, len(clustering.frames),
                ",".join([str(fn) for fn in clustering.frames]), clustering.representative), file=outf)


