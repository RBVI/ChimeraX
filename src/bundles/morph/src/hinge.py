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

def findHinges(rList0, rList1, segments, min_hinge_spacing = 6):
	"""Find hinge residues, which are defined as the interface
	residues between segments that are at least 6 residues long."""
	segmentId = {}
	segId = 1
	for seg in segments:
		for r in seg[0]:
			segmentId[r] = segId
		segId += 1
	hingeIndices = []
	prevId = None
	prevCount = 0
	thisId = None
	thisCount = 0
	for rIndex, r in enumerate(rList0):
		newId = segmentId[r]
		if newId == thisId:
			thisCount += 1
		else:
			# Transition.  Check the previous two segment lengths
			# to see if we found a hinge; then move "this" to "prev"
			# and start a new segment.
			if prevCount >= min_hinge_spacing and thisCount >= min_hinge_spacing:
				hingeIndices.append(rIndex - thisCount)
				prevCount = thisCount
			else:
				prevCount += thisCount
			thisCount = 1
			prevId = thisId
			thisId = newId
	if prevCount >= min_hinge_spacing and thisCount >= min_hinge_spacing:
		hingeIndices.append(len(rList0) - thisCount)
	return hingeIndices

def splitOnHinges(hingeIndices, rList):
	"Split molecule into lists of consecutive residues at hinge residues"
	# Assumes that hinge indices are sorted
	if not hingeIndices:
		return [ rList ]
	else:
		start = 0
		segments = []
		for i in hingeIndices:
			segments.append(rList[start:i])
			start = i
		segments.append(rList[start:])
		return segments
