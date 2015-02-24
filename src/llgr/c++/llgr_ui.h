// vi: set expandtab shiftwidth=4 softtabstop=4:
#ifndef LLGR_UI_H
#define LLGR_UI_H

namespace llgr {

enum VSphereCursor {
	Rotation, ZRotation
};

// y coordinate should be inverted -- zero at bottom of window
LLGR_IMEX void vsphere_setup(Id vsphere, double radius, double center[2]);
LLGR_IMEX VSphereCursor vsphere_press(Id vsphere, int x, int y);
LLGR_IMEX VSphereCursor vsphere_drag(Id vsphere, int x, int y, bool throttle, /*OUT*/ double spin_axis[3], /*OUT*/ double *spin_angle);
LLGR_IMEX void vsphere_release(Id vsphere);

// picking
LLGR_IMEX extern Id pick(int x, int y);
#if 0
// TODO
LLGR_IMEX extern Objects pickarea(int x, int y, int width, int height);

// selection highlights
// TODO: more control
// TODO: selected versus movable/active
LLGR_IMEX extern void selection(const Objects& objs);
#endif

} // namespace

#endif
