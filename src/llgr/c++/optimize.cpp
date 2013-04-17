#include "llgr_int.h"

namespace llgr {

bool dirty;

void
optimize()
{
	dirty = false;
	// TODO: preprocess objects
	//
	// TODO: sort objects by program
	// figure out which objects can be combined for instancing
	// figure out which objects can be combined because they are
	// adjacent indices into same buffers
	// vertex array objects?
}

} // namespace
