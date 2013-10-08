#include "Chain.h"

void 
Chain::set(unsigned i, Residue *r, char character = -1)
{
	unsigned char c;
	if (character < 0) {
		//TODO: lookup via map in sequence, and also append/delete funcs?
	} else {
		c = (unsigned char)character;
	}
	_residues[i] = r;
	this->[i] = c;
}
