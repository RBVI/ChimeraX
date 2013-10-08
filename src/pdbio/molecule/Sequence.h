#ifndef molecule_Sequence
#define molecule_Sequence

#include <vector>

class Sequence {
public:
	typedef std::vector<unsigned char> Contents;
private:
	Contents  _sequence;

public:
	const Contents&  sequence() const { return _sequence; }
	unsigned char&  operator[](unsigned i) { return _sequence[i]; }
	unsigned char  operator[](unsigned i) const { return _sequence[i]; }
};

#endif  // molecule_Sequence
