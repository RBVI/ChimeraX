#ifndef templates_TmplCoordSet
#define	templates_TmplCoordSet

#include <vector>
#include "TmplCoord.h"

class TmplMolecule;

class TmplCoordSet {
	friend class TmplMolecule;
	void	operator=(const TmplCoordSet &);	// disable
		TmplCoordSet(const TmplCoordSet &);	// disable
		~TmplCoordSet();
	std::vector<TmplCoord>	_coords;
public:
	void	add_coord(TmplCoord element);
	typedef std::vector<TmplCoord> Coords;
	const Coords	&coords() const { return _coords; }
	const TmplCoord	*find_coord(std::size_t) const;
	TmplCoord	*find_coord(std::size_t);
public:
	int		id() const { return _csid; }
private:
	int	_csid;
private:
	TmplCoordSet(TmplMolecule *, int key);
};

#endif  // templates_TmplCoordSet
