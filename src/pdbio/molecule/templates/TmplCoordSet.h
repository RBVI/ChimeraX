#ifndef templates_TmplCoordSet
#define	templates_TmplCoordSet

#include <vector>
#ifdef UNPORTED
#include "TAexcept.h"
#endif  // UNPORTED
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
#ifdef UNPORTED
	void	removeCoord(TmplCoord *element);
#endif  // UNPORTED
	typedef std::vector<TmplCoord> Coords;
	const Coords	&coords() const { return _coords; }
	const TmplCoord	*find_coord(std::size_t) const;
	TmplCoord	*find_coord(std::size_t);
public:
	int		id() const { return _csid; }
#ifdef UNPORTED
	void		fill(const TmplCoordSet *source);
#endif  // UNPORTED
private:
	int	_csid;
private:
	TmplCoordSet(TmplMolecule *, int key);
#ifdef UNPORTED
	TmplCoordSet(TmplMolecule *, int key, int size);
#endif  // UNPORTED
};

#endif  // templates_TmplCoordSet
