#ifndef base_geom_Connectible
#define base_geom_Connectible

#include "Coord.h"
#include <vector>
#include <map>

template <class FinalConnection, class FinalConnectible>
class Connectible {
public:
	typedef std::map<FinalConnectible *, FinalConnection *> ConnectionsMap;
	typedef std::vector<FinalConnection *> Connections;

private:
	ConnectionsMap  _connections;

public:
	void  add_connection(FinalConnection *c) {
		_connections[c->other_end(static_cast<FinalConnectible *>(this))] = c;
	}
	Connections  connections() const;
	const ConnectionsMap &  connections_map() const { return _connections; }
	bool  connects_to(FinalConnectible *c) const {
		return _connections.find(c) != _connections.end();
	}
	virtual const Coord &  coord() const = 0;
	void  remove_connection(FinalConnection *c) {
		_connections.erase(c->other_end(static_cast<FinalConnectible *>(this)));
	}
	virtual void  set_coord(const Point & coord) = 0;
};

template <class FinalConnection, class FinalConnectible>
typename Connectible<FinalConnection, FinalConnectible>::Connections
Connectible<FinalConnection, FinalConnectible>::connections() const
{
	std::vector<FinalConnection *> result;
	result.reserve(_connections.size());
	for (typename ConnectionsMap::const_iterator cmi = _connections.begin();
			cmi != _connections.end(); ++cmi) {
		result.push_back(cmi->second);
	}
	return result;
}

#endif  // base_geom_Connectible
