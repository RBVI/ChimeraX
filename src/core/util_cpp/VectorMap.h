// vi: set expandtab ts=4 sw=4:
#ifndef util_VectorMap
#define util_VectorMap

#include <functional>  // equal_to
#include <utility>  // pair
#include <stdexcept>
#include <vector>

namespace util {

// VectorMap designed for maps where # of entries will always be small
// and memory use is a concern
template <class Key, class Value, class Pred = std::equal_to<Key>>
class VectorMap {
public:
    typedef Key  key_type;
    typedef Value  mapped_type;
    typedef std::pair<Key, Value>  value_type;
    typedef Pred  key_equal;
    typedef Value&  reference;
    typedef const Value&  const_reference;
    typedef typename std::vector<value_type>::iterator  iterator;
    typedef typename std::vector<value_type>::const_iterator  const_iterator;
    typedef typename std::vector<value_type>::size_type  size_type;
    typedef typename std::vector<value_type>::difference_type  difference_type;
private:
    std::vector<value_type>  _map;
public:
    VectorMap(int pre_reserve = 3) { if (pre_reserve) reserve(pre_reserve); }

    mapped_type&  at(const key_type& k);
    const mapped_type&  at(const key_type& k) const;
    iterator  begin() { return _map.begin(); }
    const_iterator  begin() const { return _map.cbegin(); }
    const_iterator  cbegin() const { return _map.cbegin(); }
    const_iterator  cend() const { return _map.cend(); }
    void  clear() { _map.clear(); }
    size_type  count(const key_type& k) const {
        if (find(k) == _map.end()) return 0; return 1;
    }
    // emplace unimplemented
    // emplace_hint unimplemented
    bool  empty() const { return _map.empty(); }
    iterator  end() { return _map.end(); }
    const_iterator  end() const { return _map.cend(); }
    // equal_range unimplemented
    iterator  erase(iterator pos) { return _map.erase(pos); }
    size_type  erase(const key_type& k);
    iterator  erase(iterator first, iterator last) {
        iterator ret;
        for (auto ii = first; ii != last; ++ii) ret = erase(ii);
        return ret;
    }
    iterator  find(const key_type& k);
    const_iterator  find(const key_type& k) const;
    // didn't implement move-semantics version of insert()
    std::pair<iterator,bool>  insert(const value_type& kv);
    template <class InputIter> void  insert(InputIter first, InputIter last) {
        for (auto ii = first; ii != last; ++ii) (void)insert(*ii);
    }
    size_type  max_size() const { return _map.max_size(); }
    mapped_type&  operator[](const key_type& k);
    mapped_type&  operator[](key_type& k) { return (*this)[(const key_type&)k]; }
    void  reserve(size_type n) { _map.reserve(n); }
    size_type  size() const { return _map.size(); }
    void  swap(VectorMap& vm) { _map.swap(vm._map); }
};

// at
template <class K, class V, class P>
typename VectorMap<K, V, P>::mapped_type&
VectorMap<K, V, P>::at(const VectorMap<K, V, P>::key_type& k)
{
    auto mi = find(k);
    if (mi == _map.end())
        throw std::out_of_range("element not in VectorMap");
    return mi->second;
}

// at (const)
template <class K, class V, class P>
const typename VectorMap<K, V, P>::mapped_type&
VectorMap<K, V, P>::at(const VectorMap<K, V, P>::key_type& k) const
{
    auto mi = find(k);
    if (mi == _map.end())
        throw std::out_of_range("element not in VectorMap");
    return mi->second;
}

// erase
template <class K, class V, class P>
typename VectorMap<K, V, P>::size_type
VectorMap<K, V, P>::erase(const VectorMap<K, V, P>::key_type& k)
{
    auto mi = find(k);
    if (mi == _map.end())
        return 0;
    erase(mi);
    return 1;
}

// find
template <class K, class V, class P>
typename VectorMap<K, V, P>::iterator
VectorMap<K, V, P>::find(const VectorMap<K, V, P>::key_type& k)
{
    for (auto mi = _map.begin(); mi != _map.end(); ++mi) {
        if (P()((*mi).first, k))
            return mi;
    }
    return _map.end();
}

// find (const)
template <class K, class V, class P>
typename VectorMap<K, V, P>::const_iterator
VectorMap<K, V, P>::find(const VectorMap<K, V, P>::key_type& k) const
{
    for (auto mi = _map.cbegin(); mi != _map.cend(); ++mi) {
        if (P()((*mi).first, k))
            return mi;
    }
    return _map.cend();
}

// insert
template <class K, class V, class P>
std::pair<typename VectorMap<K, V, P>::iterator,bool>
VectorMap<K, V, P>::insert(const VectorMap<K, V, P>::value_type& kv)
{
    auto mi = find(kv.first);
    if (mi == _map.end()) {
        _map.push_back(kv);
        return std::pair<iterator,bool>(_map.begin()+(_map.size()-1),true);
    }
    return std::pair<iterator,bool>(mi,false);
}

// operator[]
template <class K, class V, class P>
typename VectorMap<K, V, P>::mapped_type&
VectorMap<K, V, P>::operator[](const VectorMap<K, V, P>::key_type& k)
{
    auto mi = find(k);
    if (mi == _map.end()) {
        _map.emplace_back(k, V());
        return _map.back().second;
    }
    return mi->second;
}

}  // namespace util

#endif  // util_VectorMap
