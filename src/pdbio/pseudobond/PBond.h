// vim: set expandtab ts=4 sw=4:
#ifndef pseudobonds_PBond
#define pseudobonds_PBond

namespace pseudobond {

template <class EndPoint>
class PBond {
// Would like "B" to be "EndPoint" in following friend declaration, but
// currently partially specialized friend declarations are illegal
template<class A, class B> friend class Group;
private:
    EndPoint*  _ends[2];
    PBond(EndPoint* e1, EndPoint *e2) { _ends[0] = e1; _ends[1] = e2; }
};

}  // namespace pseudobond

#endif  // pseudobonds_PBond
