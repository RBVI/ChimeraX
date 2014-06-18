// vim: set expandtab ts=4 sw=4:
#ifndef pseudobonds_Link
#define pseudobonds_Link

namespace pseudobond {

template <typename EP> class Group;

template <class EndPoint>
class Link {
    friend class Group<EndPoint>;
private:
    EndPoint*  _ends[2];
    Link(EndPoint* e1, EndPoint *e2) { _ends[0] = e1; _ends[1] = e2; }
};

}  // namespace pseudobond

#endif  // pseudobonds_Link
