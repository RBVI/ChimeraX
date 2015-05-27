// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_destruct
#define basegeom_destruct

#include <set>

namespace basegeom {

class DestructionObserver {
// Base class for classes that are interested in getting only one
// notification once a releted set of destructors have executed
public:
    DestructionObserver();
    virtual ~DestructionObserver();
    virtual void  destructors_done(const std::set<void*>& destroyed) = 0;
};

class DestructionCoordinator {
// Keeps track of what object starts a possible chain of destructor
// calls; when the parent destructor finishes, make callbacks to
// registered functions
    static void*  _destruction_parent;
    static std::set<DestructionObserver*>  _observers;
    static std::set<void*>  _destroyed;
public:
    static void  deregister_observer(DestructionObserver* d_o) {
        _observers.erase(d_o);
    }
    static void*  destruction_parent() { return _destruction_parent; }
    static void  finalizing_destruction(void* instance) {
        if (_destruction_parent == instance) {
            for (auto o: _observers) {
                o->destructors_done(_destroyed);
            }
            _destruction_parent = nullptr;
            _destroyed.clear();
        };
    }
    static void  initiating_destruction(void* instance) {
        if (_destruction_parent == nullptr)
            _destruction_parent = instance;
        _destroyed.insert(instance);
    }
    static void  register_observer(DestructionObserver* d_o) {
        _observers.insert(d_o);
    }
};

class DestructionUser {
// Used in each class destructor to possibly set the instance as the
// "parent" of a chain of destructors (and clear that when the destructor
// scope exits); adds itself to the list of things that got destroyed
    void*  _instance;
public:
    DestructionUser(void* instance): _instance(instance) {
        DestructionCoordinator::initiating_destruction(_instance);
    }
    virtual ~DestructionUser() {
        DestructionCoordinator::finalizing_destruction(_instance);
    }
};

inline DestructionObserver::DestructionObserver()
{
    DestructionCoordinator::register_observer(this);
}

inline DestructionObserver::~DestructionObserver()
{
    DestructionCoordinator::deregister_observer(this);
}

}  // namespace basegeom

#endif  // basegeom_destruct
