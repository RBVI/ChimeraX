// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_destruct
#define atomstruct_destruct

#include <set>

#include "imex.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX DestructionObserver {
// Base class for classes that are interested in getting only one
// notification once a releted set of destructors have executed
public:
    DestructionObserver();
    virtual ~DestructionObserver();
    virtual void  destructors_done(const std::set<void*>& destroyed) = 0;
};

class ATOMSTRUCT_IMEX DestructionCoordinator {
// Keeps track of what object starts a possible chain of destructor
// calls; when the parent destructor finishes, make callbacks to
// registered functions
// 'parent' is when the destruction initiator should also be destroyed itself,
// 'batcher' when it shouldn't
    static void*  _destruction_batcher;
    static void*  _destruction_parent;
    static std::set<DestructionObserver*>  _observers;
    static std::set<void*>  _destroyed;
    static int _num_notifications_off;
public:
    static void  deregister_observer(DestructionObserver* d_o) {
        _observers.erase(d_o);
    }
    static void*  destruction_parent() { return _destruction_parent; }
    static void  finalizing_destruction(void* instance) {
        bool notification_time = _destruction_batcher == instance
        || (_destruction_batcher == nullptr && _destruction_parent == instance);
        if (notification_time) {
            // both these need to be here in case the observers cause more
            // (recursive) destructions
            _destruction_batcher = nullptr;
            if (_destruction_parent == instance)
                _destruction_parent = nullptr;
            // copy the _destroyed set in case
            // the observers destroy anything
            decltype(_destroyed) destroyed_copy;
            destroyed_copy.swap(_destroyed);
            if (destroyed_copy.size() > 0) {
                auto observers_copy = _observers;
                for (auto o: observers_copy) {
                    if (_observers.find(o) != _observers.end())
                        o->destructors_done(destroyed_copy);
                }
            }
        } else if (_destruction_parent == instance)
            _destruction_parent = nullptr;
    }
    static void  initiating_destruction(void* instance, bool batcher = false) {
        if (batcher) {
            if (_destruction_batcher == nullptr
            && _destruction_parent == nullptr)
                _destruction_batcher = instance;
        } else {
            if (_destruction_parent == nullptr)
                _destruction_parent = instance;
            if (_num_notifications_off == 0)
                _destroyed.insert(instance);
        }
    }
    static void  notifications_off() { _num_notifications_off++; }
    static void  notifications_on() { _num_notifications_off--; }
    static void  register_observer(DestructionObserver* d_o) {
        _observers.insert(d_o);
    }
};

class ATOMSTRUCT_IMEX DestructionBatcher {
// Used when an object will be initiating a chain of sub-object destructions,
// but the object itself is not being destroyed
    void*  _instance;
public:
    DestructionBatcher(void* instance): _instance(instance) {
        DestructionCoordinator::initiating_destruction(_instance, true);
    }
    virtual ~DestructionBatcher() {
        DestructionCoordinator::finalizing_destruction(_instance);
    }
};

class ATOMSTRUCT_IMEX DestructionUser {
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

class ATOMSTRUCT_IMEX DestructionNotificationsOff {
// Used in routines where destruction notifications are not useful,
// such as in code that reads structures and makes temporary items
// that are destroyed before the final structures are delivered
public:
    DestructionNotificationsOff() {
        DestructionCoordinator::notifications_off();
    }
    ~DestructionNotificationsOff() {
        DestructionCoordinator::notifications_on();
    }
};

}  // namespace atomstruct

#endif  // atomstruct_destruct
