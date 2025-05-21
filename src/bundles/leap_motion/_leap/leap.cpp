// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
// Pack triangles in Stereo Lithography (STL) file format.
//
#include <iostream>			// for debugging

#include <Windows.h>			// use HANDLE
#include <process.h>			// use _beginthread()

#include <Python.h>			// use PyObject

#include <arrays/pythonarray.h>		// use c_array_to_python()

#include "LeapC.h"			// Leap Motion CLeap SDK, use LEAP_CONNECTION

static const char* result_string(eLeapRS r);

// ----------------------------------------------------------------------------
//
class LastHands
{
public:
  LastHands() : time(0), nhands(0) {}

  LEAP_HAND *hand(int hand)
  {
    eLeapHandType htype = (hand == 0 ? eLeapHandType_Left : eLeapHandType_Right);
    for (uint32_t h = 0 ; h < nhands ; ++h)
      if (hands[h].type == htype)
	return &hands[h];
    return NULL;
  }

  double age()
  {
    int64_t t_now = LeapGetNow(); // microseconds
    return (t_now - time) * 1e-6;
  }
  
  void set(const LEAP_TRACKING_EVENT *event)
  {
    time = event->info.timestamp;
    nhands = event->nHands;
    LEAP_HAND *ehands = event->pHands;
    int nh = (nhands <= max_hands ? nhands : max_hands);
    for (int h = 0 ; h < nh ; ++h)
      memcpy(&hands[h], ehands+h, sizeof(LEAP_HAND));
  }
  
private:
  static const int max_hands = 4;
  int64_t time;
  uint32_t nhands;
  LEAP_HAND hands[max_hands];
};

// ----------------------------------------------------------------------------
//
class LeapPolling
 {
public:
   LeapPolling(LEAP_CONNECTION connection_handle, bool messages = false)
     : _connection_handle(connection_handle), _is_running(false), _messages(messages) {}
  virtual ~LeapPolling() { stop_polling(); }
  void start_polling();
  void stop_polling();
  virtual void new_tracking_event(const LEAP_TRACKING_EVENT *event) {};
  void lock();
  void unlock();
  void poll();

private:
  LEAP_CONNECTION _connection_handle;
  bool _is_running;
  bool _messages;

  // Threading variables
  HANDLE _polling_thread;
  CRITICAL_SECTION _data_lock;
};

// ----------------------------------------------------------------------------
//
class Leap : LeapPolling 
{
public:
  Leap(LEAP_CONNECTION connection, bool messages = false)
    : LeapPolling(connection, messages), connection(connection)
    {
      start_polling();
    }
  ~Leap() { LeapCloseConnection(connection); }

  bool hand_state(int hand, float max_age,
		  float palm_pos[3], float palm_normal[3], float finger_dir[3],
		  float *pinch_strength)
  {
    bool got_state = false;
    lock();
    if (last_hands.age() <= max_age)
      {
	LEAP_HAND *h = last_hands.hand(hand);
	if (h)
	  {
	    LEAP_PALM &palm = h->palm;
	    for (int a = 0 ; a < 3 ; ++a)
	      {
		palm_pos[a] = palm.position.v[a];
		palm_normal[a] = palm.normal.v[a];
		finger_dir[a] = palm.direction.v[a];
	      }
	    *pinch_strength = h->pinch_strength;
	    got_state = true;
	  }
      }
    unlock();

    return got_state;
  }

  virtual void new_tracking_event(const LEAP_TRACKING_EVENT *event)
  {
    last_hands.set(event);
  }

private:
  LEAP_CONNECTION connection;
  LastHands last_hands;
};

extern "C" void service_message_loop(void *leap_polling)
{
  static_cast<LeapPolling * >(leap_polling)->poll();
}

void LeapPolling::start_polling()
{
  _is_running = true;
  InitializeCriticalSection(&_data_lock);
  _polling_thread = (HANDLE)_beginthread(service_message_loop, 0, this);
}

void LeapPolling::stop_polling()
{
  if(!_is_running)
    return;
  _is_running = false;
  WaitForSingleObject(_polling_thread, INFINITE);
  CloseHandle(_polling_thread);
}

void LeapPolling::lock()
{
  EnterCriticalSection(&_data_lock);
}

void LeapPolling::unlock()
{
  LeaveCriticalSection(&_data_lock);
}

void LeapPolling::poll()
{
  eLeapRS result;
  LEAP_CONNECTION_MESSAGE msg;
  while(_is_running)
    {
      unsigned int timeout = 1000;
      result = LeapPollConnection(_connection_handle, timeout, &msg);
      if(result != eLeapRS_Success)
	continue;

      if (msg.type == eLeapEventType_Tracking)
	{
	  lock();
	  new_tracking_event(msg.tracking_event);
	  unlock();
	}
      else if (msg.type == eLeapEventType_LogEvents)
	if (_messages)
	  {
	    const LEAP_LOG_EVENTS *le = msg.log_events;
	    for (uint32_t e = 0 ; e < le->nEvents ; ++e)
	      std::cerr << le->events[e].message << std::endl;
	  }
      else if (msg.type == eLeapEventType_Policy)
	if (_messages)
	  {
	    const LEAP_POLICY_EVENT *pe = msg.policy_event;
	    std::cerr << "leap policy flags = 0x" << std::hex << pe->current_policy << std::endl;
	  }
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
leap_open(PyObject *, PyObject *args, PyObject *keywds)
{
  int head_mounted = 0, messages = 0;
  const char *kwlist[] = {"head_mounted", "debug", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("|bb"),
				   (char **)kwlist,
				   &head_mounted,
				   &messages))
    return NULL;

  LEAP_CONNECTION connection;
  LEAP_CONNECTION_CONFIG *config = NULL;
  eLeapRS result = LeapCreateConnection(config, &connection);
  if (result != eLeapRS_Success)
    return PyErr_Format(PyExc_ConnectionError,
			"leap_open(): Failed to connect, error %s %x",
			result_string(result), (int)result);
  eLeapRS oresult = LeapOpenConnection(connection);
  if (oresult != eLeapRS_Success)
    return PyErr_Format(PyExc_ConnectionError,
			"leap_open(): Failed to open, error %s %x",
			result_string(oresult), (int)oresult);

  uint64_t set_policy = (head_mounted ? eLeapPolicyFlag_OptimizeHMD : 0);
  uint64_t clear_policy = (head_mounted ? 0 : eLeapPolicyFlag_OptimizeHMD);
  eLeapRS presult = LeapSetPolicyFlags(connection, set_policy, clear_policy);
  
  Leap *leap = new Leap(connection, (messages != 0));
  return python_voidp(leap);
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
leap_close(PyObject *, PyObject *args, PyObject *keywds)
{
  void *leap;
  const char *kwlist[] = {"leap", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_voidp, &leap))
    return NULL;

  delete static_cast<Leap *>(leap);
  return python_none();
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
leap_hand_state(PyObject *, PyObject *args, PyObject *keywds)
{
  void *leap;
  int hand;	// 0 = left, 1 = right
  float max_age;
  const char *kwlist[] = {"leap", "hand", "max_age", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&if"),
				   (char **)kwlist,
				   parse_voidp, &leap,
				   &hand,
				   &max_age))
    return NULL;

  Leap *l = static_cast<Leap *>(leap);
  float palm_pos[3], palm_normal[3], finger_dir[3], pinch_strength;

  if (!l->hand_state(hand, max_age, palm_pos, palm_normal, finger_dir, &pinch_strength))
    return python_none();

  return python_tuple(c_array_to_python(palm_pos, 3),
		      c_array_to_python(palm_normal, 3),
		      c_array_to_python(finger_dir, 3),
		      PyFloat_FromDouble(pinch_strength));
}


// ----------------------------------------------------------------------------
// Translates eLeapRS result codes into a human-readable string.
//
static const char* result_string(eLeapRS r)
{
  switch(r){
    case eLeapRS_Success:                  return "Success";
    case eLeapRS_UnknownError:             return "UnknownError";
    case eLeapRS_InvalidArgument:          return "InvalidArgument";
    case eLeapRS_InsufficientResources:    return "InsufficientResources";
    case eLeapRS_InsufficientBuffer:       return "InsufficientBuffer";
    case eLeapRS_Timeout:                  return "Timeout";
    case eLeapRS_NotConnected:             return "NotConnected";
    case eLeapRS_HandshakeIncomplete:      return "HandshakeIncomplete";
    case eLeapRS_BufferSizeOverflow:       return "BufferSizeOverflow";
    case eLeapRS_ProtocolError:            return "ProtocolError";
    case eLeapRS_InvalidClientID:          return "InvalidClientID";
    case eLeapRS_UnexpectedClosed:         return "UnexpectedClosed";
    case eLeapRS_UnknownImageFrameRequest: return "UnknownImageFrameRequest";
    case eLeapRS_UnknownTrackingFrameID:   return "UnknownTrackingFrameID";
    case eLeapRS_RoutineIsNotSeer:         return "RoutineIsNotSeer";
    case eLeapRS_TimestampTooEarly:        return "TimestampTooEarly";
    case eLeapRS_ConcurrentPoll:           return "ConcurrentPoll";
    case eLeapRS_NotAvailable:             return "NotAvailable";
    case eLeapRS_NotStreaming:             return "NotStreaming";
    case eLeapRS_CannotOpenDevice:         return "CannotOpenDevice";
    default:                               return "unknown result type.";
  }
}

// ----------------------------------------------------------------------------
//
static PyMethodDef leap_methods[] = {
  {const_cast<char*>("leap_open"), (PyCFunction)leap_open,
   METH_VARARGS|METH_KEYWORDS,
   "leap_open()\n"
   "\n"
   "Open connection to Leap Motion device, returning connection handle.\n"
   "Implemented in C++.\n"
  },
  {const_cast<char*>("leap_close"), (PyCFunction)leap_close,
   METH_VARARGS|METH_KEYWORDS,
   "leap_close(connection)\n"
   "\n"
   "Close the leap motion connection.\n"
   "Implemented in C++.\n"
  },
  {const_cast<char*>("leap_hand_state"), (PyCFunction)leap_hand_state,
   METH_VARARGS|METH_KEYWORDS,
   "leap_hand_state(connection, hand, max_age)\n"
   "\n"
   "Return hand position info for hand (0 = left, 1 = right):\n"
   "  palm position as length 3 float numpy array)\n"
   "  palm normal vector as length 3 float numpy array)\n"
   "  finger direction vector as length 3 float numpy array)\n"
   "  pinch, whether index finger and thumb touch, boolean\n"
   "If no hand tracking data for last max_age seconds return None.\n"
   "Implemented in C++.\n"
  },
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef leap_def =
{
	PyModuleDef_HEAD_INIT,
	"_leap",
	"Leap Motion access",
	-1,
	leap_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__leap()
{
	return PyModule_Create(&leap_def);
}
