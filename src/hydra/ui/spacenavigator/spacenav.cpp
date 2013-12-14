// ----------------------------------------------------------------------------
//
#include <Python.h>			// use PyObject

#include <string.h>			// use strlen(), memcpy()
#include <stdlib.h>			// use atexit()

#include <Carbon/Carbon.h>
#include "ConnexionClientAPI.h"

// ----------------------------------------------------------------------------
// Last reported device state.
//
UInt16 cid = 0;
int state[7];
double event_time = 0;

// ----------------------------------------------------------------------------
//
static void device_cb(io_connect_t, 
		      natural_t messageType, 
		      void *messageArgument)
{
  ConnexionDeviceStatePtr msg = (ConnexionDeviceStatePtr)messageArgument;
    
  if (messageType != kConnexionMsgDeviceState ||
      //      msg->client != cid ||
      (msg->command != kConnexionCmdHandleAxis &&
       msg->command != kConnexionCmdHandleButtons))
    return;

  if (msg->command == kConnexionCmdHandleAxis)
    {
      for (int i = 0 ; i < 6 ; ++i)
	state[i] = msg->axis[i];
      event_time = msg->time;
    }
  else if (msg->command == kConnexionCmdHandleButtons)
    {
      state[6] |= msg->buttons;
      event_time = msg->time;
    }
}

// ----------------------------------------------------------------------------
//
static void close_device()
{
  UnregisterConnexionClient(cid);
  CleanupConnexionHandlers();
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *spacenav_state(PyObject *, PyObject *args)
{
  if (!PyArg_ParseTuple(args, const_cast<char *>("")))
    return NULL;

  bool zero = true;
  for (int i = 0 ; i < 7 && zero ; ++i)
    zero = (zero && (state[i] == 0));

  if (zero)
    {
      Py_INCREF(Py_None);
      return Py_None;
    }

  PyObject *s = PyTuple_New(8);
  for (int i = 0 ; i < 7 ; ++i)
    PyTuple_SetItem(s, i, PyLong_FromLong(state[i]));
  state[6] = 0;	// Clear button press.
  PyTuple_SetItem(s, 7, PyFloat_FromDouble(event_time));

  return s;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *spacenav_connect(PyObject *, PyObject *args)
{
  const char *appsig = NULL;
  if (!PyArg_ParseTuple(args, const_cast<char *>("|s"), &appsig))
    return NULL;

  if (appsig && strlen(appsig) != 4)
    {
      PyErr_SetString(PyExc_TypeError, "Application signature argument must be 4 characters");
      return NULL;
    }

  OSStatus err = InstallConnexionHandlers(device_cb, 0L, 0L);
  if (err)
    {
      PyErr_SetString(PyExc_TypeError, "Installing connexion handler failed");
      return NULL;
    }


  //
  // Using the Chimera Mac bundle signature 'Chim' we don't get any
  // callbacks to the space navigator handler.  If I change
  // chimera/appleevents.py so that it does not activate the X11 app
  // when the Chimera icon is clicked then we do get callbacks but
  // the Chimera application is not active (window not highlighted).
  //
  // UInt32 appsig = 'Chim';
  // UInt32 appsig = '????';	// X11.app signature, doesn't work
  // UInt32 appsig = 'EMAx';	// Emacs signature works.
  // UInt32 appsig = 'Pyth';	// doesn't work.
  // UInt32 appsig = 'OOo2';	// open office, x11 app, works when oo rather than X11 has focus
  //

  UInt32 appsig4 = kConnexionClientWildcard;
  if (appsig)
    memcpy(&appsig4, appsig, 4);

  cid = RegisterConnexionClient(appsig4, 0,
				kConnexionClientModeTakeOver,
				kConnexionMaskAll);
  atexit(close_device);

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static struct PyMethodDef spacenav_methods[] =
{
  {const_cast<char*>("connect"), (PyCFunction)spacenav_connect, METH_VARARGS, NULL},
  {const_cast<char*>("state"), (PyCFunction)spacenav_state, METH_VARARGS, NULL},
  {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int spacenav_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int spacenav_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_spacenavigator",
        NULL,
        sizeof(struct module_state),
        spacenav_methods,
        NULL,
        spacenav_traverse,
        spacenav_clear,
        NULL
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
extern "C" PyObject *
PyInit__spacenavigator(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    
    if (module == NULL)
      return NULL;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_spacenavigator.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
