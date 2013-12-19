// ----------------------------------------------------------------------------
//
#include <Python.h>			// use PyObject

#include <stdlib.h>			// use atexit()


#include "OVR.h"

using namespace OVR;

class ovr_sensor {
public:
  ovr_sensor() : error(NULL), pManager(0), pHMD(0), pSensor(0), sFusion(0)
  {
    OVR::System::Init();
     
    // *** Initialization - Create the first available HMD Device
    pManager = *DeviceManager::Create();
    pHMD     = *pManager->EnumerateDevices<HMDDevice>().CreateDevice();
    if (!pHMD)
      {
	error = "Failed to create oculus device.";
        return;
      }
    pSensor  = *pHMD->GetSensor();

    sFusion = new SensorFusion();
    if (pSensor)
      sFusion->AttachToSensor(pSensor);
    else
      error = "Failed to attach to oculus sensor.";
  }
  ~ovr_sensor()
  {
    pSensor.Clear();
    pHMD.Clear();
    pManager.Clear();
    if (sFusion)
      delete sFusion;
    OVR::System::Destroy();
  }
  Quatf orientation()
  {
    return sFusion->GetOrientation();
  }
  Ptr<HMDDevice> device() const
  {
    return pHMD;
  }

  const char *error;
private:
  Ptr<DeviceManager> pManager;
  Ptr<HMDDevice>     pHMD;
  Ptr<SensorDevice>  pSensor;
  SensorFusion       *sFusion;
};

static ovr_sensor *ovrs = NULL;

// ----------------------------------------------------------------------------
//
extern "C" PyObject *oculus_state(PyObject *, PyObject *args)
{
  if (!PyArg_ParseTuple(args, const_cast<char *>("")))
    return NULL;

  if (!ovrs)
    {
      PyErr_SetString(PyExc_TypeError, "Must call oculus_connect() before oculus_state()");
      return NULL;
    }

  // Get orientation quaternion to control view
  Quatf q = ovrs->orientation();

  PyObject *s = PyTuple_New(4);
  PyTuple_SetItem(s, 0, PyFloat_FromDouble(q.w));
  PyTuple_SetItem(s, 1, PyFloat_FromDouble(q.x));
  PyTuple_SetItem(s, 2, PyFloat_FromDouble(q.y));
  PyTuple_SetItem(s, 3, PyFloat_FromDouble(q.z));

  return s;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *oculus_parameters(PyObject *, PyObject *args)
{
  if (!PyArg_ParseTuple(args, const_cast<char *>("")))
    return NULL;

  if (!ovrs)
    {
      PyErr_SetString(PyExc_TypeError, "Must call oculus_connect() before oculus_state()");
      return NULL;
    }

  PyObject *params = PyDict_New();

  HMDInfo di;
  ovrs->device()->GetDeviceInfo(&di);

  PyDict_SetItemString(params, "HScreenSize", PyFloat_FromDouble(di.HScreenSize));
  PyDict_SetItemString(params, "VScreenSize", PyFloat_FromDouble(di.VScreenSize));
  PyDict_SetItemString(params, "VScreenCenter", PyFloat_FromDouble(di.VScreenCenter));
  PyDict_SetItemString(params, "EyeToScreenDistance", PyFloat_FromDouble(di.EyeToScreenDistance));
  PyDict_SetItemString(params, "LensSeparationDistance", PyFloat_FromDouble(di.LensSeparationDistance));
  PyDict_SetItemString(params, "InterpupillaryDistance", PyFloat_FromDouble(di.InterpupillaryDistance));
  PyDict_SetItemString(params, "HResolution", PyFloat_FromDouble(di.HResolution));
  PyDict_SetItemString(params, "VResolution", PyFloat_FromDouble(di.VResolution));
  PyObject *dk = PyTuple_New(4);
  for (int i = 0 ; i < 4 ; ++i)
    PyTuple_SetItem(dk, i, PyFloat_FromDouble(di.DistortionK[i]));
  PyDict_SetItemString(params, "DistortionK", dk);

  return params;
}

// ----------------------------------------------------------------------------
//
static void close_device()
{
  if (ovrs)
    {
      delete ovrs;
      ovrs = NULL;
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *oculus_connect(PyObject *, PyObject *args)
{
  if (!PyArg_ParseTuple(args, const_cast<char *>("")))
    return NULL;

  if (ovrs)
    delete ovrs;

  ovrs = new ovr_sensor();
  if (ovrs->error)
    {
      PyErr_SetString(PyExc_TypeError, ovrs->error);
      delete ovrs;
      ovrs = NULL;
      return NULL;
    }

  atexit(close_device);

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static struct PyMethodDef oculus_methods[] =
{
  {const_cast<char*>("connect"), (PyCFunction)oculus_connect, METH_VARARGS, NULL},
  {const_cast<char*>("parameters"), (PyCFunction)oculus_parameters, METH_VARARGS, NULL},
  {const_cast<char*>("state"), (PyCFunction)oculus_state, METH_VARARGS, NULL},
  {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int oculus_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int oculus_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_oculus",
        NULL,
        sizeof(struct module_state),
        oculus_methods,
        NULL,
        oculus_traverse,
        oculus_clear,
        NULL
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
extern "C" PyObject *
PyInit__oculus(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    
    if (module == NULL)
      return NULL;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_oculus.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
