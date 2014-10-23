// ----------------------------------------------------------------------------
//
#include <iostream>			// use std::cerr
#include <Python.h>			// use PyObject
#include <math.h>			// use atan(), M_PI
#include <stdlib.h>			// use atexit()


#include "OVR.h"
#include "../Src/OVR_CAPI_GL.h"

class Oculus {
public:
  Oculus()
  {
    this->in_frame = false;
    ovr_Initialize();

    this->error = NULL;
    this->hmd = ovrHmd_Create(0);	// First available head mounted device.

    if (hmd)
      {
	// Output oculus details.
	std::cerr << "Oculus product name: " << hmd->ProductName
		  << ", firmware version " << hmd->FirmwareMajor << "." << hmd->FirmwareMinor
		  << ", resolution " << hmd->Resolution.w << " " << hmd->Resolution.h;
	const ovrFovPort *fov = hmd->DefaultEyeFov;
	std::cerr << ", field of view, up " << fov[0].UpTan << " down " << fov[0].DownTan
		  << " left " << fov[0].LeftTan << " right " << fov[0].RightTan
		  << std::endl;
      }
    else
      {
	this->error = "No Oculus detected.";
	return;
      }

    unsigned int supportedTrackingCaps = (ovrTrackingCap_Orientation |
					  ovrTrackingCap_MagYawCorrection |
					  ovrTrackingCap_Position);

    unsigned int requiredTrackingCaps = 0;
    // Start the sensor which provides the Rift’s pose and motion.
    ovrBool ok = ovrHmd_ConfigureTracking(hmd, supportedTrackingCaps, requiredTrackingCaps);
    this->error = (ok ? NULL : "HMD lacks required capabilities");

    ovrEyeRenderDesc eyeRenderDesc[2];
    if (!initialize_distortion_rendering(eyeRenderDesc))
      this->error = "Initializing distortion rendering failed.";

    interpupillary_distance = eyeRenderDesc[0].ViewAdjust.x - eyeRenderDesc[1].ViewAdjust.x;

    std::cerr << "Eye render info" << std::endl;
    for (int e = 0 ; e < 2 ; ++e)
      {
	ovrEyeRenderDesc *ed = &eyeRenderDesc[e];
	ovrRecti *er = &ed->DistortedViewport;
	ovrVector3f *va = &ed->ViewAdjust;
	std::cerr << "eye " << e
		  << " tangent half field of view up " << ed->Fov.UpTan
		  << " down " << ed->Fov.DownTan << " left " << ed->Fov.LeftTan << " right " << ed->Fov.RightTan
		  << ", viewport " << er->Pos.x << " " << er->Pos.y << " " << er->Size.w << " " << er->Size.h
		  << " view adjust " << va->x << " " << va->y << " " << va->z
		  << std::endl;
      }
    // TODO: Use field of view and view ports and view matrix translation from eyeRenderDesc.
  }

  ~Oculus()
  {
    if (hmd)
      ovrHmd_Destroy(hmd);
    ovr_Shutdown();
  }     
  bool pose(float xyz[3], int *xyz_valid, float quat[4], int *quat_valid)
  {
    // Query the HMD for the current tracking state.
    // TODO: use ovrHmd_BeginFrame() or ovrHmd_BeginFrameTiming() for second arg to predict frame display time.
    double t;
    if (in_frame)
      t = ovr_GetTimeInSeconds();
    else
      {
	in_frame = true;
	ovrFrameTiming ft = ovrHmd_BeginFrame(hmd, 0);
	t = ft.ScanoutMidpointSeconds;
	// TODO: must call ovrHmd_EndFrame() after rendering is done to accumulate render times.
      }
    ovrTrackingState ts  = ovrHmd_GetTrackingState(hmd, t);
    *xyz_valid = (ts.StatusFlags & ovrStatus_PositionTracked);
    *quat_valid = (ts.StatusFlags & ovrStatus_OrientationTracked);
    if (!*xyz_valid && !*quat_valid)
      return false;

    ovrPoseStatef pose = ts.HeadPose;
    ovrPosef p = pose.ThePose;

    // TODO: ovrHmd_GetEyePose() must be called between ovrHmd_BeginFrameTiming and ovrHmd_EndFrameTiming.
    render_pose[0] = ovrHmd_GetEyePose(hmd, ovrEye_Left);
    render_pose[1] = ovrHmd_GetEyePose(hmd, ovrEye_Right);
    // TODO: report positions for both eyes.

    ovrQuatf q = p.Orientation;	// struct with float x,y,z,w members.
    quat[0] = q.x;
    quat[1] = q.y;
    quat[2] = q.z;
    quat[3] = q.w;

    ovrVector3f pos = p.Position;	// struct with float x,y,z members.
    xyz[0] = pos.x;
    xyz[1] = pos.y;
    xyz[2] = pos.z;

    return true;
  }
  void screen_size(int *w, int *h)
  {
    *w = hmd->Resolution.w;
    *h = hmd->Resolution.h;
  }
  void render_size(int *w, int *h)
  {
    float pixelsPerDisplayPixel = 1.0;	// Quality scaling.
    ovrSizei size_left = ovrHmd_GetFovTextureSize(hmd, ovrEye_Left, hmd->DefaultEyeFov[0], pixelsPerDisplayPixel);
    ovrSizei size_right = ovrHmd_GetFovTextureSize(hmd, ovrEye_Right, hmd->DefaultEyeFov[1], pixelsPerDisplayPixel);
    *w = size_left.w;
    *h = size_left.h;
  }
  void field_of_view(float *up_tan, float *down_tan, float *left_tan, float *right_tan)
  {
    const ovrFovPort *fov = hmd->DefaultEyeFov;
    *up_tan = fov[0].UpTan;
    *down_tan = fov[0].DownTan;
    *left_tan = fov[0].LeftTan;
    *right_tan = fov[0].RightTan;
  }
  bool initialize_distortion_rendering(ovrEyeRenderDesc eyeRenderDesc_out[2])
  {
    // Configure OpenGL.
    ovrGLConfig cfg;
    ovrSizei screen_size;
    screen_size.w = hmd->Resolution.w;
    screen_size.h = hmd->Resolution.h;
    std::cerr << "idr " << hmd->Resolution.w << " " << hmd->Resolution.h << std::endl;
    cfg.OGL.Header.API         = ovrRenderAPI_OpenGL;
    cfg.OGL.Header.RTSize      = screen_size;
    cfg.OGL.Header.Multisample = 0;
    unsigned int distortionCaps = ovrDistortionCap_Chromatic;
    ovrBool result = ovrHmd_ConfigureRendering(hmd, &cfg.Config, distortionCaps,
    					       hmd->DefaultEyeFov, eyeRenderDesc_out);
    return result;
  }
  void distortion_rendering(int texture_width, int texture_height,
			    unsigned int left_texture_id, unsigned int right_texture_id)
  {
    if (!in_frame)
      return;

    ovrSizei tsize;
    tsize.w = texture_width;
    tsize.h = texture_height;

    ovrRecti left_viewport, right_viewport;
    left_viewport.Pos.x = 0;
    left_viewport.Pos.y = 0;
    left_viewport.Size.w = texture_width;
    left_viewport.Size.h = texture_height;
    right_viewport.Pos.x = 0;
    right_viewport.Pos.y = 0;
    right_viewport.Size.w = texture_width;
    right_viewport.Size.h = texture_height;
      
    ovrGLTexture eyeTextures[2];  // Left and right eye.
    eyeTextures[0].OGL.Header.API            = ovrRenderAPI_OpenGL;
    eyeTextures[0].OGL.Header.TextureSize    = tsize;
    eyeTextures[0].OGL.Header.RenderViewport = left_viewport;
    eyeTextures[0].OGL.TexId                 = left_texture_id;
    eyeTextures[1].OGL.Header.API            = ovrRenderAPI_OpenGL;
    eyeTextures[1].OGL.Header.TextureSize    = tsize;
    eyeTextures[1].OGL.Header.RenderViewport = right_viewport;
    eyeTextures[1].OGL.TexId                 = right_texture_id;

    // Let OVR do distortion rendering and swap buffers.
    /*
    std::cerr << "EndFrame "
	      << " " << texture_width << " " << texture_height
	      << " " << left_texture_id << " " << right_texture_id
	      << std::endl;
    */
    ovrHmd_EndFrame(hmd, render_pose, &eyeTextures[0].Texture);
    this->in_frame = false;

    // Dismiss health warning.
    ovrHSWDisplayState hswDisplayState;
    ovrHmd_GetHSWDisplayState(hmd, &hswDisplayState);
    if (hswDisplayState.Displayed)
      ovrHmd_DismissHSWDisplay(hmd);
  }
  float interpupillary_distance;
  const char *error;
private:
  ovrHmd hmd;	// Head mounted device pointer.
  ovrPosef render_pose[2];	// Position and orientation for left and right eyes.
  bool in_frame;
};

static Oculus *ovrs = NULL;

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
  float xyz[3], q[4];
  int xyz_valid, q_valid;
  ovrs->pose(xyz, &xyz_valid, q, &q_valid);

  PyObject *s = PyTuple_New(7);
  if (xyz_valid)
    {
      PyTuple_SetItem(s, 0, PyFloat_FromDouble(xyz[0]));
      PyTuple_SetItem(s, 1, PyFloat_FromDouble(xyz[1]));
      PyTuple_SetItem(s, 2, PyFloat_FromDouble(xyz[2]));
    }
  else
    {
      PyTuple_SetItem(s, 0, Py_None);      Py_INCREF(Py_None);
      PyTuple_SetItem(s, 1, Py_None);      Py_INCREF(Py_None);
      PyTuple_SetItem(s, 2, Py_None);      Py_INCREF(Py_None);
    }
  if (q_valid)
    {
      PyTuple_SetItem(s, 3, PyFloat_FromDouble(q[3]));
      PyTuple_SetItem(s, 4, PyFloat_FromDouble(q[0]));
      PyTuple_SetItem(s, 5, PyFloat_FromDouble(q[1]));
      PyTuple_SetItem(s, 6, PyFloat_FromDouble(q[2]));
    }
  else
    {
      PyTuple_SetItem(s, 3, Py_None);      Py_INCREF(Py_None);
      PyTuple_SetItem(s, 4, Py_None);      Py_INCREF(Py_None);
      PyTuple_SetItem(s, 5, Py_None);      Py_INCREF(Py_None);
      PyTuple_SetItem(s, 6, Py_None);      Py_INCREF(Py_None);
    }

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

  int w, h, tw, th;
  ovrs->screen_size(&w, &h);
  ovrs->render_size(&tw, &th);
  float up_tan, down_tan, left_tan, right_tan;
  ovrs->field_of_view(&up_tan, &down_tan, &left_tan, &right_tan);
  float ipd = ovrs->interpupillary_distance;

  PyDict_SetItemString(params, "width", PyLong_FromLong(w));
  PyDict_SetItemString(params, "height", PyLong_FromLong(h));
  PyDict_SetItemString(params, "texture width", PyLong_FromLong(tw));
  PyDict_SetItemString(params, "texture height", PyLong_FromLong(th));
  PyDict_SetItemString(params, "fov up", PyFloat_FromDouble(up_tan));
  PyDict_SetItemString(params, "fov down", PyFloat_FromDouble(down_tan));
  PyDict_SetItemString(params, "fov left", PyFloat_FromDouble(left_tan));
  PyDict_SetItemString(params, "fov right", PyFloat_FromDouble(right_tan));
  PyDict_SetItemString(params, "interpupillary distance", PyFloat_FromDouble(ipd));

  return params;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *oculus_render(PyObject *, PyObject *args)
{
  int texture_width, texture_height;
  unsigned int left_texture_id, right_texture_id;
  if (!PyArg_ParseTuple(args, const_cast<char *>("iiII"),
			&texture_width, &texture_height,
			&left_texture_id, &right_texture_id))
    return NULL;

  if (!ovrs)
    {
      PyErr_SetString(PyExc_TypeError, "Must call oculus_connect() before oculus_state()");
      return NULL;
    }
  ovrs->distortion_rendering(texture_width, texture_height,
			     left_texture_id, right_texture_id);

  Py_INCREF(Py_None);
  return Py_None;
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

  ovrs = new Oculus();
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
extern "C" PyObject *oculus_disconnect(PyObject *, PyObject *args)
{
  if (!PyArg_ParseTuple(args, const_cast<char *>("")))
    return NULL;

  close_device();

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
static struct PyMethodDef oculus_methods[] =
{
  {const_cast<char*>("connect"), (PyCFunction)oculus_connect, METH_VARARGS, NULL},
  {const_cast<char*>("disconnect"), (PyCFunction)oculus_disconnect, METH_VARARGS, NULL},
  {const_cast<char*>("parameters"), (PyCFunction)oculus_parameters, METH_VARARGS, NULL},
  {const_cast<char*>("state"), (PyCFunction)oculus_state, METH_VARARGS, NULL},
  {const_cast<char*>("render"), (PyCFunction)oculus_render, METH_VARARGS, NULL},
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
