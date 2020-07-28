# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

#
# Python/COM code for getting Space Navigator events on Windows.  Derived from
#
#       http://www.3dconnexion.com/forum/viewtopic.php?=&p=5732
#
# Requires comtypes Python module which is a pure Python interface to COM
# on Windows using the ctypes module.  The ctypes module is included in
# Python 2.5.  The comtypes module is not included in the standard Python
# distribution, available our SourceForge.
#
class Space_Device_Win32:

    def __init__(self):

        self.buttons = []
        self.rotation = None
        self.translation = None

        from comtypes import client
        device = client.CreateObject("TDxInput.Device")

        devnames = {
            6:"Space Navigator",
            4:"Space Explorer",
            25:"Space Traveler",
            29:"Space Pilot"
            }
        name = devnames.get(device.Type, 'unknown')

        status = device.Connect()
        if status:
            raise IOError('COM Connection to %s device failed.' % name)

        # Setup button listener
        self.keyboard = device.Keyboard
        self.kev = client.GetEvents(self.keyboard, self)

        # Setup sensor listener
        self.sensor = device.Sensor
        self.sev = client.GetEvents(self.sensor, self)

        # May need PumpEvents() called periodically to process Windows
        #   message queue.
        # client.PumpEvents(timeout = 0.01)

    def cont(self):     # Event callback
        return True
   
    def KeyDown(self, code):     # Event callback
        try:
            name = self.keyboard.GetKeyName(code)
        except Exception:
            name = code
        self.buttons.append(name)

    def KeyUp(self, code):     # Event callback
        pass
   
    def SensorInput(self, inval=None):    #  Event callback
        r = self.sensor.Rotation
        a = r.Angle
        self.rotation = (r.X*a, r.Y*a, r.Z*a)
        
        t = self.sensor.Translation
        self.translation = (t.X, t.Y, t.Z)

    def last_event(self):

        if (len(self.buttons) == 0 and
            self.rotation is None and
            self.translation is None):
            return None

        if self.rotation is None: self.rotation = (0,0,0)
        if self.translation is None: self.translation = (0,0,0)

        s = (self.rotation, self.translation, self.buttons)

        self.buttons = []
        self.rotation = None
        self.translation = None

        return s
