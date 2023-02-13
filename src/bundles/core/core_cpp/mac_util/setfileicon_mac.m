/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

//
// Set the icon image for a file on a Mac HFS+ file system.
//
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

extern "C"
int set_mac_file_icon(const char *file_path_utf8, const char *image_data, Py_ssize_t image_bytes)
{
  //  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  NSString *path = [[NSString alloc] initWithUTF8String:file_path_utf8];
  BOOL set;
  if (image_data == NULL)
    {
      set = [[NSWorkspace sharedWorkspace] setIcon:nil forFile:path options:0];
    }
  else
    {
      NSData *idata = [[NSData alloc] initWithBytes:image_data length:image_bytes];
      NSImage* image = [[NSImage alloc] initWithData:idata];
      set = [[NSWorkspace sharedWorkspace] setIcon:image forFile:path options:0];
      [image release];
      [idata release];
    }
  [path release];
  //    [pool drain];
  return set;
}
