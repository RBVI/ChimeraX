/*
 * === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
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
