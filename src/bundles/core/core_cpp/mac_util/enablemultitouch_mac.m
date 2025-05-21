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

#import <AppKit/AppKit.h>

extern "C"
void enable_mac_multitouch(long nsview_pointer)
{
  NSView *view = (NSView *)(nsview_pointer);
  // TODO This is deprecated in macOS 10.12.  Says to use NSView allowedTouchTypes instead.
  [view setAcceptsTouchEvents:YES];
  // Have not figured out syntax.
  //  [view allowedTouchTypes:NSTouch.TouchTypeMask.direct];
  //  [view allowedTouchTypes:NSNumber(touchType: .direct)];
  //  [view allowedTouchTypes:NSTouch.TouchTypeMask(0)];
  //  view.allowedTouchTypes = [ NSNumber(value:NSTouch.TouchTypeMask.direct.rawValue) ];
}
