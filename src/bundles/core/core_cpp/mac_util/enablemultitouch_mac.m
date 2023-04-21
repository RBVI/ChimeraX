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
