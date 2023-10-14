/*
 * === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
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
