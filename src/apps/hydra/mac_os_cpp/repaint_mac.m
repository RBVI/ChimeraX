//
// Force a window to redraw.  This is to work around a Mac Qt bug where statusbar repaint() does not draw.
//
#import <AppKit/AppKit.h>

void repaint_nsview(unsigned long win)
{
  NSView *view = (NSView *)(win);
  [[view window] display];
}
