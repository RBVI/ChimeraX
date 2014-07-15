//
// Enable touch events on an NSView.
//
#import <AppKit/AppKit.h>

void mac_accept_touch_events(void *nsview)
{
  NSView *v = nsview;
  [v setAcceptsTouchEvents:YES];
}
