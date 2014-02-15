//
// Set the icon image for a file on a Mac HFS+ file system.
//
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>

int set_mac_file_icon(const char *file_path_utf8, const char *image_data, int image_bytes)
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
