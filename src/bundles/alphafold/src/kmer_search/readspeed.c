#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  const char *path = argv[1];
  FILE *fp = fopen(path, "rb");
  size_t block_size = 1024 * 1024 * 1024;
  void *buffer = malloc(block_size);
  size_t total = 0;
  while (1)
    {
      size_t bytes = fread(buffer, 1, block_size, fp);
      total += bytes;
      if (bytes == 0)
	break;
    }
  fclose(fp);
  float gb = (float)total / (1024 * 1024 * 1024);
  printf("read %zu bytes, %.2f GB\n", total, gb);
  return 0;
}
