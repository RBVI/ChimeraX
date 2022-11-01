#include <assert.h>	// use assert
#include <stdio.h>	// use fgets
#include <stdlib.h>	// use malloc
#include <string.h>	// use memset

#define MAX_SEQUENCE_LENGTH 100000
#define MAX_PATH_LENGTH 2048

long num_kmers(int k);
long kmer_counts(FILE *seqs_file, int k, int unique, long kmin, long kmax, unsigned int *counts);
int sequence_length(const char *line);
unsigned short *sequence_entry_sizes(FILE *seqs_file, long nseq);
void file_paths(const char *sequences_path, char *size_path, char *sizes_path,
		char *counts_path, char *seqi_path);
void write_file(const char *path, void *data, size_t num_items, size_t item_size);
void kmer_sequence_indices(FILE *seqs_file, int k, unsigned int *counts,
			   int unique, long kmin, long kmax, unsigned int *seqi, long seqi_len);
const char *next_line(FILE *file);
int sequence_kmers(const char *sequence, int seq_len, int k, int unique,
		   int kmin, int kmax, unsigned int *kmer_indices, int max_indices);

//
// Timing 1 million sequences, takes 10 seconds.  Most of the time is because
// of the random access of arrays (counts, offsets, seq indices, found) that don't fit on
// the CPU cache.  Commenting those out reduces time to 2.5 seconds.
// So the key to optimizing this further would be to avoid random access over larger arrays.
// This C routine is only 7 times faster than Python (70 seconds).
//
void create_index_files(const char *sequences_path, int k, int unique, long max_memory)
{
  long nk = num_kmers(k);
  size_t nbytes = nk * sizeof(unsigned int);
  unsigned int *counts = (unsigned int *) malloc(nbytes);
  memset(counts, 0, nbytes);
  long kmin = 0, kmax = nk;

  FILE *seqs_file = fopen(sequences_path, "rb");
  long num_sequences = kmer_counts(seqs_file, k, unique, kmin, kmax, counts);
  unsigned short *sizes = sequence_entry_sizes(seqs_file, num_sequences);

  char size_path[MAX_PATH_LENGTH], sizes_path[MAX_PATH_LENGTH];
  char counts_path[MAX_PATH_LENGTH], seqi_path[MAX_PATH_LENGTH];
  file_paths(sequences_path, size_path, sizes_path, counts_path, seqi_path);

  char info[256];
  sprintf(info, "{\"k\": %d, \"num_sequences\": %ld}", k, num_sequences);
  write_file(size_path, info, strlen(info), sizeof(char));
  write_file(sizes_path, sizes, num_sequences, sizeof(unsigned short));
  write_file(counts_path, counts, nk, sizeof(unsigned int));

  FILE *seqi_file = fopen(seqi_path, "wb");
  if (max_memory == 0)
    {
      long ni = 0;
      for (long i = 0 ; i < nk  ; ++i)
	ni += counts[i];
      max_memory = ni * sizeof(unsigned int);
    }

  long max_seqi = max_memory / sizeof(unsigned int);
  unsigned int *seqi = (unsigned int *) malloc(max_seqi * sizeof(unsigned int));
  kmin = kmax = 0;
  while (kmax < nk)
    {
      long ni = 0;
      for (kmax = kmin ; kmax < nk && ni + counts[kmax] <= max_seqi ; ++kmax)
	ni += counts[kmax];
      kmer_sequence_indices(seqs_file, k, counts, unique, kmin, kmax, seqi, max_seqi);
      fwrite(seqi, sizeof(unsigned int), ni, seqi_file);
      kmin = kmax;
    }
  fclose(seqi_file);

  fclose(seqs_file);
}

void file_paths(const char *sequences_path, char *size_path, char *sizes_path,
		char *counts_path, char *seqi_path)
{
  // Remove path suffix.
  char basepath[MAX_PATH_LENGTH];
  strcpy(basepath, sequences_path);
  size_t plen = strlen(basepath);
  for (size_t i = plen ; i >= 0 ; --i)
    if (basepath[i] == '.')
      {
	basepath[i] = '\0';
	break;
      }
  // Add new suffixes.
  sprintf(size_path, "%s%s", basepath, ".size");
  sprintf(sizes_path, "%s%s", basepath, ".sizes");
  sprintf(counts_path, "%s%s", basepath, ".counts");
  sprintf(seqi_path, "%s%s", basepath, ".seqs");
}

void write_file(const char *path, void *data, size_t num_items, size_t item_size)
{
  FILE *f = fopen(path, "wb");
  fwrite(data, item_size, num_items, f);
  fclose(f);
}
  
unsigned short *sequence_entry_sizes(FILE *seqs_file, long nseq)
{
  unsigned short *sizes = (unsigned short *) malloc(nseq * sizeof(unsigned short));

  fseek(seqs_file, 0, SEEK_SET);
  for (long i = 0 ; i < nseq ; ++i)
    {
      const char *tline = next_line(seqs_file);
      assert(tline);
      unsigned short tlen = strlen(tline);
      const char *sline = next_line(seqs_file);
      assert(sline);
      unsigned short slen = strlen(sline);
      sizes[i] = tlen + slen;
    }

  return sizes;
}

#define MAX_LINE_LENGTH (MAX_SEQUENCE_LENGTH+1)
const char *next_line(FILE *file)
{
  static char line[MAX_LINE_LENGTH];
  return fgets(line, MAX_LINE_LENGTH, file);
}
  
long kmer_counts(FILE *seqs_file, int k, int unique, long kmin, long kmax, unsigned int *counts)
{
  unsigned int ki[MAX_SEQUENCE_LENGTH];
  int ki_len = MAX_SEQUENCE_LENGTH;
  long nseq = 0;
  fseek(seqs_file, 0, SEEK_SET);
  while (1)
    {
      const char *line = next_line(seqs_file);
      if (!line)
	break;
      if (line[0] == '>')
	continue;
      int nki = sequence_kmers(line, sequence_length(line), k, unique, kmin, kmax, ki, ki_len);
      for (int i = 0 ; i < nki ; ++i)
	counts[ki[i]] += 1;
      nseq += 1;
    }

  return nseq;
}

int sequence_length(const char *line)
{
  int seq_len = strlen(line);
  if (line[seq_len-1] == '\n')
    seq_len -= 1;
  return seq_len;
}

void kmer_sequence_indices(FILE *seqs_file, int k, unsigned int *counts,
			   int unique, long kmin, long kmax, unsigned int *seqi, long seqi_len)
{
  counts += kmin;
  long noff = kmax - kmin;
  unsigned long *offsets = (unsigned long *) malloc(noff * sizeof(unsigned long));
  unsigned long csum = 0;
  for (long i = 0 ; i < noff ; ++i)
    {
      offsets[i] = csum;
      csum += counts[i];
    }
  for (long i = 0 ; i < noff ; ++i)
    counts[i] = 0;
  
  unsigned int ki[MAX_SEQUENCE_LENGTH];
  int ki_len = MAX_SEQUENCE_LENGTH;

  long nseq = 0;
  fseek(seqs_file, 0, SEEK_SET);  
  while (1)
    {
      const char *line = next_line(seqs_file);
      if (!line)
	break;
      if (line[0] == '>')
	continue;
      int nki = sequence_kmers(line, sequence_length(line), k, unique, kmin, kmax, ki, ki_len);
      for (int i = 0 ; i < nki ; ++i)
	{
	  unsigned int kii = ki[i];
	  unsigned long o = offsets[kii] + counts[kii];
	  seqi[o] = nseq;
	  counts[kii] += 1;
	}
      nseq += 1;
    }
}

long num_kmers(int k)
{
  long nk = 1;
  for (int i = 0 ; i < k ; ++i)
    nk *= 20;
  return nk;
}

int sequence_kmers(const char *sequence, int seq_len, int k, int unique,
		   int kmin, int kmax, unsigned int *kmer_indices, int max_indices)
{
  const char *aa = "ACDEFGHIKLMNPQRSTVWY";
  static int *aaindex = NULL;
  static unsigned short *found = NULL;
  static unsigned short fval = -1;
  
  if (aaindex == NULL)
    {
      aaindex = (int *)malloc(4*256);
      for (int i = 0 ; i < 20 ; ++i)
	aaindex[aa[i]] = i;
    }

  if (unique)
    {
      if (found == NULL)
	{
	  long nk = num_kmers(k);
	  found = (unsigned short *) malloc(sizeof(unsigned short)*nk);
	}
      fval += 1;
      if (fval == 0)
	{
	  long nk = num_kmers(k);
	  for (int i = 0 ; i < nk ; ++i)
	    found[i] = 0;
	  fval = 1;
	}
    }
  
  // Sequence with characters replaced by integers 0-19.
  int c = 0;
  int nkmer = seq_len - k + 1;
  for (int i = 0 ; i < nkmer ; ++i)
    {
      const char *si = sequence + i;
      int ki = 0;
      for (int j = 0 ; j < k ; ++j)
	ki = 20*ki + aaindex[si[j]];
      if (ki < kmin || ki >= kmax)
	continue;
      if (unique)
	{
	  if (found[ki] == fval)
	    continue;
	  found[ki] = fval;
	}
      ki -= kmin;
      kmer_indices[c] = ki;
      c += 1;
    }
  return c;
}
