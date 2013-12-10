#include "llgr_int.h"

namespace llgr {

Id internal_buffer_id;
AllBuffers all_buffers;

static float identity4x4_data[4][4] = {
	{ 1, 0, 0, 0 },
	{ 0, 1, 0, 0 },
	{ 0, 0, 1, 0 },
	{ 0, 0, 0, 1 }
};

GLenum
cvt_buffer_target(BufferTarget bt)
{
	switch (bt) {
	  case ARRAY: return GL_ARRAY_BUFFER;
	  case ELEMENT_ARRAY: return GL_ELEMENT_ARRAY_BUFFER;
	}
	return 0;
}

void
create_buffer(Id data_id, BufferTarget target, unsigned size, const void *data)
{
	if (!initialized)
		init();
	if (all_buffers.empty()) {
		unsigned char *d = new unsigned char [sizeof identity4x4_data];
		memcpy(d, identity4x4_data, sizeof identity4x4_data);
		all_buffers[0] = BufferInfo(GL_ARRAY_BUFFER, sizeof identity4x4_data, d);
	}
	GLenum gl_target = cvt_buffer_target(target);;
	GLuint buffer;
	auto i = all_buffers.find(data_id);
	if (i == all_buffers.end()) {
		glGenBuffers(1, &buffer);
	} else {
		const BufferInfo &bi = i->second;
		buffer = bi.buffer;
		delete [] bi.data;
	}
	all_buffers[data_id] = BufferInfo(buffer, gl_target);

	glBindBuffer(gl_target, buffer);
	glBufferData(gl_target, size, data, GL_STATIC_DRAW);
	glBindBuffer(gl_target, 0);
}

void
delete_buffer(Id data_id)
{
	auto i = all_buffers.find(data_id);
	if (i != all_buffers.end()) {
		BufferInfo bi = i->second;
		all_buffers.erase(i);
		delete [] bi.data;
		if (bi.buffer)
			glDeleteBuffers(1, &bi.buffer);
	}
}

void
clear_buffers()
{
	AllBuffers save;

	all_buffers.swap(save);
	for (auto& i: save) {
		const BufferInfo &bi = i.second;
		delete [] bi.data;
		if (bi.buffer)
			glDeleteBuffers(1, &bi.buffer);
	}
	internal_buffer_id = 0;
	// clear internal data structures that created buffers
	clear_primitives();
	clear_matrices();
}

void
create_singleton(Id data_id, unsigned size, const void *data)
{
	if (!initialized)
		init();
	if (all_buffers.empty()) {
		unsigned char *d = new unsigned char [sizeof identity4x4_data];
		memcpy(d, identity4x4_data, sizeof identity4x4_data);
		all_buffers[0] = BufferInfo(GL_ARRAY_BUFFER, sizeof identity4x4_data, d);
	}
	auto i = all_buffers.find(data_id);
	if (i != all_buffers.end()) {
		const BufferInfo &bi = i->second;
		delete [] bi.data;
		if (bi.buffer)
			glDeleteBuffers(1, &bi.buffer);
	}
	BufferInfo bi(GL_ARRAY_BUFFER, size, NULL);
	bi.data = new unsigned char [size];
	if (data)
		memcpy(bi.data, data, size);
	all_buffers[data_id] = bi;
}

#if 0
void
create_singleton_index(Id data_id, Id reference_data_id, unsigned offset, unsigned index)
{
	auto j = all_buffers.find(reference_data_id);
	if (j == all_buffers.end())
		return;
	const BufferInfo &rbi = j->second;
	if (rbi.buffer == 0)
		return;
	auto i = all_buffers.find(data_id);
	if (i != all_buffers.end()) {
		const BufferInfo &bi = i->second;
		delete [] bi.data;
		if (bi.buffer)
			glDeleteBuffers(1, &bi.buffer);
	}
	unsigned length = size * data_size(type);
	all_buffers[data_id] = BufferInfo(rbi.buffer, rbi.target, 0, length, type);
}
#endif

#if 0
void
update_buffer(Id data_id, unsigned offset, unsigned size, unsigned stride, unsigned length, DataType type, const void *data)
{
	auto i = all_buffers.find(data_id);
	if (i == all_buffers.end())
		return;
	const BufferInfo &bi = i->second;

	glBindBuffer(bi.target, bi.buffer);
	unsigned char *buf_data = static_cast<unsigned char *>(glMapBuffer(bi.target, GL_WRITE_ONLY));

	// TODO: fill in

	glUnmapBuffer(bi.target);
	glBindBuffer(bi.target, 0);
}
#endif

} // namespace llgr
