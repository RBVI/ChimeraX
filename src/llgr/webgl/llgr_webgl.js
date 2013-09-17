/*
 * Copyright (c) 2013 The Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the University of California, San Francisco.  The name of the
 * University may not be used to endorse or promote products derived
 * from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

var llgr = {};	// only llgr is exported

(function () {
"use strict";

var programs = {};
var buffers = null;
var matrices = {};
var objects = {};

var gl;	// set with set_context()

var internal_buffer_id = 0;	// decrement before using
var current_program = null;

var name_map = {
	position: "position",
	normal: "normal",
};
function attribute_alias(name)
{
	var alias = name_map[name];
	if (alias !== undefined) {
		return alias;
	}
	return name;
}

// primitive caches
function PrimitiveInfo(data_id, index_count, index_id, index_type)
{
	this.data_id = data_id;
	this.index_count = index_count;
	this.index_id = index_id;
	this.index_type = index_type;
}
var proto_spheres = {};
var proto_cylinders = {};

function ShaderProgram(program, vs, fs)
{
	this.program = program;
	this.vs = vs;
	this.fs = fs;
	this.uniforms = {};
	this.attributes = {};
	this.pending_uniforms = [];

	// introspect for uniform/attribute names and locations
	var total = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
	var i, info, name;
	for (i = 0; i < total; ++i) {
		info = gl.getActiveUniform(program, i);
		name = info.name;
		this.uniforms[name] = [gl.getUniformLocation(program, name),
							info.type, info.size];
	}
	total = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
	for (i = 0; i < total; ++i) {
		info = gl.getActiveAttrib(program, i);
		name = info.name;
		this.attributes[name] = [gl.getAttribLocation(program, name),
							info.type, info.size];
	}
	return;
	// debug: print out uniform and attribute locations
	console.log("program uniforms:");
	var u, a;
	for (name in this.uniforms) {
		u = this.uniforms[name];
		console.log(name, u);
	}
	console.log("program attributes:");
	for (name in this.attributes) {
		a = this.attributes[name];
		console.log(name, a);
	}
}

ShaderProgram.prototype.gl_dealloc = function ()
{
	gl.deleteProgram(this.program);
	gl.deleteShader(this.vs);
	gl.deleteShader(this.fs);
};

ShaderProgram.prototype.uniform_location = function(name)
{
	if (!this.uniforms.hasOwnProperty(name))
		return undefined;
	return this.uniforms[name][0];
};

ShaderProgram.prototype.uniform_type = function(name)
{
	if (!this.uniforms.hasOwnProperty(name))
		return undefined;
	return this.uniforms[name][1];
};

ShaderProgram.prototype.uniform_size = function(name)
{
	if (!this.uniforms.hasOwnProperty(name))
		return undefined;
	return this.uniforms[name][2];
};

ShaderProgram.prototype.attribute_location = function(name)
{
	if (!this.attributes.hasOwnProperty(name))
		return undefined;
	return this.attributes[name][0];
};

ShaderProgram.prototype.attribute_type = function(name)
{
	if (!this.attributes.hasOwnProperty(name))
		return undefined;
	return this.attributes[name][1];
};

ShaderProgram.prototype.attribute_size = function(name)
{
	if (!this.attributes.hasOwnProperty(name))
		return undefined;
	return this.attributes[name][2];
};

ShaderProgram.prototype.setup = function ()
{
	gl.useProgram(this.program);
	current_program = this;
	var len = this.pending_uniforms.length;
	for (var i = 0; i < len; ++i) {
		var u = this.pending_uniforms[i];
		var args = [this.uniform_location(u[1])].concat(u.slice(2));
		u[0].apply(gl, args);
	}
	this.pending_uniforms = [];
};

ShaderProgram.prototype.cleanup = function ()
{
	gl.useProgram(null);
	current_program = null;
};

function BufferInfo()
{
	// create BufferInfo object
	if (arguments.length == 2) {
		this.buffer = arguments[0];
		this.target = arguments[1];
		this.size = 0;
		this.data = 0;
	} else if (arguments.length == 3) {
		this.buffer = 0;
		this.target = arguments[0];
		this.size = arguments[1];
		this.data = arguments[2];
	}
	this.offset = 0;	// filled in later
}

function init_buffers()
{
	// buffer zero hold the identity matrix
	var identity = new Float32Array([
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	]);
	buffers = {};
	buffers[0] = new BufferInfo(llgr.ARRAY, identity.byteLength, identity);
}

function MatrixInfo(id, renorm)
{
	// create MatrixInfo object
	this.data_id = id;
	this.renormalize = renorm;
}

function ObjectInfo(program_id, matrix_id, attrinfo, primitive, first, count, index_id, index_type)
{
	// create ObjectInfo object
	if (index_id === undefined) index_id = 0;
	if (index_type === undefined) index_type = llgr.UByte;

	this.program_id = program_id;
	this.matrix_id = matrix_id;
	this.hide = false;
	this.transparent = false;
	this.all_ai = attrinfo;
	this.ptype = primitive;
	this.first = first;
	this.count = count;
	this.index_buffer_id = index_id;
	this.index_buffer_type = index_type;
}

function check_attributes(obj_id, program_id, ai)
{
	if (!(program_id in programs)) {
		console.log("missing program for object " + obj_id);
		return;
	}
	var sp = programs[program_id];
	for (var name in sp.attributes) {
		if (name.lastIndexOf("instanceTransform", 0) == 0)
			continue;
		var found = false;
		for (var i = 0; i < ai.length; ++i) {
			if (ai[i].name == name) {
				found = true;
				break;
			}
		}
		if (!found) {
			console.log("missing attribute " + name
						+ " in object " + obj_id);
		}
	}
}

function cvt_DataType(dt)
{
	switch (dt) {
	  case llgr.Byte: return gl.BYTE;
	  case llgr.UByte: return gl.UNSIGNED_BYTE;
	  case llgr.Short: return gl.SHORT;
	  case llgr.UShort: return gl.UNSIGNED_SHORT;
	  case llgr.Int: return gl.INT;
	  case llgr.UInt: return gl.UNSIGNED_INT;
	  case llgr.Float: return gl.FLOAT;
	  default: return 0;
	}
}

function data_size(type)
{
	switch (type) {
	  case llgr.Byte: case llgr.UByte: return 1;
	  case llgr.Short: case llgr.UShort: return 2;
	  case llgr.Int: case llgr.UInt: return 4;
	  case llgr.Float: return 4;
	  default: return 0;
	}
}

// cache OpenGl state
var enabled = [		// vertex attribute arrays
	false, false, false, false,
	false, false, false, false,
	false, false, false, false,
	false, false, false, false,
];
var enabled_count = [
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0,
];
var enabled_buf = [
	null, null, null, null,
	null, null, null, null,
	null, null, null, null,
	null, null, null, null,
];
var curbuf = {	// currently bound buffer
};

function setup_attribute(sp, ai)
{
	var  bi = buffers[ai.data_id];
	if (bi === undefined)
		return;
	if (!(ai.name in sp.attributes))
		return;
	var loc = sp.attribute_location(ai.name);
	var count, total;
	switch (sp.attribute_type(ai.name)) {
	  case gl.FLOAT: case gl.INT: case gl.BOOL:
		total = count = 1; break;
	  case gl.FLOAT_VEC2: case gl.INT_VEC2: case gl.BOOL_VEC2:
		total = count = 2; break;
	  case gl.FLOAT_VEC3: case gl.INT_VEC3: case gl.BOOL_VEC3:
		total = count = 3; break;
	  case gl.FLOAT_VEC4: case gl.INT_VEC4: case gl.BOOL_VEC4:
		total = count = 4; break;
	  case gl.FLOAT_MAT2:
		total = 4; count = 2; break;
	  case gl.FLOAT_MAT3:
		total = 9; count = 3; break;
	  case gl.FLOAT_MAT4:
		total = 16; count = 4; break;
	// TODO: gl.SAMPLER_1D gl.SAMPLER_2D gl.SAMPLER_3D
	// gl.SAMPLER_CUBE
	// gl.SAMPLER_1D_SHADOW gl.SAMPLER_2D_SHADOW
	  default:
		total = count = 1; break;
	}

	if (bi.buffer) {
		if (bi.buffer != curbuf[bi.target]) {
			gl.bindBuffer(bi.target, bi.buffer);
			curbuf[bi.target] = bi.buffer;
		}
		// TODO: handle total > count -- arrays of matrices
		gl.vertexAttribPointer(loc, ai.count,
			cvt_DataType(ai.type), ai.normalized, ai.stride,
			ai.offset);
		if (!enabled[loc]) {
			gl.enableVertexAttribArray(loc);
			enabled[loc] = true;
			enabled_count[loc] = 0;
			enabled_buf[loc] = null;
		}
		return;
	}

	// attribute doesn't change per-vertex
	if (ai.type != llgr.Float) {
		// so far, WebGL only supports Float attributes
		return;
	}
	if (enabled_count[loc] === count && enabled_buf[loc] === bi.data) {
		return;
	}
	enabled_count[loc] = count;
	enabled_buf[loc] = bi.data;
	var bfa = new Float32Array(bi.data, 0);
	var offset = 0;
	while (total > 0) {
		if (enabled[loc]) {
			gl.disableVertexAttribArray(loc);
			enabled[loc] = false;
		}
		var fa = bfa.subarray(offset, offset + count);
		switch (count) {
		  case 1: gl.vertexAttrib1fv(loc, fa); break;
		  case 2: gl.vertexAttrib2fv(loc, fa); break;
		  case 3: gl.vertexAttrib3fv(loc, fa); break;
		  case 4: gl.vertexAttrib4fv(loc, fa); break;
		}
		//offset += count * data_size(ai.type);
		offset += count;
		total -= count;
		loc += 1;
	}
}

function convert_data(data)
{
	if (data instanceof ArrayBuffer
	|| data instanceof DataView
	|| data instanceof Int8Array
	|| data instanceof Uint8Array
	|| data instanceof Int16Array
	|| data instanceof Uint16Array
	|| data instanceof Int32Array
	|| data instanceof Uint32Array
	|| data instanceof Float32Array) {
		return data;
	}
	var little_endian, size, words;
	//[little_endian, size, words] = data;
	little_endian = data[0];
	size = data[1];
	words = data[2];
	data = new ArrayBuffer(size);
	var view = new DataView(data, 0);
	var i = 0, w = 0;
	while (size >= 4) {
		view.setUint32(i, words[w], little_endian);
		i += 4;
		size -= 4;
		++w;
	}
	if (size >= 2) {
		view.setUint16(i, words[w], little_endian);
		i += 2;
		size -= 2;
		++w;
	}
	if (size >= 1) {
		view.setUint8(i, words[w], little_endian);
		i += 1;
		size -= 1;
	}
	return data;
}

function build_sphere(num_vertices)
{
	var bands = Math.round(Math.sqrt(num_vertices));
	if (bands < 4)
		bands = 4;
	var spokes = Math.round(num_vertices / bands);
	if (spokes < 4)
		spokes = 4;

	// from http://learningwebgl.com/cookbook/index.php/How_to_draw_a_sphere
	var np = [];	// interleaved normal & position
	var indices = [];
	for (var i = 0; i <= bands; ++i) {
		var theta = i * Math.PI / bands;
		var sin_theta = Math.sin(theta);
		var cos_theta = Math.cos(theta);
		for (var j = 0; j <= spokes; ++j) {
			var phi = j * 2 * Math.PI / spokes;
			var sin_phi = Math.sin(phi);
			var cos_phi = Math.cos(phi);

			var x = cos_phi * sin_theta;
			var y = cos_theta;
			var z = sin_phi * sin_theta;

			// normal
			np.push(x);
			np.push(y);
			np.push(z);

			// position
			np.push(x);
			np.push(y);
			np.push(z);

			// indices
			if ((i < bands) && (j < spokes)) {
				var first = (i * (spokes + 1)) + j;
				var second = first + spokes + 1;

				indices.push(first);
				indices.push(first + 1);
				indices.push(second);

				indices.push(second);
				indices.push(first + 1);
				indices.push(second + 1);
			}
		}
	}

	var np_id = --internal_buffer_id;
	var index_id = --internal_buffer_id;
	llgr.create_buffer(np_id, llgr.ARRAY, new Float32Array(np));
	llgr.create_buffer(index_id, llgr.ELEMENT_ARRAY,
						new Uint16Array(indices));
	proto_spheres[num_vertices] = new PrimitiveInfo(np_id, indices.length,
						index_id, llgr.UShort);
}

function build_cylinder(num_spokes, bottom, top)
{
	bottom = bottom || false;
	top = top || false;
	// TODO: add optional bottom and top caps

	var np = new Float32Array(12 * num_spokes);	// normal, position
	var num_indices = num_spokes * 2 + 2;
	var indices, index_type;
	if (num_indices < 256) {
		indices = new Uint8Array(num_indices);
		index_type = llgr.UByte;
	} else if (num_indices < 65536) {
		indices = new Uint16Array(num_indices);
		index_type = llgr.UShort;
	} else {
		// needs OES_element_index_uint extension
		indices = new Uint32Array(num_indices);
		index_type = llgr.UInt;
	}
	for (var i = 0; i < num_spokes; ++i) {
		var theta = 2 * Math.PI * i / num_spokes;
		var x = Math.cos(theta);
		var z = Math.sin(theta);
		var o = i * 6;
		np[o + 0] = x;
		np[o + 1] = 0;
		np[o + 2] = z;
		np[o + 3] = x;
		np[o + 4] = -1;
		np[o + 5] = z;
		var o2 = (i + num_spokes) * 6;
		np[o2 + 0] = np[o + 0];
		np[o2 + 1] = np[o + 1];
		np[o2 + 2] = np[o + 2];
		np[o2 + 3] = np[o + 3];
		np[o2 + 4] = 1;
		np[o2 + 5] = np[o + 5];
		indices[i * 2 + 0] = i;
		indices[i * 2 + 1] = i + num_spokes;
	}
	indices[num_spokes * 2 + 0] = 0;
	indices[num_spokes * 2 + 1] = num_spokes;

	var np_id = --internal_buffer_id;
	var index_id = --internal_buffer_id;
	llgr.create_buffer(np_id, llgr.ARRAY, np);
	llgr.create_buffer(index_id, llgr.ELEMENT_ARRAY, indices);
	proto_cylinders[num_spokes] = new PrimitiveInfo(np_id, num_indices,
							index_id, index_type);
}

llgr = {
	set_context: function(context) {
		gl = context;
	},

	// enum DataType
	Byte: 0,
	UByte: 1,
	Short: 2,
	UShort: 3,
	Int: 4,
	UInt: 5,
	Float: 6,
	// enum ShaderType
	IVec1: 0,
	IVec2: 1,
	IVec3: 2,
	IVec4: 3,
	UVec1: 4,		// OpenGL ES 3 placeholder
	UVec2: 5,		// ditto
	UVec3: 6,		// ditto
	UVec4: 7,		// ditto
	FVec1: 8,
	FVec2: 9,
	FVec3: 10,
	FVec4: 11,
	Mat2x2: 12,
	Mat3x3: 13,
	Mat4x4: 14,
	Mat2x3: 15,		// ditto
	Mat3x2: 16,		// ditto
	Mat2x4: 17,		// ditto
	Mat4x2: 18,		// ditto
	Mat3x4: 19,		// ditto
	Mat4x3: 20,		// ditto
	// enum BufferTarget
	ARRAY: 0x8892,		// same as GL_ARRAY_BUFFER
	ELEMENT_ARRAY: 0x8893,	// same as GL_ELEMENT_ARRAY_BUFFER
	// enum PrimitiveType
	Points: 0,
	Lines: 1,
	Line_loop: 2,
	Line_strip: 3,
	Triangles: 4,
	Triangle_strip: 5,
	Triangle_fan: 6,

	AttributeInfo: function (name, data_id, offset, stride, count,
							data_type, normalized) {
		if (normalized === undefined) normalized = false;

		this.name = name;
		this.data_id = data_id;
		this.offset = offset;
		this.stride = stride;
		this.count = count;
		this.type = data_type;
		this.normalized = normalized;
	},

	create_program: function (program_id, vert_shader, frag_shader) {
		if (program_id <= 0) {
			throw "need positive program id";
		}
		if (program_id in programs) {
			programs[program_id].gl_dealloc();
			delete programs[program_id];
		}
		var vs = gl.createShader(gl.VERTEX_SHADER);
		var fs = gl.createShader(gl.FRAGMENT_SHADER);
		gl.shaderSource(vs, vert_shader);
		gl.shaderSource(fs, frag_shader);
		gl.compileShader(vs);
		gl.compileShader(fs);
		if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
			console.log(gl.getShaderInfoLog(vs));
			gl.deleteShader(vs);
			gl.deleteShader(fs);
			return;
		}
		if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
			console.log(gl.getShaderInfoLog(fs));
			gl.deleteShader(vs);
			gl.deleteShader(fs);
			return;
		}
		var program = gl.createProgram();
		gl.attachShader(program, vs);
		gl.attachShader(program, fs);
		// bind to 0 for efficient desktop emulation
		// TODO: allow for other name than position
		gl.bindAttribLocation(program, 0, "position");
		gl.linkProgram(program);
		if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
			console.log(gl.getProgramInfoLog(program));
			gl.deleteProgram(program);
			gl.deleteShader(vs);
			gl.deleteShader(fs);
			return;
		}

		programs[program_id] = new ShaderProgram(program, vs, fs);
	},
	delete_program: function (program_id) {
		if (program_id in programs) {
			sp = programs[program_id];
			if (sp === current_program) {
				current_program = null;
				gl.useProgram(0);
			}
			sp.gl_dealloc();
			delete programs[program_id];
		}
	},
	clear_programs: function () {
		current_program = null;
		gl.useProgram(null);
		for (var pid in programs) {
			var sp = programs[pid];
			sp.gl_dealloc();
		}
		programs = {};
	},

	set_uniform: function (program_id, name, shader_type, data) {
		// defer setting uniform until program is in use
		data = convert_data(data);
		var fa = new Float32Array(data, 0);
		var ia = new Int32Array(data, 0);
		var u;
		switch (shader_type) {
		  case llgr.FVec1: u = [gl.uniform1fv, name, fa]; break;
		  case llgr.FVec2: u = [gl.uniform2fv, name, fa]; break;
		  case llgr.FVec3: u = [gl.uniform3fv, name, fa]; break;
		  case llgr.FVec4: u = [gl.uniform4fv, name, fa]; break;
		  case llgr.IVec1: u = [gl.uniform1iv, name, ia]; break;
		  case llgr.IVec2: u = [gl.uniform2iv, name, ia]; break;
		  case llgr.IVec3: u = [gl.uniform3iv, name, ia]; break;
		  case llgr.IVec4: u = [gl.uniform4iv, name, ia]; break;
		  case llgr.Mat2x2:
			   u = [gl.uniformMatrix2fv, name, false, fa]; break;
		  case llgr.Mat3x3:
			   u = [gl.uniformMatrix3fv, name, false, fa]; break;
		  case llgr.Mat4x4:
			   u = [gl.uniformMatrix4fv, name, false, fa]; break;
		  default:
			console.log('unknown uniform shader type for ' + name
							+ ': ' + shader_type);
			return;
		}
		var all_programs;
		if (program_id) {
			all_programs = { program_id: programs[program_id] };
		} else {
			all_programs = programs;
		}
		for (var pid in programs) {
			var sp = programs[pid];
			if (sp === current_program) {
				var location = sp.uniform_location(name);
				if (location === undefined)
					continue;
				args = [location].concat(u.slice(2));
				u[0].apply(gl, args);
			} else {
				sp.pending_uniforms.push(u);
			}
		}
	},

	set_uniform_matrix: function (program_id, name, transpose, shader_type, data) {
		// defer setting uniform until program is in use
		data = convert_data(data);
		var fa = new Float32Array(data, 0);
		var u = [];
		switch (shader_type) {
		  case llgr.Mat2x2:
			  u = [gl.uniformMatrix2fv, name, transpose, fa]; break;
		  case llgr.Mat3x3:
			  u = [gl.uniformMatrix3fv, name, transpose, fa]; break;
		  case llgr.Mat4x4:
			  u = [gl.uniformMatrix4fv, name, transpose, fa]; break;
		  default:
			console.log('only uniform matrix shader types allowed');
			return;
		}
		var all_programs;
		if (program_id) {
			all_programs = { program_id: programs[program_id] };
		} else {
			all_programs = programs;
		}
		for (var pid in programs) {
			var sp = programs[pid];
			if (sp === current_program) {
				var location = sp.uniform_location(name);
				if (location === undefined)
					continue;
				args = [location].concat(u.slice(2));
				u[0].apply(gl, args);
			} else {
				sp.pending_uniforms.push(u);
			}
		}
	},

	create_buffer: function (data_id, buffer_target, data) {
		if (buffers === null)
			init_buffers();
		data = convert_data(data);
		var bi = buffers[data_id];
		if (bi && bi.buffer) gl.deleteBuffer(bi.buffer);
		var buffer = gl.createBuffer();
		gl.bindBuffer(buffer_target, buffer);
		gl.bufferData(buffer_target, data, gl.STATIC_DRAW);
		gl.bindBuffer(buffer_target, null);
		if (curbuf[buffer_target] !== undefined) {
			delete curbuf[buffer_target];
		}
		buffers[data_id] = new BufferInfo(buffer, buffer_target);
	},
	delete_buffer: function (data_id) {
		if (data_id in buffers) {
			var bi = buffers[data_id];
			if (bi.buffer) gl.deleteBuffer(bi.buffer);
			delete buffers[data_id];
		}
	},
	clear_buffers: function () {
		for (var bid in buffers) {
			var bi = buffers[bid];
			if (bi.buffer) gl.deleteBuffer(bi.buffer);
		}
		buffers = null;
		llgr.clear_matrices();
		llgr.clear_primitives();
	},
	create_singleton: function (data_id, data) {
		if (buffers === null)
			init_buffers();
		data = convert_data(data);
		var bi = buffers[data_id];
		if (bi && bi.buffer) gl.deleteBuffer(bi.buffer);
		// TODO: want copy of data or read-only reference
		buffers[data_id] = new BufferInfo(gl.ARRAY_BUFFER,
							data.byteLength, data);
	},

	// matrix_id of zero is reserved for identity matrix
	create_matrix: function (matrix_id, matrix_4x4, renormalize) {
		if (renormalize === undefined) renormalize = false;
		var data_id = --internal_buffer_id;
		var data = new Float32Array(16);
		for (var i = 0; i < 16; ++i) {
			data[i] = matrix_4x4[i];
		}
		llgr.create_singleton(data_id, data);
		matrices[matrix_id] = new MatrixInfo(data_id, renormalize);
	},
	delete_matrix: function (matrix_id) {
		info = matrices[matrix_id];
		llgr.delete_buffer(info.data_id);
		delete matrices[matrix_id];
	},
	clear_matrices: function () {
		if (buffers !== null) {
			for (mid in matrices) {
				var info = matrices[mid];
				llgr.delete_buffer(info.data_id);
			}
		}
		matrices = {};
	},

	set_attribute_alias: function (name, value) {
		if (name == value || !value) {
			delete name_map[name];
		} else {
			name_map[name] = value;
		}
	},

	create_object: function (obj_id, program_id, matrix_id,
			list_of_attributeInfo, primitive_type, first, count,
			index_buffer_id, index_buffer_type) {
		if (index_buffer_id === undefined)
			index_buffer_id = 0;
		if (index_buffer_type === undefined)
			index_buffer_type = llgr.UByte;

		var ais = [];
		for (var i = 0; i < list_of_attributeInfo.length; ++i) {
			var args = list_of_attributeInfo[i];
			var name, data_id, offset, stride, cnt, type,
								normalized;
			if (args instanceof llgr.AttributeInfo) {
				var alias = attribute_alias(args.name);
				if (name == alias) {
					ais.push(args);
					continue;
				}
				name = alias;
				data_id = args.data_id;
				offset = args.offset;
				stride = args.stride;
				cnt = args.count;
				type = args.type;
				normalized = args.normalized;
			} else {
				//[name, data_id, offset, stride, cnt, type,
				//			normalized] = args;
				name = attribute_alias(args[0]);
				data_id = args[1];
				offset = args[2];
				stride = args[3];
				cnt = args[4];
				type = args[5];
				normalized = args[6];
			}
			ais.push(new llgr.AttributeInfo(name, data_id, offset,
					stride, cnt, type, normalized));
		}
		check_attributes(obj_id, program_id, ais);
		objects[obj_id] = new ObjectInfo(program_id, matrix_id,
			ais, primitive_type,
			first, count, index_buffer_id, index_buffer_type);
	},
	delete_object: function (obj_id) {
		delete objects[obj_id];
	},
	clear_objects: function () {
		objects = {}; }, 
	hide_objects: function (list_of_objects) {
		for (var obj_id in list_of_objects) {
			if (obj_id in objects)
				objects[obj_id].hide = true;
		}
	},
	show_objects: function (list_of_objects) {
		for (var obj_id in list_of_objects) {
			if (obj_id in objects)
				objects[obj_id].hide = false;
		}
	},

	transparent: function (list_of_objects) {
		for (var obj_id in list_of_objects) {
			if (obj_id in objects)
				objects[obj_id].transparent = true;
		}
	},
	opaque: function (list_of_objects) {
		for (var obj_id in list_of_objects) {
			if (obj_id in objects)
				objects[obj_id].transparent = false;
		}
	},

	selection_add: function (list_of_objects) {
		// TODO
	},
	selection_remove: function (list_of_objects) {
		// TODO
	},
	selection_clear: function () {
		// TODO
	},

	clear_primitives: function () {
		if (buffers !== null) {
			var radius, info;
			for (radius in proto_spheres) {
				info = proto_spheres[radius];
				llgr.delete_buffer(info.data_id);
				llgr.delete_buffer(info.index_id);
			}
			for (radius in proto_cylinders) {
				info = proto_cylinders[radius];
				llgr.delete_buffer(info.data_id);
				llgr.delete_buffer(info.index_id);
			}
		}
		proto_spheres = {};
		proto_cylinders = {};
	},
	add_sphere: function (obj_id, radius, program_id, matrix_id,
				list_of_attributeInfo) {
		var N = 300; // TODO: make dependent on radius in pixels
		if (!(N in proto_spheres)) {
			build_sphere(N);
		}
		var pi = proto_spheres[N];
		var mai = list_of_attributeInfo.slice(0);
		mai.push(new llgr.AttributeInfo("normal", pi.data_id, 0, 24, 3,
								llgr.Float));
		mai.push(new llgr.AttributeInfo("position", pi.data_id, 12, 24,
								3, llgr.Float));
		var scale_id = --internal_buffer_id;
		var scale = new Float32Array([radius, radius, radius]);
		llgr.create_singleton(scale_id, scale);
		mai.push(new llgr.AttributeInfo("instanceScale", scale_id, 0,
							0, 3, llgr.Float));
		llgr.create_object(obj_id, program_id, matrix_id, mai,
				llgr.Triangles, 0,
				pi.index_count, pi.index_id, pi.index_type);
	},
	add_cylinder: function (obj_id, radius, length, program_id, matrix_id,
				list_of_attributeInfo) {
		var N = 50;	// TODO: make dependent on radius in pixels
		if (!(N in proto_cylinders)) {
			build_cylinder(N);
		}
		var pi = proto_cylinders[N];
		var mai = list_of_attributeInfo.slice(0);
		mai.push(new llgr.AttributeInfo("normal", pi.data_id, 0, 24, 3,
								llgr.Float));
		mai.push(new llgr.AttributeInfo("position", pi.data_id, 12, 24,
								3, llgr.Float));
		var scale_id = --internal_buffer_id;
		var scale = new Float32Array([radius, length / 2, radius]);
		llgr.create_singleton(scale_id, scale);
		mai.push(new llgr.AttributeInfo("instanceScale", scale_id, 0,
							0, 3, llgr.Float));
		llgr.create_object(obj_id, program_id, matrix_id, mai,
				llgr.Triangle_strip, 0,
				pi.index_count, pi.index_id, pi.index_type);
	},

	clear_all: function () {
		llgr.clear_objects();
		llgr.clear_buffers();
		llgr.clear_programs();
	},

	set_clear_color: function(red, green, blue, alpha) {
		gl.clearColor(red, green, blue, alpha);
	},

	load_json: function (json) {
		var funcs = {
			create_program: llgr.create_program,
			delete_program: llgr.delete_program,
			clear_programs: llgr.clear_programs,
			set_uniform: llgr.set_uniform,
			set_uniform_matrix: llgr.set_uniform_matrix,
			create_buffer: llgr.create_buffer,
			delete_buffer: llgr.delete_buffer,
			clear_buffers: llgr.clear_buffers,
			create_singleton: llgr.create_singleton,
			create_matrix: llgr.create_matrix,
			delete_matrix: llgr.delete_matrix,
			clear_matrices: llgr.clear_matrices,
			set_attribute_alias: llgr.set_attribute_alias,
			create_object: llgr.create_object,
			delete_object: llgr.delete_objects,
			clear_objects: llgr.clear_objects,
			hide_objects: llgr.hide_objects,
			show_objects: llgr.show_objects,
			transparent: llgr.transparent,
			opaque: llgr.opaque,
			selection_add: llgr.selection_add,
			selection_remove: llgr.selection_remove,
			selection_clear: llgr.selection_clear,
			clear_primitives: llgr.clear_primitives,
			add_sphere: llgr.add_sphere,
			add_cylinder: llgr.add_cylinder,
			clear_all: llgr.clear_all,
			set_clear_color: llgr.set_clear_color,
		};
		for (var i = 0; i < json.length; ++i) {
			var fname = json[i][0];
			if (!(fname in funcs)) {
				console.log("unknown llgr function: " + fname);
				continue;
			}
			if (json[i].length === 1) {
				funcs[fname]();
			} else {
				funcs[fname].apply(undefined, json[i][1]);
			}
		}
	},

	render: function () {
		// TODO: if (dirty) optimize();
		var current_program_id = null;
		var current_sp = null;
		var current_matrix_id = null;
		var matrix_ai = new llgr.AttributeInfo("instanceTransform", 0,
							0, 0, 16, llgr.Float);
		// TODO: only for opaque objects
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
		gl.enable(gl.DEPTH_TEST);
		gl.enable(gl.CULL_FACE);
		gl.disable(gl.BLEND);
		for (var oid in objects) {
			var oi = objects[oid];
			if (oi.hide || !oi.program_id)
				continue;
			if (oi.program_id != current_program_id) {
				var sp = programs[oi.program_id];
				if (sp === undefined)
					continue;
				if (current_sp)
					current_sp.cleanup();
				current_sp = sp;
				sp.setup();
				current_program_id = oi.program_id;
				current_matrix_id = null;
			}
			// setup index buffer
			var ibi = undefined;
			if (oi.index_buffer_id) {
				ibi = buffers[oi.index_buffer_id];
			}
			// setup instance matrix attribute
			if (oi.matrix_id != current_matrix_id) {
				if (oi.matrix_id === 0) {
					matrix_ai.data_id = 0;
				} else {
					var mi = matrices[oi.matrix_id];
					if (mi === undefined)
						continue;
					matrix_ai.data_id = mi.data_id;
				}
				setup_attribute(sp, matrix_ai);
				current_matrix_id = oi.matrix_id;
			}
			// setup other attributes
			for (var j = 0; j < oi.all_ai.length; ++j) {
				var ai = oi.all_ai[j];
				setup_attribute(sp, ai);
			}
			// finally draw object
			if (!ibi) {
				if (curbuf[gl.ELEMENT_ARRAY_BUFFER] !== undefined) {
					gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
					delete curbuf[gl.ELEMENT_ARRAY_BUFFER];
				}
				gl.drawArrays(oi.ptype, oi.first, oi.count);
			} else {
				if (oi.index_buffer_type == llgr.UInt
				&& !gl.getExtension("OES_element_index_uint")) {
					console.warn("unsigned integer indices are not supported");
				}
				if (curbuf[gl.ELEMENT_ARRAY_BUFFER] !== ibi.buffer) {
					gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibi.buffer);
					curbuf[gl.ELEMENT_ARRAY_BUFFER] = ibi.buffer;
				}
				gl.drawElements(oi.ptype, oi.count,
					cvt_DataType(oi.index_buffer_type), 
					oi.first
					* data_size(oi.index_buffer_type));
			}
		}
		for (var loc in enabled) {
			if (enabled[loc]) {
				gl.disableVertexAttribArray(loc);
				enabled[loc] = false;
				enabled_count[loc] = 0;
				enabled_buf[loc] = null;
			}
		}
		if (curbuf[gl.ARRAY_BUFFER] !== undefined) {
			gl.bindBuffer(gl.ARRAY_BUFFER, null);
			delete curbuf[gl.ARRAY_BUFFER];
		}
		if (curbuf[gl.ELEMENT_ARRAY_BUFFER] !== undefined) {
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
			delete curbuf[gl.ELEMENT_ARRAY_BUFFER];
		}
		if (sp) {
			sp.cleanup();
		}
	},
};

})();
