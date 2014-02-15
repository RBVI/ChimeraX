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

var all_programs = {};
var all_buffers = null;
var all_matrices = {};
var all_objects = {};
var all_groups = {};

// set with set_context()
var gl;		// OpenGL API
var vao_ext;	// ES_vertex_array_object API
var inst_ext;	// ANGLE_instanced_arrays API

var internal_buffer_id = 0;	// decrement before using
var current_program = null;

var name_map = {
	position: "position",
	normal: "normal"
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
var proto_cones = {};
var proto_fans = {};

function ShaderVariable(location, type, size)
{
	this.location = location;
	this.type = type;
	this.size = size;
}

ShaderVariable.prototype.location_info = function ()
{
	// return number of locations and number of elements per location
	// for a given GL type
	switch (this.type) {
	  case gl.FLOAT: return [1, 1];
	  case gl.FLOAT_VEC2: return [1, 2];
	  case gl.FLOAT_VEC3: return [1, 3];
	  case gl.FLOAT_VEC4: return [1, 4];
	  case gl.INT: return [1, 1];
	  case gl.INT_VEC2: return [1, 2];
	  case gl.INT_VEC3: return [1, 3];
	  case gl.INT_VEC4: return [1, 4];
	  case gl.UNSIGNED_INT: return [1, 1];
	  case gl.UNSIGNED_INT_VEC2: return [1, 2];
	  case gl.UNSIGNED_INT_VEC3: return [1, 3];
	  case gl.UNSIGNED_INT_VEC4: return [1, 4];
	  case gl.BOOL: return [1, 1];
	  case gl.BOOL_VEC2: return [1, 2];
	  case gl.BOOL_VEC3: return [1, 3];
	  case gl.BOOL_VEC4: return [1, 4];
	  case gl.FLOAT_MAT2: return [2, 2];
	  case gl.FLOAT_MAT3: return [3, 3];
	  case gl.FLOAT_MAT4: return [4, 4];
	  case gl.FLOAT_MAT2x3: return [2, 3];
	  case gl.FLOAT_MAT2x4: return [2, 4];
	  case gl.FLOAT_MAT3x2: return [3, 2];
	  case gl.FLOAT_MAT3x4: return [3, 4];
	  case gl.FLOAT_MAT4x2: return [4, 2];
	  case gl.FLOAT_MAT4x3: return [4, 3];
	  default:
		console.log("unknown GL type 0x", this.type.toString(16));
		return undefined;
	}
};

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
		this.uniforms[name] = new ShaderVariable(
					gl.getUniformLocation(program, name),
					info.type, info.size);
	}
	total = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
	for (i = 0; i < total; ++i) {
		info = gl.getActiveAttrib(program, i);
		name = info.name;
		this.attributes[name] = new ShaderVariable(
					gl.getAttribLocation(program, name),
					info.type, info.size);
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

ShaderProgram.prototype.setup = function ()
{
	gl.useProgram(this.program);
	current_program = this;
	var len = this.pending_uniforms.length;
	for (var i = 0; i < len; ++i) {
		var u = this.pending_uniforms[i];
		var args = [this.uniforms[u[1]].location].concat(u.slice(2));
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
		this.data = null;
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
	all_buffers = {};
	all_buffers[0] = new BufferInfo(llgr.ARRAY, identity.byteLength, identity);
}

function MatrixInfo(id, renorm)
{
	// create MatrixInfo object
	this.data_id = id;
	this.renormalize = renorm;
}

function SingletonInfo(data_type, normalized, data, location, num_locations, num_elements)
{
	this.data_type = data_type;
	this.normalized = normalized;
	this.data = data;
	this.base_location = location;
	this.num_locations = num_locations;
	this.num_elements = num_elements;
}

function GroupInfo(group_id)
{
	this.group_id = group_id;
	this.objects = [];	// used as a set of integers
	this.optimized = false;
	this.ois = [];		// ObjectInfos to render
	this.buffers = [];	// generated buffers
}

GroupInfo.prototype.hashCode = function ()
{
	return "" + this.group_id;
};

GroupInfo.prototype.equals = function (obj)
{
	return (obj instanceof GroupInfo) && (this.group_id == obj.group_id);
};

GroupInfo.prototype.clear = function (and_objects)
{
	if (and_objects === undefined) and_objects = false;
	if (and_objects) {
		_.each(this.objects, function (obj_id) {
			llgr.delete_object(obj_id);
		});
	}
	this.objects.length = 0;
	this.reset_optimization();
};

GroupInfo.prototype.add = function (objects)
{
	this.objects = _.union(this.objects, objects);
};

GroupInfo.prototype.remove = function (objects)
{
	this.objects = _.difference(this.objects, objects);
};

GroupInfo.prototype.reset_optimization = function ()
{
	if (!this.optimized)
		return;
	this.optimized = false;
	this.ois.length = 0;
	// free instancing buffers
	_.each(this.buffers, function (buffer) {
		gl.deleteBuffer(buffer);
	});
	this.buffers.length = 0;
};

GroupInfo.prototype.optimize = function ()
{
	var this_gi = this;	// for inner functions
	var i, size;

	// scan through objects looking for possible instancing
	this.optimized = true;
	this.ois.length = 0;

	// first pass: group objects by program id, tranparency,
	// primitive type, if indexing, and array buffers
	function group_key(oi) {
		var ais = _.filter(oi.ais, function (ai) {
			return ai._is_array;
		});
		return [oi.program_id, oi._transparent, oi.ptype,
			oi.index_buffer_id, oi.index_buffer_type, ais];
	}
	var groupings = {};
	_.each(this.objects, function (obj_id) {
		var oi = all_objects[obj_id];
		if (oi === undefined || oi.incomplete || oi._hide)
			return;
		var k = group_key(oi);
		try {
			groupings[k].push(oi);
		} catch (e) {
			// assert(e isinstanceof TypeError);
			groupings[k] = [oi];
		}
	});

	// second pass: if all objects in a group have the same
	// first and count, then aggregate singletons, and use
	// instancing.
	var separate = {};
	var instancing = {};
	for (var key in groupings) {
		var ois = groupings[key];
		if (ois.length == 1) {
			//separate[key] = ois;
			// for speed in WebGL, always use instancing
			instancing[key] = ois;
			continue;
		}
		var first = true;
		var info;
		var same = !_.some(ois, function (oi) {
			if (first) {
				info = [oi.first, oi.count];
				first = false;
				return false;
			}
			if (!_.isEqual([oi.first, oi.count], info)) {
				separate[key] = ois;
				return true;
			}
			return false;
		});
		if (!same)
			continue;
		// TODO: separate transparent objects
		// TODO: group by compatible singletons
		instancing[key] = ois;
	}

	// TODO: remove assumption that all singletons are the same
	// TODO: if a particular  singleton is the same in all objects,
	// then keep it as a singleton
	var sp = null;
	var current_program_id = null;
	var si;
	_.each(instancing, function (ois) {
		var ai, bi;
		// aggregate singletons
		var proto = ois[0];
		var oi = new ObjectInfo(proto.program_id, 0, [], proto.ptype,
			proto.first, proto.count,
			proto.index_buffer_id, proto.index_buffer_type);
		oi.instance_count = ois.length;
		if (oi.program_id != current_program_id) {
			sp = all_programs[oi.program_id];
			if (sp === undefined) {
				current_program_id = null;
				oi.incomplete = true;
				return;
			}
			current_program_id = oi.program_id;
		}
		oi.vao = vao_ext.createVertexArrayOES();
		vao_ext.bindVertexArrayOES(oi.vao);

		_.each(proto.ais, function (ai) {
			if (!ai._is_array || oi.incomplete)
				return;
			var sv = sp.attributes[ai.name];
			bi = all_buffers[ai.data_id];
			if (bi === undefined) {
				// buffer deleted after object creation,
				// and before rendering
				oi.incomplete = true;
				return;
			}
			var num_locations = sv.location_info()[0];
			setup_array_attribute(bi, ai, sv.location,
							num_locations);
		});
		if (oi.incomplete)
			return;
		if (proto.index_buffer_id) {
			var ibi = all_buffers[proto.index_buffer_id];
			if (ibi !== undefined) {
				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibi.buffer);
			}
		}
		// interleave singleton data
		// TODO: reorder singletons so smaller values pack
		var sizes = [];
		var stride = 0;
		_.each(proto.singleton_cache, function (si) {
			size = si.num_locations * si.num_elements * data_size(si.data_type);
			// round to multiple of 4
			var remainder = size % 4;
			if (remainder)
				size += (4 - remainder);
			sizes.push(size);
			stride += size;
		});
		if (stride === 0) {
			this_gi.ois.push(oi);
			return;
		}
		// TODO: assert(stride < glGetInteger(GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET))
		var buffer = gl.createBuffer();
		var buflen = ois.length * stride;
		this_gi.buffers.push(buffer);
		gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
		gl.bufferData(gl.ARRAY_BUFFER, buflen, gl.STATIC_DRAW);

		var pos = 0;
		_.each(ois, function (oi2) {
			for (i = 0; i < sizes.length; ++i) {
				size = sizes[i];
				si = oi2.singleton_cache[i];
				gl.bufferSubData(gl.ARRAY_BUFFER, pos,
								si.data);
				pos += size;
			}
		});
		if (pos != buflen) {
			console.log("tried to fill " + buflen + " byte buffer with " + pos + " bytes");
		}
		bi = new BufferInfo(buffer, llgr.ARRAY);
		var offset = 0;
		for (i = 0; i < sizes.length; ++i) {
			size = sizes[i];
			si = proto.singleton_cache[i];
			ai = new llgr.AttributeInfo(null, null, offset, stride,
				si.num_elements, si.data_type,
				si.normalized);
			offset += size;
			setup_array_attribute(bi, ai, si.base_location,
						si.num_locations);
			for (var j = 0; j != si.num_locations; ++j) {
				inst_ext.vertexAttribDivisorANGLE(si.base_location + j, 1);
			}
		}
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		if (!oi.incomplete)
			this_gi.ois.push(oi);
	});

	// third pass: look at left over groupings
	// TODO: If different first and count, try to group
	// into sequental order, and combine into one mega-object.

	// forth pass: anything not handled above
	sp = null;
	current_program_id = null;
	ois = _.flatten(separate);
	_.each(ois, function (oi) {
		if (oi.program_id != current_program_id) {
			sp = all_programs[oi.program_id];
			if (sp === undefined) {
				current_program_id = null;
				oi.incomplete = true;
				return;
			}
			current_program_id = oi.program_id;
		}
		oi.vao = vao_ext.createVertexArrayOES();
		vao_ext.bindVertexArrayOES(oi.vao);
		_.each(oi.ais, function (ai) {
			if (!ai._is_array || oi.incomplete)
				return;
			var sv = sp.attributes[ai.name];
			var num_locations = sv.location_info()[0];
			var bi = all_buffers[ai.data_id];
			if (bi === undefined) {
				// buffer deleted after object creation,
				// and before rendering
				oi.incomplete = true;
				return;
			}
			setup_array_attribute(bi, ai, sv.location,
							num_locations);
		});
		if (oi.index_buffer_id) {
			var ibi = all_buffers[oi.index_buffer_id];
			if (ibi !== undefined) {
				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibi.buffer);
			}
		}
	});

	vao_ext.bindVertexArrayOES(null);
	this.ois = this.ois.concat(ois);
};

GroupInfo.prototype.render = function ()
{
	// TODO: change API to be multipass by shader program
	// and transparency
	if (!this.optimized)
		this.optimize();

	var sp = null;
	var current_program_id = 0;
	_.each(this.ois, function (oi) {
		if (oi.hide() || oi.incomplete)
			return;
		// setup program
		if (oi.program_id != current_program_id) {
			var new_sp = all_programs[oi.program_id];
			if (new_sp === undefined)
				return;
			new_sp.setup();
			sp = new_sp;
			current_program_id = oi.program_id;
		}
		if (sp === null)
			return;
		vao_ext.bindVertexArrayOES(oi.vao);
		_.each(oi.singleton_cache, function (si) {
			setup_singleton_attribute(si.data,
				si.data_type, si.normalized,
				si.base_location, si.num_locations,
				si.num_elements);
		});
		// finally draw object
		var offset;
		if (oi.instance_count === 0) {
			if (oi.index_buffer_id === 0) {
				gl.drawArrays(oi.ptype, oi.first, oi.count);
			} else {
				if (oi.first === 0)
					offset = 0;
				else
					offset = oi.first *
					    data_size(oi.index_buffer_type);
				gl.drawElements(oi.ptype, oi.count,
					cvt_data_type(oi.index_buffer_type),
					offset);
			}
		} else if (oi.index_buffer_id === 0) {
			inst_ext.drawArraysInstancedANGLE(oi.ptype, oi.first,
						oi.count, oi.instance_count);
		} else {
			if (oi.first === 0)
				offset = 0;
			else
				offset = oi.first *
				    data_size(oi.index_buffer_type);
			inst_ext.drawElementsInstancedANGLE(oi.ptype, oi.count,
				cvt_data_type(oi.index_buffer_type), offset,
				oi.instance_count);
		}
	});
	vao_ext.bindVertexArrayOES(null);
	if (sp)
		sp.cleanup();
};

GroupInfo.prototype.pick = function ()
{
	// TODO
}; 
function ObjectInfo(program_id, matrix_id, attrinfo, primitive, first, count, index_id, index_type)
{
	// create ObjectInfo object
	if (index_id === undefined) index_id = 0;
	if (index_type === undefined) index_type = llgr.UByte;

	this.program_id = program_id;
	this.matrix_id = matrix_id;
	this.ais = attrinfo;		// TODO: copy
	this.ptype = primitive;
	this.first = first;
	this.count = count;
	this.index_buffer_id = index_id;
	this.index_buffer_type = index_type;
	this._hide = false;
	this._transparent = false;
	this.selected = false;
	this.singleton_cache = [];
	this.vao = null;
	this.groups = new HashSet();
	this.incomplete = false;
	this.instance_count = 0;
}

ObjectInfo.prototype.hide = function ()
{
	return this._hide;
};

ObjectInfo.prototype.setHide = function (on_off)
{
	if (on_off == this._hide)
		return;
	this._hide = on_off;
	_.each(self.groups.values(),  function (gi) {
		if (gi.optimized)
			gi.reset_optimization();
	});
};

ObjectInfo.prototype.transparent = function ()
{
	return this._transparent;
};

ObjectInfo.prototype.setTransparent = function (on_off)
{
	if (on_off == self._transparent)
		return;
	this._transparent = on_off;
	_.each(self.groups.values(),  function (gi) {
		if (gi.optimized)
			gi.reset_optimization();
	});
};

function check_attributes(obj_id, oi)
{
	if (!(oi.program_id in all_programs)) {
		console.log("missing program for object " + obj_id);
		return;
	}
	oi.singleton_cache.length = 0;
	var sp = all_programs[oi.program_id];
	for (var name in sp.attributes) {
		var ai = null;
		for (var i = 0; i < oi.ais.length; ++i) {
			if (oi.ais[i].name == name) {
				ai = oi.ais[i];
				break;
			}
		}
		if (ai === null) {
			console.log("missing attribute " + name +
						" in object " + obj_id);
			continue;
		}
		var bi = all_buffers[ai.data_id];
		if (bi === undefined) {
			console.log("missing buffer for attribute " + name +
					" in object " + obj_id);
			oi.incomplete = true;
			return;
		}
		var sv = sp.attributes[name];
		var info = sv.location_info();
		if (bi.data === null) {
			ai._is_array = true;
			continue;
		}
		if (sv.location === 0) {
			console.log("warning: vertices must be in an array in object " + obj_id);
			oi.incomplete = true;
			return;
		}
		ai._is_array = false;
		oi.singleton_cache.push(new SingletonInfo(ai.data_type,
				ai.normalized, bi.data, sv.location,
				info[0], info[1]));
	}
	// sort attributes -- arrays first
	oi.ais.sort(function (a, b) {
		if (a._is_array && !b._is_array)
			return -1;
		if (a.name < b.name)
			return -1;
		if (a.name > b.name)
			return 1;
		return 0;
	});
	// sort singleton cache by base location
	oi.singleton_cache.sort(function (a, b) {
		return a.base_location - b.base_location;
	});
	oi.incomplete = false;
}

function cvt_data_type(dt)
{
	switch (dt) {
	  case llgr.Byte: return gl.BYTE;
	  case llgr.UByte: return gl.UNSIGNED_BYTE;
	  case llgr.Short: return gl.SHORT;
	  case llgr.UShort: return gl.UNSIGNED_SHORT;
	  case llgr.Int: return gl.INT;
	  case llgr.UInt: return gl.UNSIGNED_INT;
	  case llgr.Float: return gl.FLOAT;
	  default: return undefined;
	}
}

function data_size(data_type)
{
	switch (data_type) {
	  case llgr.Byte: case llgr.UByte: return 1;
	  case llgr.Short: case llgr.UShort: return 2;
	  case llgr.Int: case llgr.UInt: return 4;
	  case llgr.Float: return 4;
	  default: return undefined;
	}
}

function setup_array_attribute(bi, ai, loc, num_locations)
{
	var gl_type = cvt_data_type(ai.data_type);
	var size = ai.count * data_size(ai.data_type);
	gl.bindBuffer(bi.target, bi.buffer);
	// TODO? if shader variable is int, use glVertexAttribIPointer
	var offset = ai.offset;
	for (var i = 0; i < num_locations; ++i) {
		gl.vertexAttribPointer(loc + i, ai.count, gl_type,
			ai.normalized, ai.stride, offset);
		gl.enableVertexAttribArray(loc + i);
		offset += size;
	}
}

var singleton_map;
var _did_only_float = false;
function setup_singleton_attribute(data, data_type, normalized, loc, num_locations, num_elements)
{
	if (data_type !== llgr.Float) {
		if (!_did_only_float) {
			console.log("WebGL only supports float singleton vertex attributes", typeof data_type, data_type);
			_did_only_float = true;
		}
	}
	if (singleton_map === undefined) {
		singleton_map = [
			undefined,
			new Hashtable(),
			new Hashtable(),
			new Hashtable(),
			new Hashtable(),
			new Hashtable()
		];
		var tmp = singleton_map[1];
		//tmp.put(llgr.Short, gl.vertexAttrib1sv);
		//tmp.put(llgr.UShort, gl.vertexAttrib1sv);
		tmp.put(llgr.Float, gl.vertexAttrib1fv);
		tmp = singleton_map[2];
		//tmp.put(llgr.Short, gl.vertexAttrib2sv);
		//tmp.put(llgr.UShort, gl.vertexAttrib2sv);
		tmp.put(llgr.Float, gl.vertexAttrib2fv);
		tmp = singleton_map[3];
		//tmp.put(llgr.Short, gl.vertexAttrib3sv);
		//tmp.put(llgr.UShort, gl.vertexAttrib3sv);
		tmp.put(llgr.Float, gl.vertexAttrib3fv);
		tmp = singleton_map[4];
		//tmp.put(llgr.Byte, gl.vertexAttrib4bv);
		//tmp.put(llgr.UByte, gl.vertexAttrib4ubv);
		//tmp.put(llgr.Short, gl.vertexAttrib4sv);
		//tmp.put(llgr.UShort, gl.vertexAttrib4usv);
		//tmp.put(llgr.Int, gl.vertexAttrib4iv);
		//tmp.put(llgr.UInt, gl.vertexAttrib4uiv);
		tmp.put(llgr.Float, gl.vertexAttrib4fv);
		tmp = singleton_map[5];
		//tmp.put(llgr.Byte, gl.vertexAttrib4Nbv);
		//tmp.put(llgr.UByte, gl.vertexAttrib4Nubv);
		//tmp.put(llgr.Short, gl.vertexAttrib4Nsv);
		//tmp.put(llgr.UShort, gl.vertexAttrib4Nusv);
		//tmp.put(llgr.Int, gl.vertexAttrib4Niv);
		//tmp.put(llgr.UInt, gl.vertexAttrib4Nuiv);
	}

	if (num_elements == 4 && normalized)
		num_elements = 5;
	var func = singleton_map[num_elements].get(data_type);
	if (func === undefined)
		return;
	var bytes;
	if (num_locations == 1) {
		bytes = new Float32Array(data);
		func.call(gl, loc, bytes);
	} else {
		var bytes = new Float32Array(data);
		for (var i = 0; i < num_locations; ++i) {
			var subarray = bytes.subarray(i * num_elements,
							(i + 1) * num_elements);
			func.call(gl, loc + i, subarray);
		}
	}
}

function convert_data(data)
{
	if (data instanceof ArrayBuffer ||
			data instanceof DataView ||
			data instanceof Int8Array ||
			data instanceof Uint8Array ||
			data instanceof Int16Array ||
			data instanceof Uint16Array ||
			data instanceof Int32Array ||
			data instanceof Uint32Array ||
			data instanceof Float32Array) {
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
	if (size >= 1) { view.setUint8(i, words[w], little_endian);
		i += 1;
		size -= 1;
	}
	return data;
}

function build_sphere(num_vertices)
{
	var bands = Math.round(Math.sqrt(num_vertices)) - 1;
	if (bands < 4)
		bands = 4;
	var spokes = Math.round(num_vertices / bands) - 1;
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

function build_cylinder(num_spokes)
{
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
		var offset = i * 6;
		np[offset + 0] = x;
		np[offset + 1] = 0;
		np[offset + 2] = z;
		np[offset + 3] = x;
		np[offset + 4] = -1;
		np[offset + 5] = z;
		var offset2 = (i + num_spokes) * 6;
		np[offset2 + 0] = np[offset + 0];
		np[offset2 + 1] = np[offset + 1];
		np[offset2 + 2] = np[offset + 2];
		np[offset2 + 3] = np[offset + 3];
		np[offset2 + 4] = 1;
		np[offset2 + 5] = np[offset + 5];
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

function build_cone(num_spokes)
{
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
		var offset = i * 6;
		np[offset + 0] = x;
		np[offset + 1] = 0;
		np[offset + 2] = z;
		np[offset + 3] = x;
		np[offset + 4] = -1;
		np[offset + 5] = z;
		var offset2 = (i + num_spokes) * 6;
		np[offset2 + 0] = np[offset + 0];
		np[offset2 + 1] = np[offset + 1];
		np[offset2 + 2] = np[offset + 2];
		np[offset2 + 3] = 0;
		np[offset2 + 4] = 1;
		np[offset2 + 5] = 0;
		indices[i * 2 + 0] = i;
		indices[i * 2 + 1] = i + num_spokes;
	}
	indices[num_spokes * 2 + 0] = 0;
	indices[num_spokes * 2 + 1] = num_spokes;

	var np_id = --internal_buffer_id;
	var index_id = --internal_buffer_id;
	llgr.create_buffer(np_id, llgr.ARRAY, np);
	llgr.create_buffer(index_id, llgr.ELEMENT_ARRAY, indices);
	proto_cones[num_spokes] = new PrimitiveInfo(np_id, num_indices,
							index_id, index_type);
}

function build_fan(num_spokes)
{
	var pts = new Float32Array(3 * num_spokes + 6);	// positions
	var num_indices = num_spokes + 2;
	pts[0] = pts[1] = pts[2] = 0;
	var offset;
	for (var i = 0; i < num_spokes; ++i) {
		var theta = 2 * Math.PI * i / num_spokes;
		var x = Math.cos(theta);
		var z = Math.sin(theta);
		offset = (num_spokes - i) * 3;
		pts[offset + 0] = x;
		pts[offset + 1] = 0;
		pts[offset + 2] = z;
	}
	offset = num_spokes * 3 + 3;
	pts[offset + 0] = pts[3];
	pts[offset + 1] = pts[4];
	pts[offset + 2] = pts[5];

	var pts_id = --internal_buffer_id;
	llgr.create_buffer(pts_id, llgr.ARRAY, pts);
	proto_fans[num_spokes] = new PrimitiveInfo(pts_id, num_indices, 0, 0);
}

llgr = {
	set_context: function(context) {
		if (gl === context)
			return;
		var missing = [];
		if (vao_ext === undefined) {
			vao_ext = getExtensionWithKnownPrefixes(context,
						"OES_vertex_array_object");
			if (vao_ext === null) {
				missing.push("missing required WebGL extension: vertex arrays");
			}
		}
		if (inst_ext === undefined) {
			inst_ext = getExtensionWithKnownPrefixes(context,
						"ANGLE_instanced_arrays");
			if (inst_ext === null) {
				missing.push("missing required WebGL extension: instanced arrays");
			}
		}
		if (missing.length > 0) {
			llgr = null;
			throw missing;
		}
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
		this.data_type = data_type;
		this.normalized = normalized;
		this._is_array = false;
	},

	create_program: function (program_id, vert_shader, frag_shader) {
		if (program_id <= 0) {
			throw "need positive program id";
		}
		if (program_id in all_programs) {
			all_programs[program_id].gl_dealloc();
			delete all_programs[program_id];
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
		var attribute0_name = "position";
		if (attribute0_name !== undefined)
			gl.bindAttribLocation(program, 0, attribute0_name);
		gl.linkProgram(program);
		if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
			console.log(gl.getProgramInfoLog(program));
			gl.deleteProgram(program);
			gl.deleteShader(vs);
			gl.deleteShader(fs);
			return;
		}
		gl.validateProgram(program);
		if (!gl.getProgramParameter(program, gl.VALIDATE_STATUS)) {
			console.log(gl.getProgramInfoLog(program));
			gl.deleteProgram(program);
			gl.deleteShader(vs);
			gl.deleteShader(fs);
			return;
		}

		all_programs[program_id] = new ShaderProgram(program, vs, fs);
	},
	delete_program: function (program_id) {
		if (program_id in all_programs) {
			sp = all_programs[program_id];
			if (sp === current_program) {
				current_program = null;
				gl.useProgram(0);
			}
			sp.gl_dealloc();
			delete all_programs[program_id];
		}
	},
	clear_programs: function () {
		current_program = null;
		gl.useProgram(null);
		_.each(all_programs, function (sp) {
			sp.gl_dealloc();
		});
		all_programs = {};
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
			console.log('unknown uniform shader type for ' + name +
							': ' + shader_type);
			return;
		}
		var programs;
		if (program_id) {
			programs = { program_id: all_programs[program_id] };
		} else {
			programs = all_programs;
		}
		_.each(programs, function (sp) {
			if (sp === current_program) {
				var location = sp.uniforms[name].location;
				if (location === undefined)
					return;
				args = [location].concat(u.slice(2));
				u[0].apply(gl, args);
			} else {
				sp.pending_uniforms.push(u);
			}
		});
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
		var programs;
		if (program_id) {
			programs = { program_id: all_programs[program_id] };
		} else {
			programs = all_programs;
		}
		_.each(programs, function (sp) {
			if (sp === current_program) {
				var location = sp.uniforms[name].location;
				if (location === undefined)
					return;
				args = [location].concat(u.slice(2));
				u[0].apply(gl, args);
			} else {
				sp.pending_uniforms.push(u);
			}
		});
	},

	create_buffer: function (data_id, buffer_target, data) {
		if (all_buffers === null)
			init_buffers();
		data = convert_data(data);
		var bi = all_buffers[data_id];
		if (bi && bi.buffer) gl.deleteBuffer(bi.buffer);
		var buffer = gl.createBuffer();
		gl.bindBuffer(buffer_target, buffer);
		gl.bufferData(buffer_target, data, gl.STATIC_DRAW);
		gl.bindBuffer(buffer_target, null);
		all_buffers[data_id] = new BufferInfo(buffer, buffer_target);
	},
	delete_buffer: function (data_id) {
		if (data_id in all_buffers) {
			var bi = all_buffers[data_id];
			if (bi.buffer) gl.deleteBuffer(bi.buffer);
			delete all_buffers[data_id];
		}
	},
	clear_buffers: function () {
		_.each(all_buffers, function (bi) {
			if (bi.buffer)
				gl.deleteBuffer(bi.buffer);
		});
		all_buffers = {};
		llgr.clear_matrices();
		llgr.clear_primitives();
	},
	create_singleton: function (data_id, data) {
		if (all_buffers === null)
			init_buffers();
		data = convert_data(data);
		var bi = all_buffers[data_id];
		if (bi && bi.buffer) gl.deleteBuffer(bi.buffer);
		// TODO: want copy of data or read-only reference
		all_buffers[data_id] = new BufferInfo(gl.ARRAY_BUFFER,
							data.byteLength, data);
	},

	// matrix_id of zero is reserved for identity matrix
	create_matrix: function (matrix_id, matrix_4x4, renormalize) {
		if (renormalize === undefined) renormalize = false;
		var data_id;
		var mi = all_matrices[matrix_id];
		if (mi === undefined)
			data_id = --internal_buffer_id;
		else
			data_id = mi.data_id;
		var data = new Float32Array(16);
		for (var i = 0; i < 16; ++i) {
			data[i] = matrix_4x4[i];
		}
		llgr.create_singleton(data_id, data);
		all_matrices[matrix_id] = new MatrixInfo(data_id, renormalize);
	},
	delete_matrix: function (matrix_id) {
		var info = all_matrices[matrix_id];
		llgr.delete_buffer(info.data_id);
		delete all_matrices[matrix_id];
	},
	clear_matrices: function () {
		if (all_buffers !== null) {
			_.each(all_matrices, function (mi) {
				llgr.delete_buffer(mi.data_id);
			});
		}
		all_matrices = {};
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
		if (index_buffer_type == llgr.UInt &&
				!gl.getExtension("OES_element_index_uint")) {
			console.warn("unsigned integer indices are not supported");
		}

		var ais = [];
		for (var i = 0; i < list_of_attributeInfo.length; ++i) {
			var args = list_of_attributeInfo[i];
			var name, data_id, offset, stride, cnt, data_type,
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
				data_type = args.data_type;
				normalized = args.normalized;
			} else {
				//[name, data_id, offset, stride, cnt,
				//		data_type, normalized] = args;
				name = attribute_alias(args[0]);
				data_id = args[1];
				offset = args[2];
				stride = args[3];
				cnt = args[4];
				data_type = args[5];
				normalized = args[6];
			}
			ais.push(new llgr.AttributeInfo(name, data_id, offset,
					stride, cnt, data_type, normalized));
		}
		var mi = all_matrices[matrix_id];
		if (mi !== undefined) {
			ais.push(new llgr.AttributeInfo("instanceTransform",
					mi.data_id, 0, 0, 16, llgr.Float));
		}
		var oi = new ObjectInfo(program_id, matrix_id, ais,
				primitive_type, first, count,
				index_buffer_id, index_buffer_type);
		llgr.delete_object(obj_id);
		check_attributes(obj_id, oi);
		all_objects[obj_id] = oi;
	},
	delete_object: function (obj_id) {
		delete all_objects[obj_id];
	},
	clear_objects: function () {
		all_objects = {};
		llgr.clear_groups();
	}, 
	hide_objects: function (objects) {
		_.each(objects, function (obj_id) {
			if (obj_id in all_objects)
				all_objects[obj_id].hide = true;
		});
	},
	show_objects: function (objects) {
		_.each(objects, function (obj_id) {
			if (obj_id in all_objects)
				all_objects[obj_id].hide = false;
		});
	},

	transparent: function (objects) {
		_.each(objects, function (obj_id) {
			if (obj_id in all_objects)
				all_objects[obj_id].transparent = true;
		});
	},
	opaque: function (objects) {
		_.each(objects, function (obj_id) {
			if (obj_id in all_objects)
				all_objects[obj_id].transparent = false;
		});
	},

	selection_add: function (objects) {
		// TODO
	},
	selection_remove: function (objects) {
		// TODO
	},
	selection_clear: function () {
		// TODO
	},

	create_group: function (group_id) {
		var gi = all_groups[group_id];
		if (gi !== undefined)
			gi.clear();
		else
			all_groups[group_id] = new GroupInfo(group_id);
	},
	delete_group: function (group_id, and_objects) {
		if (and_objects === undefined)
			and_objects = false;
		var gi = all_groups[group_id];
		if (gi === undefined)
			return;
		gi.clear(and_objects);
		delete all_groups[group_id];
	},
	clear_groups: function (and_objects) {
		if (and_objects === undefined)
			and_objects = false;
		_.each(all_groups, function (gi) {
			gi.clear(and_objects);
		});
		all_groups = {};
	},
	group_add: function (group_id, objects) {
		var gi = all_groups[group_id];
		if (gi === undefined)
			return;
		gi.add(objects);
		_.each(objects, function (obj_id) {
			var oi = all_objects[obj_id];
			if (oi !== undefined)
				oi.groups.add(gi);
		});
	},
	group_remove: function (group_id, objects) {
		var gi = all_groups[group_id];
		if (gi === undefined)
			return;
		_.each(objects, function (obj_id) {
			var oi = all_objects[obj_id];
			if (oi !== undefined)
				oi.groups.remove(gi);
		});
		gi.remove(objects);
	},
	hide_group: function (group_id) {
		if (group_id in all_groups) {
			llgr.hide_objects(all_groups[group_id]);
		}
	},
	show_group: function (group_id) {
		if (group_id in all_groups) {
			llgr.show_objects(all_groups[group_id]);
		}
	},
	selection_add_group: function (group_id) {
		if (group_id in all_groups) {
			llgr.selection_add(all_groups[group_id]);
		}
	},
	selection_remove_group: function (group_id) {
		if (group_id in all_groups) {
			llgr.selection_remove(all_groups[group_id]);
		}
	},

	clear_primitives: function () {
		if (all_buffers !== null) {
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
			for (radius in proto_cones) {
				info = proto_cones[radius];
				llgr.delete_buffer(info.data_id);
				llgr.delete_buffer(info.index_id);
			}
			for (radius in proto_fans) {
				info = proto_fans[radius];
				llgr.delete_buffer(info.data_id);
			}
		}
		proto_spheres = {};
		proto_cylinders = {};
		proto_cones = {};
		proto_fans = {};
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
	add_cone: function (obj_id, radius, length, program_id, matrix_id,
				list_of_attributeInfo) {
		var N = 50;	// TODO: make dependent on radius in pixels
		if (!(N in proto_cones)) {
			build_cone(N);
		}
		var pi = proto_cones[N];
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
	add_disk: function (obj_id, inner_radius, outer_radius, program_id,
				    matrix_id, list_of_attributeInfo) {
		// TODO: don't ignore inner_radius
		var N = 50;	// TODO: make dependent on radius in pixels
		if (!(N in proto_fans)) {
			build_fan(N);
		}
		var pi = proto_fans[N];
		var mai = list_of_attributeInfo.slice(0);
		var normal_id = --internal_buffer_id;
		var normal = new Float32Array([0, 1, 0]);
		llgr.create_singleton(normal_id, normal);
		mai.push(new llgr.AttributeInfo("normal", normal_id, 0, 0, 3,
								llgr.Float));
		mai.push(new llgr.AttributeInfo("position", pi.data_id, 0, 0, 3,
								llgr.Float));
		var scale_id = --internal_buffer_id;
		var scale = new Float32Array([outer_radius, 1, outer_radius]);
		llgr.create_singleton(scale_id, scale);
		mai.push(new llgr.AttributeInfo("instanceScale", scale_id, 0,
							0, 3, llgr.Float));
		llgr.create_object(obj_id, program_id, matrix_id, mai,
				llgr.Triangle_fan, 0,
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
			delete_object: llgr.delete_object,
			clear_objects: llgr.clear_objects,
			hide_objects: llgr.hide_objects,
			show_objects: llgr.show_objects,
			transparent: llgr.transparent,
			opaque: llgr.opaque,
			selection_add: llgr.selection_add,
			selection_remove: llgr.selection_remove,
			selection_clear: llgr.selection_clear,
			create_group: llgr.create_group,
			delete_group: llgr.delete_group,
			clear_groups: llgr.clear_group,
			group_add: llgr.group_add,
			hide_group: llgr.hide_group,
			show_group: llgr.show_group,
			selection_add_group: llgr.selection_add_group,
			selection_remove_group: llgr.selection_remove_group,
			add_sphere: llgr.add_sphere,
			add_cylinder: llgr.add_cylinder,
			add_cone: llgr.add_cone,
			add_disk: llgr.add_disk,
			clear_primitives: llgr.clear_primitives,
			clear_all: llgr.clear_all,
			set_clear_color: llgr.set_clear_color
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

	render: function (groups) {
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
		gl.enable(gl.DEPTH_TEST);
		gl.enable(gl.DITHER);
		gl.disable(gl.SCISSOR_TEST);

		// TODO: only for opaque objects
		gl.enable(gl.CULL_FACE);
		gl.disable(gl.BLEND);

		var gis = [];
		_.each(groups, function (gid) {
			var gi = all_groups[gid];
			if (gi === undefined || gi.objects.length === 0)
				return;
			gis.push(gi);
		});

		// TODO: multipass to minimize shader program changes
		// and support transparency
		_.each(gis, function (gi) {
			//try {
				gi.render();
			//} catch (e) {
			//	console.log('render failed:', e.message,
			//		' in file ', e.fileName,
			//		' on line ', e.lineNumber);
			//}
		});
	}
};

})();
