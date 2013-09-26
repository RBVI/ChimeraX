// --------------------------------------------------------------------
// Session Dialog functions
// --------------------------------------------------------------------

// public API
var $c2_session = {};

(function () {
"use strict";

// --------------------------------------------------------------------
// Functions for communicating with server
// --------------------------------------------------------------------

function init(url) {
	$c2_session.server.url = url;
}

function list_sessions() {
	// Retrieve list of sessions from server
	if ($c2_session.server.url === null) {
		alert("Session module is uninitialized.");
		return null;
	}
	var data = {
		action: "jlist",
	}
	return $.getJSON($c2_session.server.url, data);
}

function create_session(session, password) {
	// Create session on server
	if ($c2_session.server.url === null) {
		alert("Session module is uninitialized.");
		return null;
	}
	var data = {
		action: "create",
		session: session,
		password: password,
	}
	return $.get($c2_session.server.url, data);
}

function delete_session(session, password) {
	// Create session on server
	if ($c2_session.server.url === null) {
		alert("Session module is uninitialized.");
		return null;
	}
	var data = {
		action: "delete",
		session: session,
		password: password,
	}
	return $.get($c2_session.server.url, data);
}

var call_id = 1;		// monotonically increasing id

function call(session, password, tag, tag_data, state, cb) {
	// Send tag and associated data to server
	if ($c2_session.server.url === null) {
		alert("Session module is uninitialized.");
		return null;
	}
	var cid = call_id;
	var call_data = [];
	if (state) {
		call_id = cid + 1;
		call_data.push([ cid, "client_state", state ]);
		cid += 1;
	}
	call_id = cid + 1;
	call_data.push([ cid, tag, tag_data ]);
	// alert("call: " + JSON.stringify(call_data));
	function clear_callback() {
		alert("getJSON failed");
	}
	var data = {
		action: "call",
		session: session,
		password: password,
		command: JSON.stringify(call_data),
	}
	return $.getJSON($c2_session.server.url, data)
			.done(debug_log)
			.fail(clear_callback);
}

function debug_log(data) {
	// TODO: be able to toggle debug output
	//var output = JSON.stringify(data) + "\n";
	var output = "";
	for (var index in data) {
		output = output + "<h4>Response " + index + "</h4>\n";
		var response = data[index];
		for (var key in response)
			output = output + "  <b>" + key + ":</b> "
				+ JSON.stringify(response[key]) + "<br>\n";
	}
	$("#debug").html(output);
}

// --------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------

$c2_session = {
	// Server API
	server: {
		url: null,
		init: init,
		list_sessions: list_sessions,
		create_session: create_session,
		delete_session: delete_session,
		call: call,
	},
};

}());

(function() {
"use strict";

// --------------------------------------------------------------------
// Functions for building user interface
// --------------------------------------------------------------------

function ui_init(url) {
	// initialize user interface (session button and dialog)
	$c2_session.server.init(url);

	// initialize public data elements
	$c2_session.user = "";
	$c2_session.session = "";
	$c2_session.password = "";

	// Register mouse-click handler for the open buttons
	$("#open_session").click(open_session);
	$("#delete_session").click(delete_session);
	$("#open_new_session").click(open_new_session);
	$("#open_shared_session").click(open_shared_session);
	update_session_list();
}

function open_session() {
	// verify and set default session parameters (name and password)
	var session = $("#my_sessions").val();
	var password = $("#my_password").val();
	if (!existing_session(session)) {
		alert("Session \"" + session + "\" does not exist.");
		return;
	}
	// alert("Session \"" + session + "\" selected.");
	save_session_info(session, password);
}

function save_session_info(session, password) {
	$c2_session.session = session;
	$c2_session.password = password;
	var msg = "No session selected";
	if (session)
		msg = "Active session: " + session;
	$("#active_session").html(msg);
	$("#session_popup").popup("close");
}

function delete_session() {
	// Delete using session parameters (name and password)
	var session = $("#my_sessions").val();
	var password = $("#my_password").val();
	$c2_session.server.delete_session(session, password)
						.done(delete_session_cb);
	alert("Deleting ession \"" + session + "\".");
}

function delete_session_cb() {
	var session = $("#my_sessions").val();
	var password = $("#my_password").val();
	alert("Session \"" + session + "\" deleted.");
	save_session_info("", "");
}

function open_new_session() {
	// Create using session parameters (name and password)
	var session = $("#new_session").val();
	var password = $("#new_password").val();
	$c2_session.server.create_session(session, password)
						.done(open_new_session_cb);
}

function open_new_session_cb() {
	var session = $("#new_session").val();
	var password = $("#new_password").val();
	// alert("Session \"" + session + "\" created.");
	save_session_info(session, password);
}

function update_session_list() {
	// disable button and initiate AJAX to update list
	$c2_session.server.list_sessions().done(update_session_list_cb)
}

function update_session_list_cb(session_info) {
	// save session information for later use
	$c2_session.user = session_info[0];
	var session_list = session_info[1];
	$c2_session.session_list = session_list;
	if (session_list.length == 0)
		alert("no sessions found on server for " + session_info[0]);
	// update combobox and button states
	var s = $("#my_sessions");
	var old_value = s.val();
	var found = false;
	s.empty();
	for (var i = 0; i != session_list.length; ++i) {
		var v = session_list[i].name;
		if (v == old_value)
			found = true;
		s.append($("<option/>").attr("value", v).text(v));
	}
	if (!found && session_list.length > 0)
		s.val(session_list[0].name);
	s.selectmenu("refresh", true);
}

function existing_session(name) {
	var session_list = $c2_session.session_list;
	for (var i = 0; i != session_list.length; ++i) {
		var s = session_list[i];
		if (name == s.name)
			return true;
	}
	return false;
}

function log(output) {
	$("#log").append(output);
}

$.extend($c2_session, {
	// Session User interface API

	// Public attributes
	user: "",
	session: "",
	password: "",
	session_list: [],

	// Public functions
	ui_init: ui_init,		// also initializes server part
	log: log,
});

}());

(function() {
"use strict";

// --------------------------------------------------------------------
// Higher level functions for communicating with server
// --------------------------------------------------------------------

var state = {};

function set_state(key, value) {
	// set state that should be saved when command is sent to server
	state[key] = value;
}

var data_functions = {};

function register_data_function(tag, func) {
	if (func === null || func === undefined) {
		delete data_functions[tag];
	} else {
		data_functions[tag] = func;
	}
}

function redistribute_data(data) {
	//console.log("redistribute_data " + data.length + " responses");
	for (var i in data) {
		var response = data[i];
		var call_id = response["id"];
		if (!response["status"]) {
			var msg = response["error"];
			if (msg !== undefined)
				if (msg.lastIndexOf("Traceback", 0) == 0) {
					msg = "Command failed:\n\n<pre>\n" + msg + "\n</pre>";
					// TODO: dialog
					show_error(msg);
				} else {
					show_error(msg);
				}
			continue;
		}
		if (response["command"]) {
			var msg = response["command"];
			$c2_session.log("<h3>" + msg + "</h3>");
		}
		var client_data = response["client_data"];
		if (client_data === undefined)
			continue;
		//console.log("  " + client_data.length + " results");
		for (var j in client_data) {
			var cd = client_data[j];
			var tag = cd[0];
			//console.log("  working on " + tag);
			if (!data_functions.hasOwnProperty(tag))
				continue;
			data_functions[tag](cd[1]);
		}
	}
}

function send_command(data) {
	// send command line to server
	if ($c2_session.session === "") {
		alert("Select or create a session first.");
		return;
	}
	$c2_session.server.call($c2_session.session, $c2_session.password,
			"command", data, state).done(redistribute_data);
}

// --------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------

$.extend($c2_session, {
	// User interface API
	register_data_function: register_data_function,
	send_command: send_command,
	set_state: set_state,
});

}());
