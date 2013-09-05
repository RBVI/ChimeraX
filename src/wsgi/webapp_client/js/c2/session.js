// --------------------------------------------------------------------
// Session Dialog functions
// --------------------------------------------------------------------

// public API
var $c2_session = {};

(function() {
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
		return;
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
		return;
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
		return;
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
		return;
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
	var output = JSON.stringify(data) + "\n";
	for (var index in data) {
		output = output + "Response " + index + "\n";
		var response = data[index];
		for (var key in response)
			output = output + "  " + key + ": "
						+ response[key] + "\n";
	}
	$("#debug").text(output);
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

// c2sd stands for Chimera2 Session Dialog

function ui_init(url) {
	// initialize user interface (session button and dialog)
	$c2_session.server.init(url);

	// initialize public data elements
	$c2_session.user = "";
	$c2_session.session = "";
	$c2_session.password = "";

	// Create session button
	var d = $("#c2_session").prepend(
		"<span id=\"c2s_button\">Session: none selected</span>" +
		"<div id=\"c2s_dialog\"/>");
	$("#c2s_button").button().css("color", "red").click(_c2sd_show);
	// Create dialog that session button displays
	$("#c2s_dialog").dialog({
		autoOpen: false,
		draggable: false,
		show: "blind",
		hide: "blind",
		width: "auto",
		autoResize: true,
		position: {
			my: "left top",
			at: "left top",
			of: d,
		},
		title: "Select Session",
		buttons: {
			Select: _c2sd_select_session,
			Create: _c2sd_create_session,
			Delete: _c2sd_delete_session,
		},
		focus: function() { $("#c2sd_session").focus(); },
	}).html(_c2sd_content);
	$("#c2sd_user").keyup(update_dialog_buttons);
	$("#c2sd_password").addClass("ui-widget-content ui-corner-all");
	$("#c2sd_session").css("min-width", "100px")
			.jec({ triggerChangeEvent: true,
				handleCursor: true })
			.change(update_dialog_buttons);
	_c2sd_update_session_list();
}

var _c2sd_content =
'<table>' +
'<tr><th align="right">User:</th>' +
	'<td><input id="c2sd_user" type="text"/></td></tr>' +
'<tr><th align="right">Session:</th>' +
	'<td><select id="c2sd_session"/></td></tr>' +
'<tr><th align="right">Password:</th>' +
	'<td><input id="c2sd_password" type="password"/></td></tr>' +
'</table>';

function _c2sd_show() {
	// display session selection dialog
	$("#c2s_dialog").dialog("open");
}

function _c2sd_select_session() {
	// verify and set default session parameters (name and password)
	var user = $("#c2sd_user").val();
	var session = $("#c2sd_session").val();
	var password = $("#c2sd_password").val();
	if (user != $c2_session.user) {
		// TODO: verify that given session/password is valid
	} else if (!_c2sd_existing_session(session)) {
		alert("Session \"" + session + "\" does not exist.");
		return;
	}
	// alert("Session \"" + session + "\" selected.");
	_c2sd_save_session_info(session, password);
}

function _c2sd_save_session_info(session, password) {
	$c2_session.session = session;
	$c2_session.password = password;
	$("#c2s_button").css("color", "")
			.button("option", "label", "Session: " + session);
	$("#c2s_dialog").dialog("close");
}

function _c2sd_create_session() {
	// Create using session parameters (name and password)
	var user = $("#c2sd_user").val();
	if (user != $c2_session.user) {
		alert("You can only create your own sessions.");
		return;
	}
	var session = $("#c2sd_session").val();
	var password = $("#c2sd_password").val();
	_c2sd_button("Create", "disable");
	$c2_session.server.create_session(session, password).done(c2sd_create_session_cb);
}

function _c2sd_create_session_cb() {
	var session = $("#c2sd_session").val();
	var password = $("#c2sd_password").val();
	// alert("Session \"" + session + "\" created.");
	_c2sd_save_session_info(session, password);
	update_dialog_buttons();
}

function _c2sd_delete_session() {
	// Delete using session parameters (name and password)
	var user = $("#c2sd_user").val();
	if (user != $c2_session.user) {
		alert("You can only delete your own sessions.");
		return;
	}
	var session = $("#c2sd_session").val();
	var password = $("#c2sd_password").val();
	_c2sd_button("Delete", "disable");
	$c2_session.server.delete_session(session, password).done(_c2sd_create_session_cb);
}

function _c2sd_delete_session_cb() {
	var session = $("#c2sd_session").val();
	var password = $("#c2sd_password").val();
	alert("Session \"" + session + "\" deleted.");
	_c2sd_save_session_info(session, password);
	update_dialog_buttons();
}

function _c2sd_update_session_list() {
	// disable button and initiate AJAX to update list
	$("#c2s_button").attr("disabled", true);
	$c2_session.server.list_sessions().done(_c2sd_update_cb)
}

function _c2sd_existing_session(name) {
	var session_list = $c2_session.session_list;
	for (var i = 0; i != session_list.length; ++i) {
		var s = session_list[i];
		if (name == s.name)
			return true;
	}
	return false;
}

function _c2sd_update_cb(session_info) {
	// save session information for later use
	$c2_session.user = session_info[0];
	var session_list = session_info[1];
	$c2_session.session_list = session_list;
	if (session_list.length == 0)
		alert("no sessions found on server for " + session_info[0]);
	// update user name
	$("#c2sd_user").attr("value", $c2_session.user);
	// update combobox and button states
	var s = $("#c2sd_session");
	s.jecOff();
	s.empty();
	for (var i = 0; i != session_list.length; ++i) {
		var v = session_list[i].name;
		s.append($("<option/>").attr("value", v).text(v));
	}
	s.jecOn();
	if (session_list.length == 1 && !$c2_session.session)
		s.jecValue(session_list[0].name, true);
	else
		s.jecValue($c2_session.session, true);
	update_dialog_buttons();
	$("#c2s_button").attr("disabled", false);
}

function _c2sd_button(name, action) {
	var selector = "#c2s_dialog + " +
			".ui-dialog-buttonpane " +
			"button:contains('" + name + "')";
	$(selector).button(action);
}

function update_dialog_buttons() {
	// Enable/disable dialog buttons based on whether
	// the current session name is valid on the server.
	// Enable/disable Create button if Select is disabled
	// and session name is not empty.
	var user = $("#c2sd_user").val();
	var session = $("#c2sd_session").val();
	if (user == "" || session == "") {
		// No user or session name, no action possible
		_c2sd_button("Select", "disable");
		_c2sd_button("Create", "disable");
		_c2sd_button("Delete", "disable");
	}
	else if (user != $c2_session.user) {
		// Not login user, no delete or create
		_c2sd_button("Select", "enable");
		_c2sd_button("Create", "disable");
		_c2sd_button("Delete", "disable");
	}
	else if (_c2sd_existing_session(session)) {
		// Real session, select or delete
		_c2sd_button("Select", "enable");
		_c2sd_button("Create", "disable");
		_c2sd_button("Delete", "enable");
	}
	else {
		// New session, create only
		_c2sd_button("Select", "disable");
		_c2sd_button("Create", "enable");
		_c2sd_button("Delete", "disable");
	}
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
	console.log("redistribute_data " + data.length + " responses");
	for (var i in data) {
		var response = data[i];
		var call_id = response["id"];
		if (!response["status"]) {
			var msg = response["stderr"];
			if (msg !== undefined)
				if (msg.lastIndexOf("Traceback", 0) == 0) {
					msg = "Command failed:\n\n" + msg;
					alert(msg);
				} else {
					// TODO: status line message
					alert(msg);
				}
			continue;
		}
		var client_data = response["client_data"];
		if (client_data === undefined)
			continue;
		console.log("  " + client_data.length + " results");
		for (var j in client_data) {
			var data = client_data[j];
			var tag = data[0];
			console.log("  working on " + tag);
			if (!data_functions.hasOwnProperty(tag))
				continue;
			data_functions[tag](data[1]);
		}
	}
}

function send_command(data) {
	// send command line to server
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
