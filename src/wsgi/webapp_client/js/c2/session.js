// --------------------------------------------------------------------
// Session Dialog functions
// --------------------------------------------------------------------

function _c2sd_init(url) {
	// initialize user interface (session button and dialog)
	_c2s_init(url);

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
	$("#c2sd_user").keyup(_c2sd_session_cb);
	$("#c2sd_password").addClass("ui-widget-content ui-corner-all");
	$("#c2sd_session").css("min-width", "100px")
			.jec({ triggerChangeEvent: true,
				handleCursor: true })
			.change(_c2sd_session_cb);
	_c2sd_update_session_list();
}

_c2sd_content =
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
	_c2s_create_session(session, password, _c2sd_create_session_cb);
}

function _c2sd_create_session_cb() {
	var session = $("#c2sd_session").val();
	var password = $("#c2sd_password").val();
	// alert("Session \"" + session + "\" created.");
	_c2sd_save_session_info(session, password);
	_c2sd_session_cb();
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
	_c2s_delete_session(session, password, _c2sd_create_session_cb);
}

function _c2sd_delete_session_cb() {
	var session = $("#c2sd_session").val();
	var password = $("#c2sd_password").val();
	alert("Session \"" + session + "\" deleted.");
	_c2sd_save_session_info(session, password);
	_c2sd_session_cb();
}

function _c2sd_update_session_list() {
	// disable button and initiate AJAX to update list
	$("#c2s_button").attr("disabled", true);
	_c2s_list_sessions(_c2sd_update_cb)
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
	_c2sd_session_cb();
	$("#c2s_button").attr("disabled", false);
}

function _c2sd_button(name, action) {
	var selector = "#c2s_dialog + " +
			".ui-dialog-buttonpane " +
			"button:contains('" + name + "')";
	$(selector).button(action);
}

function _c2sd_session_cb() {
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

// --------------------------------------------------------------------
// Functions for communicating with server
// --------------------------------------------------------------------

function _c2s_init(url) {
	$c2_session.server.url = url;
}

function _c2s_list_sessions(cb) {
	// Retrieve list of sessions from server
	if ($c2_session.server.url == "") {
		alert("Session module is uninitialized.");
		return;
	}
	var data = {
		action: "jlist",
	}
	return $.getJSON($c2_session.server.url, data, cb);
}

function _c2s_create_session(session, password, cb) {
	// Create session on server
	if ($c2_session.server.url == "") {
		alert("Session module is uninitialized.");
		return;
	}
	var data = {
		action: "create",
		session: session,
		password: password,
	}
	return $.get($c2_session.server.url, data, cb);
}

function _c2s_delete_session(session, password, cb) {
	// Create session on server
	if ($c2_session.server.url == "") {
		alert("Session module is uninitialized.");
		return;
	}
	var data = {
		action: "delete",
		session: session,
		password: password,
	}
	return $.get($c2_session.server.url, data, cb);
}

function _c2s_call(session, password, tag, tag_data, cb) {
	var cid = $c2_session._call_id;
	$c2_session._call_id = cid + 1;
	var call_data = [ [ cid, tag, tag_data ] ];
	// alert("_c2s_call: " + JSON.stringify(call_data));
	$c2_session._call_callbacks[cid] = cb;
	function clear_callback() {
		alert("getJSON failed");
		delete $c2_session._call_callbacks[cid];
	}
	var data = {
		action: "call",
		session: session,
		password: password,
		command: JSON.stringify(call_data),
	}
	return $.getJSON($c2_session.server.url, data, _c2s_call_cb)
			.error(clear_callback);
}

function _c2s_call_cb(data) {
	output = JSON.stringify(data) + "\n";
	for (var index in data) {
		output = output + "Response " + index + "\n";
		response = data[index];
		for (var key in response)
			output = output + "  " + key + ": "
						+ response[key] + "\n";
		// redistribute data
		var callback = $c2_session._call_callbacks[response.id];
		callback(response);
		delete $c2_session._call_callbacks[response.id];
	}
	$("#debug").text(output);
}

// --------------------------------------------------------------------
// Higher level functions for communicating with server
// --------------------------------------------------------------------

function _c2s_command(data, cb) {
	_c2s_call($c2_session.session, $c2_session.password,
			"command", data, cb);
}

// --------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------

$c2_session = {
	// Server API
	server: {
		url: "",
		init: _c2s_init,
		list_sessions: _c2s_list_sessions,
		create_session: _c2s_create_session,
		delete_session: _c2s_delete_session,
		call: _c2s_call,
	},

	// User interface API

	// Public attributes
	user: "",
	session: "",
	password: "",
	session_list: [],

	// Private attributes
	_call_id: 1,
	_call_callbacks: {},

	// Public functions
	init: _c2sd_init,
	command: _c2s_command,
}
