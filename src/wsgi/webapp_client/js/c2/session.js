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
	var data = {
		action: "call",
		session: session,
		password: password,
		command: JSON.stringify(call_data),
	}
	return $.getJSON($c2_session.server.url, data)
			.done(debug_log);
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
	$("#my-sessions-input").on("change keyup paste click",
			update_session_buttons);
	$("#open-session").click(open_session);
	$("#delete-session").click(delete_session);
	$("#new-session").click(open_new_session);
	$("#refresh-session").click(update_session_list);
	$("#open-shared-session").click(open_shared_session);
	update_session_list();
}

function callback_failed(jqxhr, textStatus, error) {
	alert("Session request failed: " + textStatus + ", " + error);
}

function update_session_buttons(event)
{
	if (event !== undefined)
		event.preventDefault();
	//event.stopPropagation();
	var session = $("#my-sessions-input").val();
	if (existing_session(session)) {
		$("#new-session").addClass("ui-state-disabled");
		$("#new-session").removeClass("ui-btn-active");
		$("#open-session").addClass("ui-btn-active");
		$("#open-session").removeClass("ui-state-disabled");
		$("#delete-session").removeClass("ui-state-disabled");
		if (event !== undefined && event.keyCode === 13)
			open_session();
	} else {
		$("#new-session").removeClass("ui-state-disabled");
		$("#new-session").addClass("ui-btn-active");
		$("#open-session").removeClass("ui-btn-active");
		$("#open-session").addClass("ui-state-disabled");
		$("#delete-session").addClass("ui-state-disabled");
		if (event !== undefined && event.keyCode === 13)
			open_new_session();
	}
}

function open_session() {
	// verify and set default session parameters (name and password)
	var session = $("#my-sessions-input").val();
	var password = $("#my-password").val();
	if (!existing_session(session)) {
		alert("Session \"" + session + "\" does not exist.");
		return;
	}
	// alert("Session \"" + session + "\" selected.");
	save_session_info("", session, password);
	$("#popup-session-open").popup("close");
}

function save_session_info(user, session, password) {
	$c2_session.user = user;
	$c2_session.session = session;
	$c2_session.password = password;
	var msg = "No session selected";
	if (session)
		msg = "Active session: " + session;
	$("#active-session").html(msg);
}

function delete_session() {
	// Delete using session parameters (name and password)
	var session = $("#my-sessions-input").val();
	var password = $("#my-password").val();
	$c2_session.server.delete_session(session, password)
			.fail(callback_failed)
			.done(delete_session_cb);
	alert("Deleting session \"" + session + "\".");
}

function delete_session_cb() {
	var session = $("#my-sessions-input").val();
	var password = $("#my-password").val();
	alert("Session \"" + session + "\" deleted.");
	save_session_info("", "", "");
	$("#popup-session-open").popup("close");
}

function open_new_session() {
	// Create using session parameters (name and password)
	var session = $("#my-sessions-input").val();
	var password = $("#my-password").val();
	$c2_session.server.create_session(session, password)
			.fail(callback_failed)
			.done(open_new_session_cb);
}

function open_new_session_cb() {
	var session = $("#my-sessions-input").val();
	var password = $("#my-password").val();
	// alert("Session \"" + session + "\" created.");
	save_session_info("", session, password);
	$("#popup-session-open").popup("close");
}

function open_shared_session() {
	// Shared session parameters (user, session and password)
	var user = $("#shared-username").val();
	var session = $("#shared-session").val();
	var password = $("#shared-password").val();
	// TODO: verify session exists
}

function open_shared_session_cb() {
	var user = $("#shared-username").val();
	var session = $("#shared-session").val();
	var password = $("#shared-password").val(); // alert("Shared session \"" + session + "\" opened.");
	save_session_info(user, session, password);
	$("#popup-session-shared").popup("close");
}

function update_session_list(event) {
	// disable button and initiate AJAX to update list
	$c2_session.server.list_sessions()
			.fail(callback_failed)
			.done(update_session_list_cb);
}

function update_session_list_cb(session_info) {
	// save session information for later use
	$c2_session.user = session_info[0];
	var session_list = session_info[1];
	$c2_session.session_list = session_list;
	update_session_buttons();
	// sort sessions by name
	session_list.sort(function (a, b) {
		var x = a.name.toLowerCase();
		var y = b.name.toLowerCase();
		if (x < y)
			return -1;
		if (x > y)
			return 1;
		return 0;
	});
	// update listview
	var ul = $("#my-sessions");
	var html = "<thead><tr><th data-priority='persist'>Session Name</th><th>Last Accessed</th></tr></thead><tbody>";
	for (var i = 0; i != session_list.length; ++i) {
		var session = session_list[i];
		var name = session_list[i].name;
		//html += '<li class="ui-state-hidden" data-filtertext=' +
		//	JSON.stringify(session.name) + '>' +
		//	_.escape(session.name) + 
		//	" (" + _.escape(session.access) + ")</li>";
		html += "<tr><td><a href='#' class='ui-btn ui-mini no-margin' onclick='$(\"#my-sessions-input\").val($(this).html()).change()'>"
			+ _.escape(session.name) + "</a></td><td>"
			+ _.escape(session.access) + "</td>";
	}
	html += "</tbody></table>";
	ul.html(html);
	ul.filterable("refresh");
	//ul.trigger("updatelayout");
	$("#popup-session-open").trigger("updatelayout");
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

function log(output) { $("#log").append(output);
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
		if (response["command"]) {
			var msg = response["command"];
			$c2_session.log("<h3>" + msg + "</h3>");
		}
		if (!response["status"]) {
			var msg = response["error"];
			if (msg !== undefined)
				if (msg.lastIndexOf("Traceback", 0) == 0) {
					msg = "Command failed:\n\n" + msg + "\n";
					// TODO: dialog
					show_error("Traceback generated during backend processing.");
					alert(msg);
				} else {
					show_error(msg);
				}
			continue;
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

function callback_failed(jqxhr, textStatus, error) {
	alert("Commaned failed: " + textStatus + ", " + error);
}

function send_command(data) {
	// send command line to server
	if ($c2_session.session === "") {
		alert("Select or create a session first.");
		return;
	}
	$c2_session.server.call($c2_session.session, $c2_session.password,
				"command", data, state)
			.fail(callback_failed)
			.done(redistribute_data);
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
