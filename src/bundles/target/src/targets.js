// vim: set expandtab shiftwidth=4 softtabstop=4:

var column_info = [
    "Name",
    "Info",
    "Atom/Bonds",
    "Ribbon",
    "Surface",
]

var tgttable = function() {
    var custom_scheme = "tgttable";
    var mouse_down_row = null;
    var mouse_down_index = null;
    var mouse_last_index = null;
    var mouse_down_toggle = null;
    var mouse_down_selected = null;

    function update_targets(targets) {
        // Clean up previous incarnation and save some state
        $("#targets_table").trigger("destroy");

        // Create table headers
        var thead = $("<thead/>");
        var row = $("<tr/>");
        for (var i = 0; i < column_info.length; i++) {
            row.append($("<th/>").text(column_info[i]));
            thead.append(row);
        }
        // Create table body
        var tbody = $("<tbody/>");
        $.each(targets, function(i, tgt) {
            var name = tgt["name"];
            var info = tgt["info"];
            var row = $("<tr/>", { class: "target_row",
                                   id: _row_id(name) });
            row.append($("<td/>").text(name));
            row.append($("<td/>").text(info));
            // TODO: add checkbox/color-selector pairs
            tbody.append(row);
        });
        $("#targets_table").empty().append(thead, tbody);

        // Re-setup jQuery handlers
        $("#targets_table").tablesorter({
            theme: 'blue',
            widgets: [ "resizable" ],
        });
    }

    // jQuery does not like '.' in id names even though JS does not care
    function _row_id(id) {
        return "row_" + id.replace('.', '_', 'g');
    }

    function init() {
        $("#nothing_btn").click(function() {
            window.location = custom_scheme + ":nothing";
        });
        $("#targets_table").tablesorter();
    }

    function get_state() {
        return {
            name:"tgttable",
        };
    }

    function set_state(state) {
    }

    return {
        custom_scheme: custom_scheme,
        update_targets: update_targets,
        get_state: get_state,
        set_state: set_state,
        init: init
    }
}();

$(document).ready(tgttable.init);
