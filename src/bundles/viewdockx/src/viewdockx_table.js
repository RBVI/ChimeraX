// vim: set expandtab shiftwidth=4 softtabstop=4:

var vdxtable = function() {
    var custom_scheme = "vdxtable";

    function update_columns(columns) {
        var numeric = columns["numeric"];
        var text = columns["text"];
        var names = text["name"];
        // Create table headers
        var thead = $("<thead/>");
        var row = $("<tr/>");
        // column 1 is the sort/show column for numeric data
        row.append($("<th/>"));
        // column 2 is for ids
        row.append($("<th/>").text("ID"));
        // column 3 is for names, if present
        if (names != null)
            row.append($("<th/>").text("NAME"));
        var text_order = ["id", "name"];
        $.each(text, function(key, v) {
            if (key != "id" && key != "name") {
                row.append($("<th/>").text(key.toUpperCase()));
                text_order.push(key);
            }
        });
        var numeric_order = [];
        $.each(numeric, function(key, v) {
            row.append($("<th/>").text(key.toUpperCase()));
            numeric_order.push(key);
        });
        thead.append(row);
        // Create table body
        var tbody = $("<tbody/>");
        $.each(text["id"], function(i, id) {
            var row = $("<tr/>");
            var query = "?id=" + id;
            var checkbox_url = custom_scheme + ":checkbox" + query;
            var link_url = custom_scheme + ":link" + query;
            row.append($("<td/>").append($("<input/>", {
                                            type: "checkbox",
                                            class: "structure",
                                            id: _checkbox_id(id),
                                            href: checkbox_url })));
            row.append($("<td/>").append($("<a/>", { href: link_url })
                                            .text(id)));
            if (names != null)
                row.append($("<td/>").text(names[i]));
            $.each(text_order, function(n, key) {
                if (key != "id" && key != "name")
                    row.append($("<td/>").text(text[key][i]));
            });
            $.each(numeric_order, function(n, key) {
                row.append($("<td/>").text(numeric[key][i]));
            });
            tbody.append(row);
        });
        $("#viewdockx_table").empty().append(thead, tbody);

        // Re-setup jQuery handlers
        $("#viewdockx_table").tablesorter({
            theme: 'blue',
            headers: { 1: { sorter: 'id_col' } }
        });
        $(".structure").click(function() {
            if ($(this).is(":checked")) {
                window.location = $(this).attr('href') + "&display=1";
            } else {
                window.location = $(this).attr('href') + "&display=0";
            }
        });
    }

    function _checkbox_id(id) {
        // jQuery does not like '.' in id names even though JS does not care
        return "cb_" + id.replace('.', '_', 'g');
    }

    function update_display(new_display) {
        for (var i = 0; i < new_display.length; i++) {
            var id = new_display[i][0];
            var checked = new_display[i][1];
            $("#" + _checkbox_id(id)).prop("checked", checked);
        }
    }

    function init() {
        $.tablesorter.addParser({
            id: 'id_col',
            is: function(s) {
                // return false so this parser is not auto detected
                return false;
            },
            format: function(s) {
                // Assume less than 100,000 models or submodels
                // and convert id to 10-character zero-padded string
                // which sorts correctly as text
                var pad = "00000";
                var padlen = pad.length;
                var parts = s.split("\n")[0].trim().split(".");
                var n = pad.substring(0, padlen - parts[0].length) + parts[0]
                      + pad.substring(0, padlen - parts[1].length) + parts[1];
                return n;

            },
            // set type, either numeric or text
            type: 'text'
        });

        $("#show_all_btn").click(function() {
            window.location = custom_scheme + ":check_all?show_all=true";
        });
        $('#chart_btn').on('click', function() {
            window.location = custom_scheme + ":chart";
        });
        $('#plot_btn').on('click', function() {
            window.location = custom_scheme + ":plot";
        });
        $('#histogram_btn').on('click', function() {
            window.location = custom_scheme + ":histogram";
        });
    }

    return {
        custom_scheme: custom_scheme,
        update_columns: update_columns,
        update_display: update_display,
        init: init
    }
}();

$(document).ready(vdxtable.init);
