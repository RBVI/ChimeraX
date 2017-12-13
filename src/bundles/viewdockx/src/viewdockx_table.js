// vim: set expandtab shiftwidth=4 softtabstop=4:

var vdxtable = function() {
    var custom_scheme = "vdxtable";
    var rating_column = "viewdockx_rating"

    function update_columns(columns) {
        var numeric = columns["numeric"];
        var text = columns["text"];
        var ids = text["id"];
        // Create table headers
        var thead = $("<thead/>");
        var row = $("<tr/>");
        // column 1 is the sort/show column for numeric data
        row.append($("<th/>"));
        // column 2 is for ratings
        row.append($("<th/>").text("RATING"));
        // column 3 is for ids
        row.append($("<th/>").text("ID"));
        // column 4 is for names, if present
        var names = text["name"];
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
            if (key != rating_column) {
                row.append($("<th/>").text(key.toUpperCase()));
                numeric_order.push(key);
            }
        });
        thead.append(row);
        // Create table body
        var tbody = $("<tbody/>");
        $.each(ids, function(i, id) {
            var row = $("<tr/>");
            var query = "?id=" + id;
            var checkbox_url = custom_scheme + ":checkbox" + query;
            var link_url = custom_scheme + ":link" + query;
            row.append($("<td/>").append($("<input/>", {
                                            type: "checkbox",
                                            class: "structure",
                                            id: _checkbox_id(id),
                                            href: checkbox_url })));
            row.append($("<td/>").append($("<div/>", {
                                            title: id,
                                            id: _rating_id(id) })));
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
        var ratings = numeric[rating_column];
        $.each(ids, function(i, id) {
            $("#" + _rating_id(id)).rateYo({
                starWidth: "10px",
                rating: ratings[i],
                fullStar: true,
                onSet: function (r, inst) {
                    window.location = custom_scheme + ":rating" +
                                      "?id=" + this.title +
                                      "&rating=" + r;
                }
            });
        });

        // Re-setup jQuery handlers
        $("#viewdockx_table").tablesorter({
            theme: 'blue',
            headers: {
                0: { sorter: false },
                1: { sorter: false },
                2: { sorter: 'id_col' }
            }
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

    function _rating_id(id) {
        // jQuery does not like '.' in id names even though JS does not care
        return "rt_" + id.replace('.', '_', 'g');
    }

    function update_display(new_display) {
        for (var i = 0; i < new_display.length; i++) {
            var id = new_display[i][0];
            var checked = new_display[i][1];
            $("#" + _checkbox_id(id)).prop("checked", checked);
        }
    }

    function update_ratings(new_ratings) {
        for (var i = 0; i < new_ratings.length; i++) {
            var id = new_ratings[i][0];
            var rating = new_ratings[i][1];
            var r = $("#" + _rating_id(id));
            if (r.rateYo("option", "rating") != rating)
                r.rateYo("option", "rating", rating);
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
                // Assume less than 100,000 submodels
                var parts = s.split("\n")[0].trim().split(".");
                return parseInt(parts[0]) * 100000 + parseInt(parts[1]);
            },
            // set type, either numeric or text
            type: 'numeric'
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
        update_ratings: update_ratings,
        init: init
    }
}();

$(document).ready(vdxtable.init);
