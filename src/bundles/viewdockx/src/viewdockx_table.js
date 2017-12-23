// vim: set expandtab shiftwidth=4 softtabstop=4:

var vdxtable = function() {
    var custom_scheme = "vdxtable";
    var rating_column = "viewdockx_rating";

    function update_columns(columns) {
        // Clean up previous incarnation and save some state
        $("#viewdockx_table").trigger("destroy");
        var cols_hidden = $("#show_columns option:not(:selected)").map(
                            function() { return this.value; }).get();

        // Build column lists
        var numeric = columns["numeric"];
        var text = columns["text"];
        var ids = text["id"];
        // Create table headers
        var thead = $("<thead/>");
        var row = $("<tr/>");
        // column 1 is the sort/show column for numeric data
        row.append($("<th/>").text("S"));
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
            var row = $("<tr/>", { class: "structure_row",
                                   id: _row_id(id) });
            var query = "?id=" + id;
            var checkbox_url = custom_scheme + ":checkbox" + query;
            var link_url = custom_scheme + ":link" + query;
            row.append($("<td/>").append($("<input/>", {
                                            type: "checkbox",
                                            class: "structure",
                                            title: id,
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
                ratedFill: "#F36C12",
                normalFill: "#DDDDDD",
                onSet: function (r, inst) {
                    var ids;
                    if ($(this).parents(".selected").length > 0) {
                        // Already selected
                        ids = $("tr.selected .structure").map(
                                function () {
                                    return $(this).prop("title");
                                }).get().join();
                        if (event)
                            event.stopPropagation();
                    } else {
                        // Not yet selected
                        ids = $(this).prop("title");
                    }
                    var url = custom_scheme + ":rating?id=" + ids +
                              "&rating=" + r;
                    window.location = url;
                }
            });
        });

        // Rebuild column selector
        var opts = []
        $.each(text_order, function(n, key) {
            opts.push({
                name: key,
                value: key,
                checked: !cols_hidden.includes(key)
            });
        });
        $.each(numeric_order, function(n, key) {
            opts.push({
                name: key,
                value: key,
                checked: !cols_hidden.includes(key)
            });
        });
        $("#show_columns").multiselect("loadOptions", opts);
        $.each(cols_hidden, function(i, key) {
            var title = key.toUpperCase();
            var n = $("th").filter(function() {
                        return $(this).text() == title;
                    }).index() + 1;     // nth-child is 1-based
            $("td:nth-child(" + n + "),th:nth-child(" + n + ")").hide();
        })

        // Re-setup jQuery handlers
        $("#viewdockx_table").tablesorter({
            theme: 'blue',
            headers: {
                0: { sorter: false },
                1: { sorter: 'rating_col' },
                2: { sorter: 'id_col' }
            }
        });
        $(".structure").click(function(event) {
            var display = $(this).is(":checked") ? 1 : 0;
            var ids;
            if ($(this).parents(".selected").length > 0) {
                // Already selected
                ids = $("tr.selected .structure").map(
                        function () {
                            return $(this).prop("title");
                        }).get().join();
                event.stopPropagation();
            } else {
                // Not yet selected
                ids = $(this).prop("title");
            }
            var url = custom_scheme + ":checkbox?id=" + ids +
                      "&display=" + display;
            window.location = url;
        });
        $(".structure_row").click(function(e) {
            if (e.ctrlKey) {
                $(this).toggleClass("selected");
            } else {
                $("tr.selected").removeClass("selected");
                $(this).addClass("selected");
            }
        })
    }

    // jQuery does not like '.' in id names even though JS does not care
    function _row_id(id) {
        return "row_" + id.replace('.', '_', 'g');
    }
    function _checkbox_id(id) {
        return "cb_" + id.replace('.', '_', 'g');
    }
    function _rating_id(id) {
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
        $("#viewdockx_table").trigger("update");
    }

    function show_column(e, opt) {
        var title = opt.value.toUpperCase();
        var n = $("th").filter(function() {
                    return $(this).text() == title;
                }).index() + 1;     // nth-child is 1-based
        var col = $("td:nth-child(" + n + "),th:nth-child(" + n + ")");
        if (opt.checked)
            col.show();
        else
            col.hide();
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
        $.tablesorter.addParser({
            id: 'rating_col',
            is: function(s) {
                // return false so this parser is not auto detected
                return false;
            },
            format: function(s, table, cell, cellIndex) {
                var rt = $(cell).children("div");
                return rt.rateYo("option", "rating");
            },
            // set type, either numeric or text
            type: 'numeric'
        });

        $("#show_all_btn").click(function() {
            window.location = custom_scheme + ":check_all?show_all=true";
        });
        $("#graph_btn").click(function() {
            window.location = custom_scheme + ":graph";
        });
        $("#plot_btn").click(function() {
            window.location = custom_scheme + ":plot";
        });
        $("#hb_btn").click(function() {
            window.location = custom_scheme + ":hb";
        });
        $("#clash_btn").click(function() {
            window.location = custom_scheme + ":clash";
        });
        $("#export_btn").click(function() {
            window.location = custom_scheme + ":export";
        });
        $("#prune_stars").rateYo({
            starWidth: "16px",
            rating: 1,
            ratedFill: "#F36C12",
            normalFill: "#DDDDDD",
            fullStar: true
        });
        $("#prune_btn").click(function() {
            window.location = custom_scheme + ":prune?stars=" +
                              $("#prune_stars").rateYo("option", "rating");
        });
        $("#show_columns").multiselect({
            placeholder: "Columns...",
            onOptionClick: show_column
        });
        $("#viewdockx_table").tablesorter();
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
