// vim: set expandtab shiftwidth=4 softtabstop=4:

var vdxtable = function() {
    var custom_scheme = "vdxtable";
    var rating_column = "viewdockx_rating";
    var rating_clicked = false;
    var mouse_down_row = null;
    var mouse_down_index = null;
    var mouse_last_index = null;
    var mouse_down_toggle = null;
    var mouse_down_selected = null;

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
        // column 1 is for ratings
        row.append($("<th/>").text("RATING"));
        // column 2 is for ids
        row.append($("<th/>").text("ID"));
        // column 3 is for names, if present
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
                                   title: id,
                                   id: _row_id(id) });
            row.append($("<td/>").append($("<div/>", {
                                            class: "rating",
                                            title: id,
                                            id: _rating_id(id) })));
            row.append($("<td/>").text(id));
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
                        ids = $("tr.selected").map(
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
                0: { sorter: 'rating_col' },
                1: { sorter: 'id_col' }
            },
            widgets: [ "resizable" ],
        });
        function mouse_update(e) {
            var my_index = $(this).index();
            if (my_index == mouse_last_index)
                return;
            var anchor_index = mouse_down_index;
            var all;
            if (my_index == anchor_index)
                all = $(this);
            else if (my_index < anchor_index)
                all = $(this).nextUntil(mouse_down_row)
                             .addBack().add(mouse_down_row);
            else
                all = $(this).prevUntil(mouse_down_row)
                             .addBack().add(mouse_down_row);
            $("tr.selected").removeClass("selected");
            if (mouse_down_toggle) {
                mouse_down_selected.addClass("selected");
                all.toggleClass("selected");
            } else {
                all.addClass("selected");
            }
            mouse_last_index = my_index;
        }
        function update_shown() {
            ids = $("tr.selected").map(function () {
                            return $(this).prop("title");
            }).get().join();
            var url = custom_scheme + ":show_only?id=" + ids;
            window.location = url;
        }
        $(".structure_row").mousedown(function(e) {
            if (e.which != 1)   // Ignore if not left mouse
                return;
            rating_clicked = false;
            mouse_down_row = $(this);
            mouse_down_index = mouse_down_row.index();
            mouse_last_index = null;
            // Windows/Linux want ctrl-key, Mac wants cmd-key
            mouse_down_toggle = e.ctrlKey || e.metaKey;
            if (mouse_down_toggle)
                mouse_down_selected = $("tr.selected");
            else
                mouse_down_selected = null;
            // Do not actually change selection in case user
            // clicks on link or checkbox that need to apply
            // to all selected rows.  mouse_last_index is set
            // to null, so any mouse movement will trigger
            // selection update.
            $(".structure_row").on("mousemove", mouse_update);
        });
        $(".rating").mouseup(function(e) {
            if ($(this).parent().parent().hasClass("selected"))
                rating_clicked = true;
        });
        $(".structure_row").mouseup(function(e) {
            if (e.which != 1)   // Ignore if not left mouse
                return;
            $(".structure_row").off("mousemove");
            if (!rating_clicked) {
                if (mouse_last_index == null)
                    mouse_update(e);
                update_shown();
            }
            rating_clicked = false;
            mouse_down_row = null;
            mouse_down_index = null;
            mouse_last_index = null;
            mouse_down_toggle = null;
            mouse_down_selected = null
        });
    }

    // jQuery does not like '.' in id names even though JS does not care
    function _row_id(id) {
        return "row_" + id.replace('.', '_', 'g');
    }
    function _rating_id(id) {
        return "rt_" + id.replace('.', '_', 'g');
    }

    function update_display(new_display) {
        for (var i = 0; i < new_display.length; i++) {
            var id = new_display[i][0];
            if (new_display[i][1])
                $("#" + _row_id(id)).addClass("selected");
            else
                $("#" + _row_id(id)).removeClass("selected");
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
            window.location = custom_scheme + ":show_all";
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
            texts: {
                placeholder: "Display columns...",
                selectedOptions: " columns displayed (click to change)",
                noneSelected: "No columns displayed (click to change)",
            },
            minHeight: 10,
            onOptionClick: show_column
        });
        $("#viewdockx_table").tablesorter();
    }

    function get_state() {
        var cols_hidden = $("#show_columns option:not(:selected)").map(
                            function() { return this.value; }).get();
        var cols_shown = $("#show_columns option:selected").map(
                            function() { return this.value; }).get();
        return {
            name:"vdxtable",
            cols_hidden:cols_hidden,
            cols_shown:cols_shown
        };
    }

    function set_state(state) {
        $("#show_columns+div :checkbox").each(function(n, opt) {
            if (opt.checked) {
                if ($.inArray(opt.value, state.cols_hidden) !== -1)
                    $(opt).trigger("click");
            } else {
                if ($.inArray(opt.value, state.cols_show) !== -1)
                    $(opt).trigger("click");
            }
        });
    }

    return {
        custom_scheme: custom_scheme,
        update_columns: update_columns,
        update_display: update_display,
        update_ratings: update_ratings,
        get_state: get_state,
        set_state: set_state,
        init: init
    }
}();

$(document).ready(vdxtable.init);
