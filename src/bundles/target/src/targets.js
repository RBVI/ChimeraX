// vim: set expandtab shiftwidth=4 softtabstop=4:

var tgttable = function() {
    var custom_scheme = "target";
    var mouse_down_row = null;
    var mouse_down_index = null;
    var mouse_last_index = null;
    var mouse_down_toggle = null;
    var mouse_down_selected = null;

    function update_columns(columns) {
        // Clean up previous incarnation and save some state
        $("#targets_table").trigger("destroy");

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
        $("#targets_table").empty().append(thead, tbody);

        // Re-setup jQuery handlers
        $("#targets_table").tablesorter({
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

    function init() {
        $("#show_all_btn").click(function() {
            window.location = custom_scheme + ":show_all";
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
        update_columns: update_columns,
        get_state: get_state,
        set_state: set_state,
        init: init
    }
}();

$(document).ready(tgttable.init);
