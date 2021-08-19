// vim: set expandtab shiftwidth=4 softtabstop=4:

var vdxtable = function() {
    var custom_scheme = "vdxtable";
    var table_options = {
        selection: true,
        rowSelect: false,
        multiSelect: true,
        keepSelection: true,
        rowCount: -1,       // Always show everyting on one page
        formatters: {
            "rating": function(column, row) {
                return '<input type="number" ratingid="' +
                       row.id + '" class="rating"/>';
            },
        },
    };
    var rating_options = {
        showCaption: false,
        showClear: false,
        animate: false,
        size: "xs",
        step: 1,
        hoverEnabled: false
    };
    var hidden = {};

    function is_visible(id) {
        return hidden[id] == null;
    }

    function update_columns(columns) {
        // Clean up previous incarnation and save some state.
        var preserve_hidden = false;
        if ($("#viewdockx_table").bootgrid("getTotalRowCount") > 0) {
            // Only save column state if there was something there before.
            // Also preserves "hidden" if we were just restored.
            preserve_hidden = true;
            hidden = {};
            var settings = $("#viewdockx_table").bootgrid("getColumnSettings");
            for (var i = 0; i < settings.length; i++) {
                var s = settings[i];
                if (!s.visible)
                    hidden[s.id] = true;
            }
        }

        // Bootgrid actually replaces the initial HTML table
        // with its own structure.  To start fresh, we need
        // to ask bootgrid to destroy itself, which restores
        // the original HTML table, and then empty out the
        // table.  We can then add back columns, reinitialize
        // bootgrid, and add rows.
        $("#viewdockx_table").bootgrid("destroy");
        var table = $("#viewdockx_table").empty();
        var numeric = columns["numeric"];
        var text = columns["text"];
        var ids = text["id"];

        // If we are not preserving column visibility (probably
        // first time through), then look at text columns and
        // hide those with unacceptably long values
        if (!preserve_hidden)
            $.each(text, function(key, v) {
                if (key != "id" && key != "name") {
                    var num_bad = 0;
                    var limit = Math.min(v.length, 10);
                    for (var i = 0; i < limit; i++)
                        if (v[i].length > 50)
                            num_bad++;
                    if (num_bad > limit / 2)
                        hidden[key] = true;
                }
            });

        // Build column lists
        var thead = $("<thead/>").appendTo(table);
        var htr = $("<tr/>").appendTo(thead);

        // Create table headers
        $("<th/>", { "data-column-id": "rating",
                     "data-visible": is_visible("rating"),
                     "data-formatter": "rating",
                     "data-sortable": false,
                     "data-searchable": false })
            .text("RATING").appendTo(htr);
        $("<th/>", { "data-column-id": "id",
                     "data-visible": is_visible("id"),
                     "data-identifier": true,
                     "data-sortable": false })
            .text("ID").appendTo(htr);
        var names = text["name"];
        if (names != null)
            $("<th/>", { "data-column-id": "name" })
                .text("NAME").appendTo(htr);
        $.each(text, function(key, v) {
            if (key != "id" && key != "name")
                $("<th/>", { "data-column-id": key,
                             "data-visible": is_visible(key) })
                    .text(key.toUpperCase()).appendTo(htr);
        });
        $.each(numeric, function(key, v) {
            if (key != "viewdockx_rating")
                $("<th/>", { "data-column-id": key,
                             "data-visible": is_visible(key),
                             "data-converter": "numeric" })
                    .text(key.toUpperCase()).appendTo(htr);
        });

        // Create table rows
        rows = []
        $.each(ids, function(i, id) {
            var row = { "row_id": i };
            $.each(text, function(key, v) {
                row[key] = v[i];
            });
            $.each(numeric, function(key, v) {
                row[key] = v[i];
            });
            rows.push(row);
        });
        function change_rating(ev, value, caption) {
            var id = $(this).attr("ratingid");
            var url = custom_scheme + ":rating?id=" + id + "&rating=" + value;
            window.location = url;
        }
        table.append($("<tbody/>"))
             .bootgrid(table_options)
             .on("selected.rs.jquery.bootgrid", update_shown)
             .on("deselected.rs.jquery.bootgrid", update_shown)
             .on("click.rs.jquery.bootgrid", function(ev, cols, row) {
                 table.bootgrid("deselect");
                 table.bootgrid("select", [ row.id ]);
             })
             .on("loaded.rs.jquery.bootgrid", function(ev) {
                 $('.rating').rating(rating_options)
                             .on("rating:change", change_rating);
                 window.location = custom_scheme + ":columns_updated";
             })
             .bootgrid("append", rows)
             .attr("tabindex", 0)           // make table accept focus
             .keydown(function(ev) {
                switch (ev.which) {
                  case 38:
                      process_arrow_key("up");
                      break
                  case 40:
                      process_arrow_key("down");
                      break
                  default:
                      return;
                }
                ev.preventDefault();
             });
    }

    function process_arrow_key(direction) {
        var url = custom_scheme + ":arrow?direction=" + direction;
        window.location = url;
    }

    function arrow_key(offset) {
        var table = $("#viewdockx_table");
        var grid = table.bootgrid().data('.rs.jquery.bootgrid');
        var selected = table.bootgrid("getSelectedRows");
        var rows = grid.currentRows;
        var indices = [];
        for (var i = 0; i < rows.length; i++) {
            if (selected.includes(rows[i].id))
                indices.push(i);
        }
        var newsel = [];
        for (var i = 0; i < indices.length; i++) {
            var n = indices[i] + offset;
            if (n >= 0 && n < rows.length)
                newsel.push(rows[n].id);
        }
        if (newsel.length > 0) {
            var url = custom_scheme + ":show_only?id=" + newsel;
            window.location = url;
        }
    }

    function update_shown() {
        var selected = $("#viewdockx_table").bootgrid("getSelectedRows");
        var url = custom_scheme + ":show_only?id=" + selected;
        window.location = url;
    }

    function update_display(new_display) {
        var select_ids = [];
        var deselect_ids = [];
        for (var i = 0; i < new_display.length; i++) {
            var id = new_display[i][0];
            if (new_display[i][1])
                select_ids.push(id);
            else
                deselect_ids.push(id);
        }
        var table = $("#viewdockx_table");
        table.off("selected.rs.jquery.bootgrid")
             .off("deselected.rs.jquery.bootgrid");
        if (select_ids.length > 0)
            table.bootgrid("select", select_ids);
        if (deselect_ids.length > 0)
            table.bootgrid("deselect", deselect_ids);
        table.on("selected.rs.jquery.bootgrid", update_shown)
             .on("deselected.rs.jquery.bootgrid", update_shown);
        var active_rows = $("tr.active");
        if (active_rows.length > 0) {
            var top_row = active_rows.first();
            if (!is_element_in_view(top_row, false))
                scroll_to_visible(top_row);
        }
    }

    function update_ratings(new_ratings) {
        for (var i = 0; i < new_ratings.length; i++) {
            var id = new_ratings[i][0];
            var rating = new_ratings[i][1];
            var r = $('.rating[ratingid="' + id + '"]');
            if (r.val() != rating)
                r.rating("update", rating);
        }
        $("#viewdockx_table").trigger("update");
    }

    function is_element_in_view(element, fully_in_view) {
        // From https://stackoverflow.com/questions/487073/how-to-check-if-element-is-visible-after-scrolling
        var pageTop = $(window).scrollTop();
        var pageBottom = pageTop + $(window).height();
        var elementTop = $(element).offset().top;
        var elementBottom = elementTop + $(element).height();
        if (fully_in_view)
            return pageTop < elementTop && pageBottom > elementBottom;
        else
            return pageTop <= elementTop && pageBottom >= elementBottom;
    }

    function scroll_to_visible(element) {
        var page_height = $(window).height();
        var page_top = $(window).scrollTop();
        var page_bottom = page_top + page_height;
        var element_height = $(element).height();
        var element_top = $(element).offset().top;
        var element_bottom = element_top + element_height;
        var top;
        if (element_bottom > page_bottom)
            $("html, body").animate({scrollTop: element_bottom - page_height});
        else if (element_top < page_top)
            $("html, body").animate({scrollTop: element_top});
    }

    function init() {
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
        $("#prune_stars").rating(rating_options);
        $("#prune_btn").click(function() {
            window.location = custom_scheme + ":prune?stars=" + $("#prune_stars").val();
        });
    }

    function get_state() {
        return {
            name:"vdxtable",
            hidden: hidden
        };
    }

    function set_state(state) {
        if (state.hidden != null)
            hidden = state.hidden;
    }

    return {
        custom_scheme: custom_scheme,
        update_columns: update_columns,
        update_display: update_display,
        update_ratings: update_ratings,
        arrow_key: arrow_key,
        get_state: get_state,
        set_state: set_state,
        init: init
    }
}();

$(document).ready(vdxtable.init);
