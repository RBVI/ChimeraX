// vim: set expandtab shiftwidth=4 softtabstop=4:

var vdxchart = function() {
    var columns = { numeric: {}, text:{} };
    var display = [];
    var id2index = {};              // structure id -> sorted data index
    var index2index = {};           // sorted data index -> raw data index
    var tooltip_shown = false;
    var shift_down = false;
    var custom_scheme = "vdxchart";
    var plot = null;

    function make_button(btype, name, value, text, checked) {
        return $("<label/>").text(text).append(
                    $("<input/>", { type: btype,
                                    name: name,
                                    value: value,
                                    class: name,
                                    checked: checked }).click(update_plot));
    }

    function update_columns(new_columns) {
        columns = new_columns;
        // Save sort and shown columns
        var sort_column = $(".sort:checked").attr("value");
        var show_columns = []
        $(".display:checked").map(function() {
            if (this.value in columns["numeric"])
                show_columns.push(this.value);
        });
        if (show_columns.length == 0)
            show_columns.push(Object.keys(columns["numeric"])[0])

        // Clear out the table and fill in with text
        // then numeric column names
        $("#column_table").empty();
        $.each(columns["text"], function(r, v) {
            $("#column_table").append($("<tr/>").append(
                                        $("<td/>"),
                                        $("<td/>").text(r),
                                        $("<td/>").addClass("value")
                                                  .prop("title", r)))
            });
        $.each(columns["numeric"], function(r, v) {
            var sort_btn = make_button("radio", "sort", r, "S",
                                       r == sort_column);
            var show_btn = make_button("checkbox", "display", r, "D",
                                       show_columns.includes(r));
            $("#column_table").append($("<tr/>").append(
                                        $("<td/>").append(sort_btn, show_btn),
                                        $("<td/>").text(r),
                                        $("<td/>").addClass("value")
                                                  .prop("title", r)))
            });
        update_plot();
    }

    function update_plot() {
        // parameters for flot
        var series = [];
        var opts = {
            series: {
                points: { show: true },
                lines: { show: true }
            },
            grid: {
                autoHighlight: false,
                hoverable: true,
                clickable: true
            },
            xaxes: [
                { },
                { show: false }
            ],
        }

        // Get order of compounds based on sort column
        var text = columns["text"];
        var numeric = columns["numeric"];
        var ids = text["id"];
        var order = ids.map(function(e, i) { return i; });
        var sort_column = $(".sort:checked").attr("value");
        if (sort_column != null) {
            var data = numeric[sort_column];
            order.sort(function(a, b) {
                if (data[a] == null)
                    return 1;
                if (data[b] == null)
                    return -1;
                if (data[a] < data[b])
                    return -1;
                if (data[a] > data[b])
                    return 1;
                return 0;
            });
        }
        index2index = [];
        for (var i = 0; i < order.length; i++)
            index2index[i] = order[i];

        // Create mapping from id->display
        display = [];
        id2index = {}
        $.each(ids, function(i, e) { id2index[e] = i; display.push(false); });

        // Generate a series for each shown column
        $(".display:checked").each(function() {
            var label = this.value;
            var data = numeric[label];
            series.push({ label: label,
                          xaxis: 2,
                          data: order.map(function(e, i) {
                                            return [i, data[order[i]]]; })
                        })
        });

        // Show the data
        plot = $.plot("#data", series, opts);
    }

    function plot_click(event, pos, item) {
        if (item == null)
            return;
        var raw_index = index2index[item.dataIndex];
        var id = columns["text"]["id"][raw_index];
        var action = shift_down ? "show_toggle" : "show_only";
        window.location = custom_scheme + ":" + action + "?id=" + id;
    }

    function plot_hover(event, pos, item) {
        if (item == null) {
            if (tooltip_shown) {
                // Hide tooltip
                $("#tooltip").css( { display: "none" } );
                tooltip_shown = false;
                // But leave table data alone
            }
        } else {
            var numeric = columns["numeric"];
            var text = columns["text"];

            // Show tooltip
            var raw_index = index2index[item.dataIndex];
            var label = item.series.label;
            var id = text["id"][raw_index];
            var name = "";
            var names = text["name"];
            if (names != null)
                name = " (" + names[raw_index] + ")";
            var value = numeric[item.series.label][raw_index];
            var where = { bottom: ($(window).height() - item.pageY) + 5 };
            if (item.pageX > ($(window).width() - item.pageX)) {
                where.left = '';
                where.right = ($(window).width() - item.pageX) + 5;
            } else {
                where.left = item.pageX + 5;
                where.right = '';
            }
            $("#tooltip").html(id + name + ": " + value).css(where).fadeIn(200);
            tooltip_shown = true;

            // Show data values in control table
            $(".value").each(function() {
                var column_name = this.title;
                var value = "";
                if (column_name in numeric)
                    value = numeric[column_name][raw_index];
                else if (column_name in text)
                    value = text[column_name][raw_index];
                this.textContent = value;
            });
        }
    }

    function shift_event(e) {
        shift_down = e.shiftKey;
    }

    function update_display(new_display) {
        // Update internal display state
        $.each(new_display, function () {
            display[id2index[this[0]]] = this[1];
        });
        // Highlight displayed models
        var num_series = plot.getData().length;
        plot.unhighlight();
        $.each(display, function(i, shown) {
            if (shown) {
                for (var n = 0; n < num_series; n++)
                    plot.highlight(n, i);
            }
        });
    }

    function init() {
        $("<div id='tooltip'></div>").css({
			position: "absolute",
			display: "none",
			border: "1px solid #fdd",
			padding: "2px",
			backgroundColor: "#fee",
			opacity: 0.80
		}).appendTo("body");
        $("#data").bind("plotclick", plot_click)
                  .bind("plothover", plot_hover)
                  .mousedown(shift_event);
    }

    return {
        update_columns: update_columns,
        update_display: update_display,
        init: init
    }
}();

$(document).ready(vdxchart.init);
