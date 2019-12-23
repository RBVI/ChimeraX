// vim: set expandtab shiftwidth=4 softtabstop=4:

var vdxchart = function() {
    var columns = { numeric: {}, text:{} };
    var display = [];
    var histograms = {};
    var id2index = {};              // structure id -> plot data index
    var index2index = {};           // plot data index -> raw data index
    var tooltip_shown = false;
    var action = false;
    var custom_scheme = "vdxchart";
    var sweep_selected = false;
    var plot = null;

    var sort_column;
    var show_columns;
    var hist_columns;

    function update_columns(new_columns) {
        columns = new_columns;
        display = Array(columns["text"]["id"].length).fill(false);
        histograms = {};
        // Save sort and shown columns
        sort_column = $(".sort:checked").attr("value");
        show_columns = []
        $(".graph:checked").map(function() {
            if (this.value in columns["numeric"])
                show_columns.push(this.value);
        });
        if (show_columns.length == 0)
            show_columns.push(Object.keys(columns["numeric"])[0]);
        hist_columns = [];
        $(".histogram:checked").map(function() {
            if (this.value in columns["numeric"])
                hist_columns.push(this.value);
        });

        // Clear out the table and fill in with text
        // then numeric column names
        var rows = [];
        $.each(columns["text"], function(r, v) {
            rows.push({"column_name": r, "type": "text"});
        });
        $.each(columns["numeric"], function(r, v) {
            rows.push({"column_name": r, "type": "numeric"});
        });
        $("#column_table").bootgrid("clear")
                          .bootgrid("append", rows);
    }

    function update_plot() {
        // Get order of compounds based on sort column
        var text = columns["text"];
        var numeric = columns["numeric"];
        var ids = text["id"];
        if (ids == null)
            return;
        index2index = ids.map(function(e, i) { return i; });
        sort_column = $(".sort:checked").attr("value");
        if (sort_column != null) {
            var data = numeric[sort_column];
            index2index.sort(function(a, b) {
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
        plot.getOptions().xaxes[2].max = ids.length;

        // Create mapping from id->display
        id2index = {}
        $.each(index2index, function(i, e) {
            // i = plot index
            // e = raw index
            id2index[ids[e]] = i;
        });

        // Generate a series for each shown column
        var colormap = {};
        var next_color = 0;
        var series = [];
        var num_bins = parseInt($("#histbins").prop("value"));
        $(".graph:checked, .histogram:checked").each(function(i) {
            var label = this.value;
            var data = numeric[label];
            var color = i;
            if (colormap[label] != null) {
                color = colormap[label];
                legend = null;
            } else {
                colormap[label] = i;
                next_color = i + 1;
                legend = label;
            }
            if ($(this).hasClass("graph")) {
                series.push({
                    color: color,
                    points: { show: true },
                    lines: { show: true },
                    label: legend,
                    xaxis: 2,
                    has_data:
                        index2index.map(function(e, i) {
                            return data[index2index[i]] != null;
                        }),
                    data:
                        index2index.map(function(e, i) {
                            return [i, data[index2index[i]]];
                        })
                })
            }
            if ($(this).hasClass("histogram")) {
                var hist = histograms[label];
                if (!hist || hist.length != num_bins) {
                    hist = make_bins(data, num_bins);
                    histograms[label] = hist;
                }
                series.push({
                    name: label,
                    color: color,
                    label: legend,
                    bars: {
                        show: true,
                        align: "center",
                        barWidth: hist.width,
                        lineWidth: 1,
                        fill: 0.2,
                        horizontal: true,
                    },
                    data: hist.data,
                    xaxis: 3
                });
            }
        });

        // Show the data
        plot.setData(series);
        plot.setupGrid();
        plot.draw();
    }

    function make_bins(data, num_bins) {
        var max = Math.max.apply(Math, data);
        var min = Math.min.apply(Math, data);
        var range = max - min;
        var step = range / num_bins;
        var counts = []
        for (var i = 0; i < num_bins; i++)
            counts.push(0);
        $.each(data, function() {
            var n = Math.min(num_bins - 1, Math.trunc((this - min) / step));
            counts[n] += 1;
        })
        var ranges = [];
        for (var i = 0; i < num_bins; i++)
            ranges.push(min + (i + 0.5) * step);
        return {
            width: step,
            data:
                ranges.map(function(e, i) {
                    return [counts[i], e];
                })
        }
    }

    function plot_click(event, pos, item) {
        if (sweep_selected || item == null)
            return;
        else if (item.series.bars.show) {
            var label = item.series.name;
            var hist = histograms[label];
            var first_bar = item.dataIndex == 0
            var last_bar = item.dataIndex == hist.data.length - 1
            var half_width = hist.width / 2;
            var center = hist.data[item.dataIndex][1];
            var low = center - half_width;
            var high = center + half_width;
            var ids = columns["text"]["id"];
            var col = columns["numeric"][label];
            var selected = $.map(col, function(v, i) {
                if (v == null)
                    return null;
                else if (first_bar && v < high)
                    return ids[i];
                else if (last_bar && v >= low)
                    return ids[i];
                else if (v >= low && v < high)
                    return ids[i];
                else
                    return null;
            })
            if (selected.length > 0)
                window.location = custom_scheme + ":" + action + "?id=" +
                                  selected.join(",");
        } else {
            var raw_index = index2index[item.dataIndex];
            var id = columns["text"]["id"][raw_index];
            window.location = custom_scheme + ":" + action + "?id=" + id;
        }
    }

    function plot_hover(event, pos, item) {
        if (item == null || item.series.bars.show) {
            // Hovering over nothing or histogram bar
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
            $(".column_value").each(function(index) {
                var column_name = $(this).attr("name");
                var container = numeric[column_name];
                if (container == null)
                    container = text[column_name];
                var value = container[raw_index];
                $(this).text(value).prop("title", value);
            });
        }
    }

    function plot_selected(event, ranges) {
        sweep_selected = true;
        var start = Math.ceil(ranges.x2axis.from - 0.2);
        var end = Math.trunc(ranges.x2axis.to + 0.2);
        var ids = [];
        for (var i = start; i <= end; i++) {
            var raw_index = index2index[i];
            var id = columns["text"]["id"][raw_index];
            ids.push(id);
        }
        if (ids.length > 0)
            window.location = custom_scheme + ":" + action + "?id=" +
                              ids.join(",");
    }

    function mousedown(e) {
        action = e.ctrlKey ? "show_toggle" : "show_only";
        sweep_selected = false;
    }

    function update_display(new_display) {
        // Update internal display state
        $.each(new_display, function () {
            display[id2index[this[0]]] = this[1];
        });
        if (plot)
            redraw_highlights(plot);
    }

    function redraw_highlights(plot) {
        // Highlight displayed models
        var series = plot.getData();
        var num_series = series.length;
        plot.unhighlight();
        $.each(display, function(i, shown) {
            if (shown) {
                for (var n = 0; n < num_series; n++)
                    if (!series[n].bars.show && series[n].has_data[i])
                        plot.highlight(n, i);
            }
        });
    }

    var bootgrid_options = {
        navigation: 0,
        selection: false,
        rowSelect: false,
        multiSelect: false,
        keepSelection: true,
        caseSensitive: false,
        rowCount: -1,
        formatters: {
            "sort": function(column, row) {
                if (row.type != "numeric")
                    return "";
                var b = '<input type="radio"' +
                        ' value="' + row.column_name + '"' +
                        ' class="sort" name="sort"';
                if (row.column_name == sort_column)
                    b += ' checked';
                b += '>';
                return b;
            },
            "graph": function(column, row) {
                if (row.type != "numeric")
                    return "";
                var b = '<input type="checkbox"' +
                        ' value="' + row.column_name + '"' +
                        ' class="graph"';
                if (show_columns.includes(row.column_name))
                    b += ' checked';
                b += '>';
                return b;
            },
            "hist": function(column, row) {
                if (row.type != "numeric")
                    return "";
                var b = '<input type="checkbox"' +
                        ' value="' + row.column_name + '"' +
                        ' class="histogram"';
                if (hist_columns.includes(row.column_name))
                    b += ' checked';
                b += '>';
                return b;
            },
            "value": function(column, row) {
                return '<span class="column_value"' +
                       ' name="' + row.column_name + '"></span>';
            },
        }
    };

    function init() {
        var opts = {
            grid: {
                autoHighlight: false,
                hoverable: true,
                clickable: true
            },
            xaxes: [
                { },
                { show: false },
                { min: 0, max: 1 }
            ],
            selection: {
                mode: "x"
            },
            colors: colorbrewer.Set2[8],
            hooks: {
                draw: [ redraw_highlights ]
            }
        }
        plot = $.plot("#data", [], opts);
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
                  .bind("plotselected", plot_selected)
                  .mousedown(mousedown);
        $("#histbins").click(update_plot);
        function set_icons() {
            function set_column(cid, icon) {
                var col = $('[data-column-id="' + cid + '"]');
                col.find(".text").addClass("fa " + icon);
            }
            set_column("column_sort", "fa-sort");
            set_column("column_graph", "fa-line-chart");
            set_column("column_hist", "fa-bar-chart");
        }
        $("#column_table").on("initialized.rs.jquery.bootgrid", set_icons)
                          .on("loaded.rs.jquery.bootgrid", function(ev) {
                              $(".sort").click(update_plot);
                              $(".graph").click(update_plot);
                              $(".histogram").click(update_plot);
                              update_plot();
                          })
                          .bootgrid(bootgrid_options);
    }

    function get_state() {
        sort_column = $(".sort:checked").attr("value");
        show_columns = []
        $(".graph:checked").map(function() {
            show_columns.push(this.value);
        });
        hist_columns = [];
        $(".histogram:checked").map(function() {
            hist_columns.push(this.value);
        });
        return {
            name: "vdxchart",
            sort_column: sort_column,
            show_columns: show_columns,
            hist_columns: hist_columns
        };
    }

    function set_state(state) {
        show_columns = state.show_columns;
        $(".graph:checkbox").each(function(r, v) {
            var should_be_checked = $.inArray(v.value, show_columns) != -1;
            var is_checked = $(v).prop("checked");
            if (is_checked != should_be_checked)
                $(v).trigger("click");
        });
        hist_columns = state.hist_columns;
        $(".histogram:checkbox").each(function(r, v) {
            var should_be_checked = $.inArray(v.value, hist_columns) != -1;
            var is_checked = $(v).prop("checked");
            if (is_checked != should_be_checked)
                $(v).trigger("click");
        });
        var c = $(".sort:checked[value='" + state.sort_column + "']");
        c.trigger("click");
    }

    return {
        update_columns: update_columns,
        update_display: update_display,
        get_state: get_state,
        set_state: set_state,
        init: init
    }
}();

$(document).ready(vdxchart.init);
