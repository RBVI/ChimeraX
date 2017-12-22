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
    var plot = null;

    function make_button(btype, name, value, checked) {
        return $("<label/>").append(
                    $("<input/>", { type: btype,
                                    name: name,
                                    value: value,
                                    class: name,
                                    checked: checked }).click(update_plot));
    }

    function update_columns(new_columns) {
        columns = new_columns;
        display = Array(columns["text"]["id"].length).fill(false);
        histograms = {};
        // Save sort and shown columns
        var sort_column = $(".sort:checked").attr("value");
        var show_columns = []
        $(".display:checked").map(function() {
            if (this.value in columns["numeric"])
                show_columns.push(this.value);
        });
        if (show_columns.length == 0)
            show_columns.push(Object.keys(columns["numeric"])[0]);
        var hist_columns = [];
        $(".histogram:checked").map(function() {
            if (this.value in columns["numeric"])
                hist_columns.push(this.value);
        });

        // Clear out the table and fill in with text
        // then numeric column names
        $("#column_table").empty()
            .append($("<col/>"), $("<col/>"), $("<col/>"),
                    $("<col/>"), $("<col/>"))
            .append($("<tr/>").append(
                $("<th>/>").text("Sort").css("text-align", "center"),
                $("<th>/>").text("Graph").css("text-align", "center"),
                $("<th>/>").text("Hist").css("text-align", "center"),
                $("<th>/>").text("Column"),
                $("<th>/>").text("Value")));
            $.each(columns["text"], function(r, v) {
            $("#column_table").append($("<tr/>").append(
                                        $("<td/>"),
                                        $("<td/>"),
                                        $("<td/>"),
                                        $("<td/>").text(r),
                                        $("<td/>").addClass("value")
                                                  .prop("title", r)))
            });
        $.each(columns["numeric"], function(r, v) {
            var sort_btn = make_button("radio", "sort", r, r == sort_column);
            var show_btn = make_button("checkbox", "graph", r,
                                       show_columns.includes(r));
            var hist_btn = make_button("checkbox", "histogram", r,
                                       hist_columns.includes(r));
            $("#column_table").append($("<tr/>").append(
                                        $("<td/>").append(sort_btn)
                                                .css("text-align", "center"),
                                        $("<td/>").append(show_btn)
                                                .css("text-align", "center"),
                                        $("<td/>").append(hist_btn)
                                                .css("text-align", "center"),
                                        $("<td/>").text(r),
                                        $("<td/>").addClass("value")
                                                  .prop("title", r)))
            });

        update_plot();
    }

    function update_plot() {
        // Get order of compounds based on sort column
        var text = columns["text"];
        var numeric = columns["numeric"];
        var ids = text["id"];
        index2index = ids.map(function(e, i) { return i; });
        var sort_column = $(".sort:checked").attr("value");
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
        $(".graph:checked").each(function(i) {
            var label = this.value;
            var data = numeric[label];
            colormap[label] = i;
            next_color = i + 1;
            series.push({
                color: i,
                points: { show: true },
                lines: { show: true },
                label: label,
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
        });
        var num_bins = parseInt($("#histbins").prop("value"));
        $(".histogram:checked").each(function() {
            var label = this.value;
            var data = numeric[label];
            var color = colormap[label];
            if (color == null) {
                color = next_color;
                next_color += 1;
            }
            var hist = histograms[label];
            if (!hist || hist.length != num_bins) {
                hist = make_bins(data, num_bins);
                histograms[label] = hist;
            }
            series.push({
                name: label,
                color: color,
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
        if (item == null)
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

    function mousedown(e) {
        action = e.ctrlKey ? "show_toggle" : "show_only";
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
                  .mousedown(mousedown);
        $("#histbins").click(update_plot);
    }

    return {
        update_columns: update_columns,
        update_display: update_display,
        init: init
    }
}();

$(document).ready(vdxchart.init);
