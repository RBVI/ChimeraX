// vim: set expandtab shiftwidth=4 softtabstop=4:

var columns = [{}, {}];
var tooltip_shown = false;
var shift_down = false;
var custom_scheme = "vdxchart";

function make_button(btype, name, value, text, checked) {
    return $("<label/>").text(text).append(
                $("<input/>", {"type":btype, "name":name,
                               "value":value, "class":name,
                               "checked":checked}).click(update_plot));
}

function update_columns(new_columns) {
    columns = new_columns;
    // Save sort and shown columns
    var sort_column = $(".sort:checked").attr("value");
    var show_columns = []
    $(".show:checked").map(function() {
        if (this.value in columns[0])
            show_columns.push(this.value);
    });
    if (show_columns.length == 0)
        show_columns.push(Object.keys(columns[0])[0])

    // Clear out the table and fill in with text
    // then numeric column names
    $("#column_table").empty();
    $.each(columns[1], function(r, v) {
        $("#column_table").append($("<tr/>").append(
                                    $("<td/>"),
                                    $("<td/>").text(r),
                                    $("<td/>").addClass("value")
                                              .prop("title", r)))
        });
    $.each(columns[0], function(r, v) {
        var sort_btn = make_button("radio", "sort", r, "sort",
                                   r == sort_column);
        var show_btn = make_button("checkbox", "show", r, "show",
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
            hoverable: true,
            clickable: true
        },
        xaxes: [
            { },
            { show: false }
        ],
    }

    // Get order of compounds based on sort column
    var text = columns[1];
    var numeric = columns[0];
    var ids = text["Id"]
    var order = ids.map(function(e, i) { return i; });
    var sort_column = $(".sort:checked").attr("value");
    if (sort_column != null) {
        var data = numeric[sort_column];
        order.sort(function(a, b) { return data[a] < data[b] ? -1 :
                                           data[a] > data[b] ? 1 : 0 });
    }

    // Generate a series for each shown column
    $(".show:checked").each(function() {
        var label = this.value;
        var data = numeric[label];
        series.push({ label: label,
                      xaxis: 2,
                      data: order.map(function(e, i) {
                                        return [i, data[order[i]]]; })
                    })
    });

    // Show the data
    $.plot("#data", series, opts);
}

function plot_click(event, pos, item) {
    if (item == null)
        return;
    var id = columns[1]["Id"][item.dataIndex];
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
        // Show tooltip
        var label = item.series.label;
        var id = columns[1]["Id"][item.dataIndex];
        var name = "";
        var names = columns[1]["Name"];
        if (names != null)
            name = " (" + names[item.dataIndex] + ")";
        var value = columns[0][item.series.label][item.dataIndex];
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
        var numeric = columns[0];
        var text = columns[1];
        $(".value").each(function() {
            var column_name = this.title;
            var value = "";
            if (column_name in numeric)
                value = numeric[column_name][item.dataIndex];
            else if (column_name in text)
                value = text[column_name][item.dataIndex];
            this.textContent = value;
        });
    }
}

function init() {
    $(document).ready(function() {
        $("<div id='tooltip'></div>").css({
			position: "absolute",
			display: "none",
			border: "1px solid #fdd",
			padding: "2px",
			"background-color": "#fee",
			opacity: 0.80
		}).appendTo("body");
        $("#data").bind("plotclick", plot_click);
        $("#data").bind("plothover", plot_hover);
        $("#data").mousedown(function(e) { shift_down = e.shiftKey; });
    });
}
