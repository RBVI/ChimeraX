// vim: set expandtab shiftwidth=4 softtabstop=4:

columns = [{}, {}];
tooltip_shown = false;

function make_button(btype, name, value, text, checked) {
    var label = document.createElement("label");
    var btn = document.createElement("input");
    btn.type = btype;
    btn.name = name;
    btn.className = name;
    btn.value = value;
    btn.checked = checked;
    btn.onclick = update_plot;
    label.appendChild(btn);
    label.appendChild(document.createTextNode(text));
    return label;
}

function reload() {
    var table = document.getElementById("column_table");
    // Save sort and shown columns
    var sort_column = null;
    var sort_buttons = document.getElementsByClassName("sort");
    for (var i = 0; i < sort_buttons.length; i++) {
        var e = sort_buttons[i];
        if (e.checked)
            sort_column = e.value;
    }
    var show_columns = [];
    var show_buttons = document.getElementsByClassName("show");
    for (var i = 0; i < show_buttons.length; i++) {
        var e = show_buttons[i];
        if (e.checked && columns[0].includes(e.value))
            show_columns.push(e.value);
    }
    if (show_columns.length == 0)
        show_columns.push(Object.keys(columns[0])[0])

    // Clear out the table and fill in with text
    // then numeric column names
    while (table.hasChildNodes())
        table.removeChild(table.lastChild);
    for (r in columns[1]) {
        var tr = document.createElement("tr");
        table.appendChild(tr);
        var td0 = document.createElement("td");
        tr.appendChild(td0);
        var td1 = document.createElement("td");
        td1.textContent = r;
        tr.appendChild(td1);
        var td2 = document.createElement("td");
        td2.className = "value";
        td2.value = r;
        tr.appendChild(td2);
    }
    for (r in columns[0]) {
        var tr = document.createElement("tr");
        table.appendChild(tr);
        var td0 = document.createElement("td");
        td0.appendChild(make_button("radio", "sort", r, "sort",
                                    r == sort_column));
        td0.appendChild(make_button("checkbox", "show", r, "show",
                                    show_columns.includes(r)));
        tr.appendChild(td0);
        var td1 = document.createElement("td");
        td1.textContent = r;
        tr.appendChild(td1);
        var td2 = document.createElement("td");
        td2.className = "value";
        td2.value = r;
        tr.appendChild(td2);
    }
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
    var sort_column = null;
    var sort_buttons = document.getElementsByClassName("sort");
    for (var i = 0; i < sort_buttons.length; i++) {
        var e = sort_buttons[i];
        if (e.checked)
            sort_column = e.value;
    }
    if (sort_column != null) {
        var data = numeric[sort_column];
        order.sort(function(a, b) { return data[a] < data[b] ? -1 :
                                           data[a] > data[b] ? 1 : 0 });
    }

    // Generate a series for each shown column
    var show_buttons = document.getElementsByClassName("show");
    for (var i = 0; i < show_buttons.length; i++) {
        var e = show_buttons[i];
        if (!e.checked)
            continue;
        var label = e.value;
        var data = numeric[label];
        series.push({ label: label,
                      xaxis: 2,
                      data: order.map(function(e, i) {
                                        return [i, data[order[i]]]; })
                    })
    }

    // Show the data
    $.plot("#data", series, opts);
}

function plot_click(event, pos, item) {
    if (item == null)
        return;
    var id = columns[1]["Id"][item.dataIndex];
    var action = "show_only";
    window.location = "viewdockx:" + action + "?id=" + id;
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
        var value_cells = document.getElementsByClassName("value");
        for (var i = 0; i < value_cells.length; i++) {
            var e = value_cells[i];
            var data = numeric[e.value];
            if (data == null)
                data = text[e.value];
            if (data == null)
                value = "";
            else {
                value = data[item.dataIndex];
                if (value == null)
                    value = "";
            }
            e.textContent = value;
        }
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
    });
}
