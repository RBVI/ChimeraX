// vim: set expandtab shiftwidth=4 softtabstop=4:

var active = function() {
    var custom_scheme = "active";
    var column_info = [
        "Tool Name",
        "Action",
    ];

    function update_tools(tools) {
        // Clean up previous incarnation and save some state
        $("#tools_table").trigger("destroy");
        $("#active_tools").empty();

        // Create table headers
        var thead = $("<thead/>");
        var row = $("<tr/>");
        row = $("<tr/>");
        for (var i = 0; i < column_info.length; i++)
            row.append($("<th/>").text(column_info[i]));
        thead.append(row);
        // Create table body
        var tbody = $("<tbody/>");
        $.each(tools, function(i, tool) {
            var name = tool["name"];
            var id = tool["id"];
            var killable = tool["killable"]
            var row = $("<tr/>", { class: "tool_row" });
            row.append($("<td/>").text(name));
            row.append(_add_dc("show", "hide", "close", id, killable))
            tbody.append(row);
        });
        $("#active_tools").append($("<table/>", { id: "tools_table" })
                                            .append(thead, tbody));
        $("img.show, img.hide, img.close").click(cb_button_click);

        // Re-setup jQuery handlers
        $("#tools_table").tablesorter({
            theme: 'blue',
            widgets: [ "resizable" ],
            sortReset: true,
            headers: { 1: { sorter: false } }
        });
    }

    function _add_dc(show, hide, close, id, killable) {
        var td = $("<td/>", { name: id })
                    .append($("<img/>", { class: "show", action: show,
                                          src: "lib/show.svg" }))
                    .append($("<span/>", { class: "spacer" }))
                    .append($("<img/>", { class: "hide", action: hide,
                                          src: "lib/hide.svg" }))
        if (killable) {
            td.append($("<span/>", { class: "spacer" }))
              .append($("<img/>", { class: "close", action: close,
                                  src: "lib/close.svg" }))
        }
        return td;
    }

    function cb_button_click(event) {
        var path = "show_hide_close";
        var img = $(event.target);
        var action = "action=" + img.attr("action");
        var td = img.parent();
        var tool = "tool=" + encodeURIComponent(td.attr("name"));
        var url = custom_scheme + ':' + path + '?' + action + '&' + tool;
        // console.log("image click " + url);
        window.location = url;
    }

    function init() {
        $("#tools_table").tablesorter();
    }

    return {
        "custom_scheme": custom_scheme,
        "update_tools": update_tools,
        "init": init
    }
}();

$(document).ready(active.init);
