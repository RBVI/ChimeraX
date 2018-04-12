// vim: set expandtab shiftwidth=4 softtabstop=4:

var tgttable = function() {
    var custom_scheme = "tgttable";
    var targets_column_info = [
        "Name",
        "Info",
        "Atom",
        "Cartoon",
        "Surface",
    ];
    var components_column_info = [
        "Type",
        "Specifier",
        "Atom",
        "Cartoon",
        "Surface",
    ];
    var show_builtins = false;

    function update_targets(targets) {
        // Clean up previous incarnation and save some state
        $("#targets_table").trigger("destroy");
        $("#targets").empty();

        // Create table headers
        var thead = $("<thead/>");
        var row = $("<tr/>");
        var msg = "";
        if (show_builtins)
            msg = " built-in and user-defined names"
        else
            msg = " user-defined names"
        row.append($("<th/>", { colspan: targets_column_info.length })
                      .text(targets.length + msg));
        thead.append(row);
        row = $("<tr/>");
        for (var i = 0; i < targets_column_info.length; i++)
            row.append($("<th/>").text(targets_column_info[i]));
        thead.append(row);
        // Create table body
        var tbody = $("<tbody/>");
        $.each(targets, function(i, tgt) {
            var name = tgt["name"];
            var info = tgt["info"];
            var row = $("<tr/>", { class: "target_row" });
            row.append($("<td/>").append($("<a/>",
                                           { "href": _sel_url(name) })
                                           .text(name)));
            row.append($("<td/>").text(info));
            // add checkbox/color-selector pairs
            row.append(_add_dc("show", "hide", "abp", name, "atom"))
            row.append(_add_dc("cartoon", "cartoon hide", "c", name, "rib"))
            row.append(_add_dc("surface", "surface hide", "s", name, "surf"))
            tbody.append(row);
        });
        $("#targets").append($("<table/>", { id: "targets_table" })
                                            .append(thead, tbody));
        $("img.show, img.hide").click(cb_button_click);
        $("input.color").change(cb_color_input);

        // Re-setup jQuery handlers
        $("#targets_table").tablesorter({
            theme: 'blue',
            widgets: [ "resizable" ],
        });
    }

    function update_components(components) {
        // Clean up previous incarnation and save some state
        $("#components_table").trigger("destroy");
        $("#components").empty();

        // Create table headers
        var thead = $("<thead/>");
        var row = $("<tr/>");
        row.append($("<th/>", { colspan: components_column_info.length })
                      .text(components.length + " model components."));
        thead.append(row);
        row = $("<tr/>");
        for (var i = 0; i < components_column_info.length; i++) {
            row.append($("<th/>").text(components_column_info[i]));
            thead.append(row);
        }
        // Create table body
        var tbody = $("<tbody/>");
        $.each(components, function(i, tgt) {
            var type = tgt["type"];
            var atomspec = tgt["atomspec"];
            var row = $("<tr/>", { class: "component_row" });
            row.append($("<td/>").text(type));
            row.append($("<td/>").append($("<a/>",
                                           { "href": _sel_url(atomspec) })
                                           .text(atomspec)));
            // add checkbox/color-selector pairs
            row.append(_add_dc("show", "hide", "abp", atomspec))
            row.append(_add_dc("cartoon", "cartoon hide", "c", atomspec))
            row.append(_add_dc("surface", "surface hide", "s", atomspec))
            tbody.append(row);
        });
        $("#components").append($("<table/>", { id: "components_table" })
                                                .append(thead, tbody));
        $("img.show, img.hide").click(cb_button_click);
        $("input.color").change(cb_color_input);

        // Re-setup jQuery handlers
        $("#components_table").tablesorter({
            theme: 'blue',
            widgets: [ "resizable" ],
        });
    }

    // jQuery does not like '.' in id names even though JS does not care
    function _row_id(id) {
        return "row_" + id.replace('.', '_', 'g');
    }

    function _sel_url(name) {
        var path = "select";
        var selector = "selector=" + encodeURIComponent(name);
        var url = custom_scheme + ':' + path + '?' + selector;
        return url;
    }

    function _add_dc(show, hide, tgt, name, type) {
        var show_icon = "lib/" + type + "show.png";
        var hide_icon = "lib/" + type + "hide.png";
        return $("<td/>", { name:name })
                    .append($("<img/>", { class: "show", action: show,
                                          src: show_icon }))
                    .append($("<span/>", { class: "spacer" }))
                    .append($("<img/>", { class: "hide", action: hide,
                                          src: hide_icon }))
                    .append($("<span/>", { class: "spacer" }))
                    .append($("<input/>", { class: "color", type: "color",
                                            value: "#ffcf00", target: tgt }));
    }

    function cb_button_click(event) {
        var path = "show_hide";
        var img = $(event.target);
        var action = "action=" + img.attr("action");
        var td = img.parent();
        var selector = "selector=" + encodeURIComponent(td.attr("name"));
        var url = custom_scheme + ':' + path + '?' + action + '&' + selector;
        // console.log("image click " + url);
        window.location = url;
    }

    function cb_color_input(event) {
        var path = "color";
        var cp = $(event.target);
        var target = "target=" + cp.attr("target");
        var color = "color=" + encodeURIComponent(cp.val());
        var td = cp.parent();
        var selector = "selector=" + encodeURIComponent(td.attr("name"));
        var url = custom_scheme + ':' + path + '?'
                  + target + '&' + selector + '&' + color;
        // console.log("color input " + url);
        window.location = url;
    }

    function init() {
        $("#builtin_checkbox").change(function(event) {
            window.location = custom_scheme + ":builtin?show="
                              + $(event.target).is(":checked");
        });
        $("#nonmatching_checkbox").change(function(event) {
            window.location = custom_scheme + ":nonmatching?hide="
                              + $(event.target).is(":checked");
        });
        $("#targets_table").tablesorter();
        $("#components_table").tablesorter();
    }

    function get_state() {
        // Not much to return yet
        return {
            name:"tgttable",
        };
    }

    function set_state(state) {
        // Nothing to do yet
        return;
    }

    return {
        custom_scheme: custom_scheme,
        update_targets: update_targets,
        update_components: update_components,
        get_state: get_state,
        set_state: set_state,
        init: init
    }
}();

$(document).ready(tgttable.init);
