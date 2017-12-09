// vim: set expandtab shiftwidth=4 softtabstop=4:

var vdxtable = {
    custom_scheme: "vdxtable",
    update_display: function(new_display) {
        for (var i = 0; i < new_display.length; i++) {
            var id = new_display[i][0];
            var checked = new_display[i][1];
            document.getElementById("cb_" + id).checked = checked;
        }
    }
}

function init() {

    $(document).ready(function() {
        $.tablesorter.addParser({
            id: 'id_col',
            is: function(s) {
                // return false so this parser is not auto detected
                return false;
            },
            format: function(s) {
                // Assume less than 100,000 models or submodels
                // and convert id to 10-character zero-padded string
                // which sorts correctly as text
                var pad = "00000";
                var padlen = pad.length;
                var parts = s.split("\n")[0].trim().split(".");
                var n = pad.substring(0, padlen - parts[0].length) + parts[0]
                      + pad.substring(0, padlen - parts[1].length) + parts[1];
                return n;

            },
            // set type, either numeric or text
            type: 'text'
        });
        $("#viewdockx_table").tablesorter({
            theme: 'blue',
            headers: {
                1: { sorter: 'id_col' }
            }
        });
    });

    $(".structure").click(function() {
        if ($(this).is(":checked")) {
            window.location = $(this).attr('href') + "&display=1";
        } else {
            window.location = $(this).attr('href') + "&display=0";
        }
    });

    $("#show_all_btn").click(function() {
        window.location = vdxtable.custom_scheme + ":check_all?show_all=true";
    });

    var data_array = [];
    var label_array = [];
    var property;

    $('#viewdockx_table tr td').on('click', function() {
        var $currentTable = $(this).closest('table');
        var index = $(this).index();
        $currentTable.find('td').removeClass('selected');
        $currentTable.find('tr').each(function() {
            $(this).find('td').eq(index).addClass('selected');
        });
        data_array = $(`#viewdockx_table td:nth-child(${index + 1}`).map(function() {
            return $(this).text();
        }).get();

        // ASSUMING NAME COLUMNS STAYS AS 2ND COLUMN. MAY NEED CHANGES LATER
        label_array = $(`#viewdockx_table td:nth-child(${2}`).map(function() {
            return $(this).text();
        }).get();

        property = $('#viewdockx_table th').eq($(this).index()).text();

    });

    $('#chart_btn').on('click', function() {
        window.location = vdxtable.custom_scheme + ":chart";
    });
    $('#plot_btn').on('click', function() {
        window.location = vdxtable.custom_scheme + ":plot";
    });
    $('#histogram_btn').on('click', function() {
        window.location = vdxtable.custom_scheme + ":histogram";
    });
}
