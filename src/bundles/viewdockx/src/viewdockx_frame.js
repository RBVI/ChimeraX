// vim: set expandtab shiftwidth=4 softtabstop=4:

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
                0: { sorter: 'id_col' }
            }
        });
    });

    $("#show_checkboxes").click(function() {
        if ($(this).is(":checked")) {
            $(".checkbox").show();
            $(".link").hide();
        } else {
            $(".checkbox").hide();
            $(".link").show();
        }
    });

    $(".struct").click(function() {
        if ($(this).is(":checked")) {
            window.location = $(this).attr('href') + "&display=1";
        } else {
            window.location = $(this).attr('href') + "&display=0";
        }
    });

    $("#check_all").click(function() {
        if ($(this).is(":checked")) {
            $(".struct").prop('checked', true);
            window.location = "viewdockx:check_all?show_all=true";
        } else {
            $(".struct").prop('checked', false);
            window.location = "viewdockx:check_all?show_all=false";
        }
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

    // var vdx_chart=null; 
    $('#graph_btn').on('click', function() {
        if (typeof vdx_chart != 'undefined') {
            vdx_chart.destroy();
        }
        var context = document.getElementById("viewdockx_chart").getContext('2d');
        vdx_chart = new Chart(context, {
            type: 'line',
            data: {
                labels: label_array, //x - axis
                datasets: [{
                    label: property,
                    data: data_array //y - axis
                }]
            }
        });
    });

    // $('#histogram_btn').on('click', function() {
    //     var data = d3.range(1000).map(d3.randomBates(10));

    //     var formatCount = d3.format(",.0f");

    //     var svg = d3.select("svg"),
    //         margin = { top: 10, right: 30, bottom: 30, left: 30 },
    //         width = +svg.attr("width") - margin.left - margin.right,
    //         height = +svg.attr("height") - margin.top - margin.bottom,
    //         g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    //     var x = d3.scaleLinear()
    //         .rangeRound([0, width]);

    //     var bins = d3.histogram()
    //         .domain(x.domain())
    //         .thresholds(x.ticks(20))
    //         (data);

    //     var y = d3.scaleLinear()
    //         .domain([0, d3.max(bins, function(d) { return d.length; })])
    //         .range([height, 0]);

    //     var bar = g.selectAll(".bar")
    //         .data(bins)
    //         .enter().append("g")
    //         .attr("class", "bar")
    //         .attr("transform", function(d) { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; });

    //     bar.append("rect")
    //         .attr("x", 1)
    //         .attr("width", x(bins[0].x1) - x(bins[0].x0) - 1)
    //         .attr("height", function(d) { return height - y(d.length); });

    //     bar.append("text")
    //         .attr("dy", ".75em")
    //         .attr("y", 6)
    //         .attr("x", (x(bins[0].x1) - x(bins[0].x0)) / 2)
    //         .attr("text-anchor", "middle")
    //         .text(function(d) { return formatCount(d.length); });

    //     g.append("g")
    //         .attr("class", "axis axis--x")
    //         .attr("transform", "translate(0," + height + ")")
    //         .call(d3.axisBottom(x));
    // });


//    $('#histogram_btn').on('click', function() {
//        for (var i = 0; i < data_array.length; i++) {
//            if(data_array[i] === "missing"){
//                data_array[i] = 0;
//            }
//        }
//
//        if(typeof(h_data) != 'undefined'){
//            //alert("hello"); test
//            d3.selectAll('g').remove()
//        }
//
//        h_data = data_array;
//
//        max = Math.max.apply(Math, h_data);
//        min = Math.min.apply(Math, h_data);
//
//        var formatCount = d3.format(",.0f");
//
//        var svg = d3.select("svg"),
//            margin = {top: 10, right: 30, bottom: 30, left: 30},
//            width = +svg.attr("width") - margin.left + 10 - margin.right,
//            height = +svg.attr("height") - margin.top - margin.bottom,
//            g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");
//
//        var x = d3.scaleLinear(h_data)
//            .domain([min, max]) // x axis domain, from min and max of array
//            .rangeRound([0, width]);
//
//        var bins = d3.histogram()
//            .domain(x.domain())
//            //.thresholds()
//            (h_data);
//
//        var y = d3.scaleLinear()
//            .domain([0, d3.max(bins, function(d) { return d.length; })])
//            .range([height, 0]);
//
//        var bar = g.selectAll(".bar")
//          .data(bins)
//          .enter().append("g")
//            .attr("class", "bar")
//            .attr("transform", function(d) { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; });
//
//        bar.append("rect")
//            .attr("x", 1)
//            .attr("width", x(bins[0].x1) - x(bins[0].x0) - 1)
//            .attr("height", function(d) { return height - y(d.length); });
//
//        bar.append("text")
//            .attr("dy", ".75em")
//            .attr("y", 6)
//            .attr("x", (x(bins[0].x1) - x(bins[0].x0)) / 2)
//            .attr("text-anchor", "middle")
//            .text(function(d) { return formatCount(d.length); });
//
//        g.append("g")
//            .attr("class", "axis axis--x")
//            .attr("transform", "translate(0," + height + ")")
//            .call(d3.axisBottom(x));
//    });








//    var margin = {
//        top: 30,
//        right: 20,
//        bottom: 30,
//        left: 50
//    };
//    var width = 600 - margin.left - margin.right;
//    var height = 270 - margin.top - margin.bottom;
//
//    var parseDate = d3.time.format("%d-%b-%y").parse;
//
//    var x = d3.time.scale().range([0, width]);
//    var y = d3.scale.linear().range([height, 0]);
//
//    var xAxis = d3.svg.axis().scale(x)
//        .orient("bottom").ticks(5);
//
//    var yAxis = d3.svg.axis().scale(y)
//        .orient("left").ticks(5);
//
//    var valueline = d3.svg.line()
//        .x(function (d) {
//          return x(d.date);
//        })
//        .y(function (d) {
//          return y(d.close);
//        });
//
//    var svg = d3.select("body")
//        .append("svg")
//        .attr("width", width + margin.left + margin.right)
//        .attr("height", height + margin.top + margin.bottom)
//        .append("g")
//        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
//
//    // Get the data
//    var data = [{
//        date: "1-May-12",
//        close: "58.13"
//    }, {
//        date: "30-Apr-12",
//        close: "53.98"
//    }, {
//        date: "27-Apr-12",
//        close: "67.00"
//    }, {
//        date: "26-Apr-12",
//        close: "89.70"
//    }, {
//        date: "25-Apr-12",
//        close: "99.00"
//    }];
//
//    data.forEach(function (d) {
//        d.date = parseDate(d.date);
//        d.close = +d.close;
//    });
//
//    // Scale the range of the data
//    x.domain(d3.extent(data, function (d) {
//        return d.date;
//        }));
//    y.domain([0, d3.max(data, function (d) {
//        return d.close;


//        })]);
//
//    svg.append("path") // Add the valueline path.
//    .attr("d", valueline(data));
//
//    svg.append("g") // Add the X Axis
//    .attr("class", "x axis")
//        .attr("transform", "translate(0," + height + ")")
//        .call(xAxis);
//
//    svg.append("g") // Add the Y Axis
//    .attr("class", "y axis")
//        .call(yAxis);


    

}

// chart.options.data.push({object}); // Add a new dataSeries

// https://canvasjs.com/docs/charts/basics-of-creating-html5-chart/updating-chart-options/
