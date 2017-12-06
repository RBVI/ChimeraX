// vim: set expandtab shiftwidth=4 softtabstop=4:

columns = [{}, {}]

function reload() {
    var numeric = columns[0];
    var keys = Object.keys(numeric);
    if (keys.length < 2)
        alert("Not enough numeric keys " + keys);
    else {
        var a0 = numeric[keys[0]];
        var a1 = numeric[keys[1]];
        $.plot("#data", [
            { label: keys[0] + "-" + keys[1],
              data: a0.map(function(e, i) { return [e, a1[i]]; })
            },
        ]);
    }
}

function init() {
    $(document).ready(function() {
        /*
        var d1 = [];
        for (var i = 0; i < 14; i += 0.5) {
                d1.push([i, Math.sin(i)]);
        }
        var d2 = [[0, 3], [4, 8], [8, 5], [9, 13]];
        // A null signifies separate line segments
        var d3 = [[0, 12], [7, 12], null, [7, 2.5], [12, 2.5]];
        $.plot("#data", [ d1, d2, d3 ]);
        */
        var f1 = [];
        var f2 = [];
        for (var i = 0; i < 14; i += 0.5) {
            f1.push(i);
            f2.push(Math.sin(i));
        }
        columns[0]["f1"] = f1;
        columns[0]["f2"] = f2;
        reload();
    });
}
