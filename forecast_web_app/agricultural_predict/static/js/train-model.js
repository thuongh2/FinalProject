$("#data_name").change(function () {
    const selectedValue = JSON.parse($(this).val().replace(/'/g, '"'));
    console.log(selectedValue);
    callDrawPlot(selectedValue.data);
    handelStoreSession("data_url", selectedValue.data);
    handelStoreSession("agricutural_name", selectedValue.type);
    handelStoreSession("model_name", $("#model_name").find(":selected").text());
    return;
});

function handelStoreSession(key, value) {
    sessionStorage.setItem(key, value);
}

function plotAcf(data, type) {
    var trace = {
        x: Array.from({length: 10}, (x, i) => i),
        y: data,
        type: "bar",
        width: 0.05,
    };
    var data = [trace];
    var layout = {
        title: type,
    };
    Plotly.newPlot("my" + type, data, layout);
}

function checkStationary(pValue) {
    if (pValue < 0.05) {
        $("#pValue").text(`P value is ${pValue}`);
        $("#pValueDesc").text(`Data is stationary`);
        return;
    }
    $("#pValue").text(`P value is ${pValue}`);
    $("#pValueDesc").text(`Data is None stationary`);
}

function callDrawPlot(selectedValue) {
    d3.csv(selectedValue, function (err, rows) {
        function unpack(rows, key) {
            return rows.map(function (row) {
                if (key === "date") {
                    return new Date(row[key]);
                } else {
                    return parseFloat(row[key]);
                }
            });
        }

        Object.keys(rows[0]).forEach(function (row) {
            var data = new Array();
            if (row === "date") return;
            var trace = {
                type: "scatter",
                mode: "lines",
                name: row,
                x: unpack(rows, "date"),
                y: unpack(rows, row),
                line: {color: "#17BECF"},
            };

            console.log(trace);
            const nodeName = "myChart" + row;
            const node = document.createElement("div");
            node.id = nodeName;

            document.getElementById("myChart").appendChild(node);

            var layout = {
                title: "Biểu đồ giá " + row,
            };

            Plotly.newPlot(nodeName, [trace], layout);
        });
    });
}

function plotStationaryData1(data) {
    plot_data = new Array();

    data.forEach((value, index) => {
        xDim = new Array();
        for (i in value.x)
            xDim.push(new Date(value.x[i].$date));
        yDim = value.y;
        mode = value.mode;

        trace = {x: xDim, y: yDim, type: "scatter", mode: "lines", name: value.name};
        console.log(trace);

        plot_data.push(trace)
    });

    var layout = {
        title: "Biểu đồ giá "
    };

    Plotly.newPlot('myChartTrainModel', plot_data, layout);
}


function plotStationaryData(data) {

    data.forEach((value, index) => {
        console.log(value.x);
        xDim = new Array();
        for (i in value.x) xDim.push(new Date(value.x[i]));
        yDim = value.y;
        mode = value.mode;

        trace = {x: xDim, y: yDim, type: "scatter", mode: "lines"};
        console.log(trace);
        const nodeName = "myDivStationary" + index;
        const node = document.createElement("div");
        node.id = nodeName;
        console.log(node);
        document.getElementById("myDivStationary").appendChild(node);

        var layout = {
            title: "Biểu đồ giá " + value.name
        };
        Plotly.newPlot(nodeName, [trace], layout);
    });
}


function loadData(state) {
    if (state) {
        $("#loadStationary").removeClass("d-none").addClass("d-flex");
        $("#plotStationaryData").addClass("d-none");
        $("#stationary").addClass("d-none");
    } else {
        $("#loadStationary").addClass("d-none").removeClass("d-flex");
        $("#plotStationaryData").removeClass("d-none");
        $("#stationary").removeClass("d-none");
    }
}

async function applyDiffSeasonal(value) {
    await loadData(true);
    var seasonal = parseInt($("#applyLog").find(":selected").text()) || 1;

    const url = "http://localhost:5000/stationary-train-model";
    const modelName = sessionStorage.getItem("model_name");
    const isStationary = "True";
    const diffType = value;
    const lag = seasonal;
    const modelData = sessionStorage.getItem("data_url");

    const params = {
        model_name: modelName,
        is_stationary: isStationary,
        diff_type: diffType,
        lag: lag,
        model_data: modelData,
    };
    handelStoreSession("stationary_data", JSON.stringify(params));

    const queryString = $.param(params);
    const fullUrl = url + "?" + queryString;

    await $.ajax({
        url: fullUrl,
        type: "GET",
        contentType: "application/json",
        success: function (response) {
            checkStationary(response.p_values);

            plotAcf(response.acf, "ACF");
            plotAcf(response.df_pacf, "PACF");
            loadData(false);
            plotStationaryData(response.plot_data);
        },
        error: function (xhr, status, error) {
            // Handle the error here
            console.log(error);
        },
    });
}

async function trainModel() {

    const model_name = sessionStorage.getItem("model_name");
    if (!model_name) {
        alertify.error("Vui lòng chọn tên model");
    }
    const model_data = sessionStorage.getItem("data_url");
    if (!model_data) {
        alertify.error("Vui lòng chọn dữ liệu");
    }
    const username = $("#username").text();
    const agricutural_name = sessionStorage.getItem("agricutural_name");

    var formEl = document.getElementById("trainModelForm");

    var formData = new FormData(formEl);

    var argument = {};

    // Loop through each form element
    for (var pair of formData.entries()) {
        var name = pair[0];
        var value = pair[1];

        if (value === undefined || value === "") {
            alertify.error("Vui lòng điền " + name);
            return;
        }
        value = parseInt(value)
        if (name == 'size')
            value = value / 100;
        argument[name] = value;
    }

    data = {
        model_name: model_name,
        model_data: model_data,
        username: username,
        agricutural_name: agricutural_name,
        argument: argument,
    };
    console.log(data);

    $('#modelTrainingInProcess').modal('show')
    await $.ajax({
        url: "http://localhost:5000/train-model-data",
        method: "POST", // First change type to method here
        contentType: "application/json; charset=utf-8",
        data: JSON.stringify(data),
        success: function (response) {

            const data = JSON.parse(response)
            console.log(data)

            handelStoreSession("model_submit_detail", JSON.stringify(data));

            if (data.status == 'DONE') {
                $("#model_detail_name").text(data.model_name);
                $("#model_detail_mape").text(data.score['mape'] | 0);
                $("#model_detail_rmse").text(data.score['rmse'] | 0);

                // show chart data như trang detail
                plotStationaryData1(data.plot_data);
                $('#detail-tab').tab('show')
            } else {
                alertify.error(data.error);
            }
        },
        error: function (error) {
            alert("error" + error);

            alertify.error(error);
        },
    });
    $('#modelTrainingInProcess').modal('hide')


}

async function submitModel() {

  data = sessionStorage.getItem("model_submit_detail")
  if(!data){ 
    alertify.error("Vui lòng train model trước khi submit")
    return
  }

  await $.ajax({
      url: "http://localhost:5000/submit-train-model-data",
      method: "POST", // First change type to method here
      contentType: "application/json; charset=utf-8",
      data: JSON.stringify(data),
      success: function (response) {
          const data = JSON.parse(response)
          console.log(data)

          if(data.$oid){
            window.location.href = '/detail-model?model_id=' + data.$oid;
            return
          }
          alertify.error("Submit model không thành công");
      },
      error: function (error) {
          alert("error" + error);

          alertify.error("Submit model không thành công");
      },
  });
  $('#modelTrainingInProcess').modal('hide')


}


$(document).ready(function () {
    $("input[type=radio][name=flexRadioDefault]").change(function () {
        var selectedValue = $(this).val();
        if (selectedValue === "diff") {
            $("#applyLog").removeClass("d-none");
        } else {
            $("#applyLog").addClass("d-none");
        }
        applyDiffSeasonal(selectedValue);
    });

    $("#applyLog").on("change", function () {
        var selectedValue = $(
            "input[type=radio][name=flexRadioDefault]:checked"
        ).val();
        applyDiffSeasonal(selectedValue);
    });

    $("#trainModelForm").submit(function (event) {
        event.preventDefault();
        alertify.set("notifier", "position", "top-right");

        trainModel(event);
    });


    $("#submitModel").on("click", function (event) {
      event.preventDefault();
      console.log("submit model");
      submitModel();
  });

    var el = document.getElementById("curr");
    var r = document.getElementById("myRange");
    el.innerText = r.valueAsNumber + "%";
    r.addEventListener("change", () => {
        el.innerText = r.valueAsNumber + "%";
    });
});
