const URL_SERVER = "http://localhost:5000";
var interval = null;
$("#data_name").change(function () {
    const selectedValue = JSON.parse($(this).val().replace(/'/g, '"'));
    console.log(selectedValue);
    callDrawPlot(selectedValue.data);
    handelStoreSession("data_url", selectedValue.data);
    handelStoreSession("agricultural_name", selectedValue.type);
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

        const checkboxTemplate = `
            <div class="form-check">
            <input class="exogenous-checkbox" type="checkbox" name="exogenous" value="{{value}}" id="checkbox-{{value}}">
            <label class="form-check-label" for="checkbox-{{value}}">
                {{label}}
            </label>
            </div>
            `;
        const container = $("#checkboxContainer");
        container.empty();

        Object.keys(rows[0]).forEach(function (name) {
            if (name === "date") return;
            const html = checkboxTemplate
                .replace("{{value}}", name)
                .replace("{{label}}", name);
            container.append(html);
        });
    });
}

function plotChartData(data) {
    var plot_data = new Array();

    data.forEach((value, index) => {
        xDim = new Array();
        for (i in value.x) xDim.push(new Date(value.x[i].$date));
        yDim = value.y;
        mode = value.mode;

        trace = {
            x: xDim,
            y: yDim,
            type: "scatter",
            mode: "lines",
            name: value.name,
        };
        console.log(trace);

        plot_data.push(trace);
    });

    var layout = {
        title: "Biểu đồ giá ",
    };

    Plotly.newPlot("myChartTrainModel", plot_data, layout);
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
            title: "Biểu đồ giá " + value.name,
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

    const url = URL_SERVER + "/stationary-train-model";
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
    handelStoreSession("diff_type", diffType);

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
            alertify.error("Lỗi!");
        },
    });
}

async function trainModel() {
    const model_name = sessionStorage.getItem("model_name");
    await sessionStorage.removeItem("dags_run_id");
    if (!model_name) {
        alertify.error("Vui lòng chọn tên model");
    }
    const model_data = sessionStorage.getItem("data_url");
    if (!model_data) {
        alertify.error("Vui lòng chọn dữ liệu");
    }
    const username = $("#username").text();
    const agricultural_name = sessionStorage.getItem("agricultural_name");

    var formEl = document.getElementById("trainModelForm");

    var formData = new FormData(formEl);

    var argument = {};

    // Loop through each form element
    for (var pair of formData.entries()) {
        var name = pair[0];
        var value = pair[1];

        if (value === undefined || value === "") {
            await alertify.error("Vui lòng điền " + name);
            return;
        }
        if (name == "exogenous") {
            var names = [];
            $("#checkboxContainer input:checked").each(function () {
                names.push(this.value);
            });
            argument[name] = names;
        } else {
            value = parseInt(value);
            if (name == "size") value = value / 100;
            argument[name] = value;
        }
    }
    diff_type = sessionStorage.getItem("diff_type");
    argument["stationary_type"] = diff_type;
    model_id = $("#modelId").text().trim();

    data = {
        model_name: model_name,
        model_data: model_data,
        username: username,
        agricultural_name: agricultural_name,
        argument: argument,
        model_id: model_id,
    };
    console.log(data);

    await $("#modelTrainingInProcess").modal("show");
    await $.ajax({
        url: URL_SERVER + "/train-model-data",
        method: "POST", // First change type to method here
        contentType: "application/json; charset=utf-8",
        data: JSON.stringify(data),
        success: function (response) {
            const data = JSON.parse(response);
            console.log(data);

            handelStoreSession("model_submit_detail", JSON.stringify(data));
            handelStoreSession("dags_run_id", data.dag_run_id);

            if (data.status === "DONE") {
                $("#model_detail_name").text(data.model_name);
                $("#model_detail_mape").text(data.score["mape"] | 0);
                $("#model_detail_rmse").text(data.score["rmse"] | 0);

                // show chart data như trang detail
                // plotChartData(data.plot_data);
            } else {
                alertify.error(data.error);
            }
        },
        error: function (error) {
            alertify.error("Thất bại" + error);
        },
    });
    //
}

async function submitModel() {
    data = sessionStorage.getItem("model_submit_detail");
    if (!data) {
        alertify.error("Vui lòng train model trước khi submit");
        return;
    }

    await $.ajax({
        url: URL_SERVER + "/submit-train-model-data",
        method: "POST", // First change type to method here
        contentType: "application/json; charset=utf-8",
        data: JSON.stringify(data),
        success: function (response) {
            const data = JSON.parse(response);
            console.log(data);

            if (data) {
                window.location.href = "/detail-model?model_id=" + data;
                return;
            }
            alertify.error("Submit model không thành công");
        },
        error: function (error) {
            alert("error" + error);

            alertify.error("Submit model không thành công");
        },
    });
    $("#modelTrainingInProcess").modal("hide");
}

const pipelineTemplate = `
                                      <div class="card">
                                            <div class="card-body">
                                                <h6 class="card-title text-primary font-bold font-weight-normal">
                                                    <div class='{{status-display}} spinner-border text-secondary'
                                                        style="width: 15px; height: 15px;" role="status">
                                                        <span class="sr-only">Loading...</span>
                                                    </div>
                                                    {{value}}
                                                </h6>
                                                <div class="d-flex justify-content-between">
                                                    <span class="badge badge-{{status-class}} align-middle">{{status}}</span>
                                                    // <span class="badge badge-secondary align-middle">Logs</span>
                                                </div>
                                            </div>
                                        </div>

`;

const arrowTemplate = `
<div class="text-center">
<i class="bi bi-arrow-down text-center text-primary "
    style="font-weight: 600!important;"></i>
</div>
`;

const statusClass = new Map([
    ["success", "success"],
    ["failed", "danger"],
    ["running", "info"],
    ["queued", "warning"],
    ["undefined", "muted"],
]);

const statusMapping = new Map([
    ["success", "Thành công"],
    ["failed", "Thất bại"],
    ["running", "Đang xử lí"],
    ["queued", "Đang xử lí"],
    ["waiting", "Chờ xử lí"],
    ["undefined", "Chờ xử lí"],
]);

function compare(a, b) {
    if (a.priority_weight < b.priority_weight) {
        return 1;
    }
    if (a.priority_weight > b.priority_weight) {
        return -1;
    }
    return 0;
}

async function loadLogTrainModel() {
    var dag_run_id = sessionStorage.getItem("dags_run_id");
    if (dag_run_id === 'undefined' || dag_run_id === 'null' || dag_run_id === "") {
        console.error("Không tìm thấy dag id");
        return;
    }

    let username = "airflow";
    let password = "airflow";
    let auth = btoa(`${username}:${password}`);
    let model_id = $("#modelId").text().trim();

    var url = "http://localhost:5000/pipeline/{dag_run_id}" + "/" + dag_run_id;
    url = url.replaceAll("{dag_run_id}", model_id);
    var settings = {
        url: url,
        method: "GET",
        headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            Authorization: `Basic ${auth}`,
        },
    };

    var responseData = null;
    await $.ajax(settings).done(function (response) {
        response = JSON.parse(response);
        console.log(response);
        responseData = response;
        if (response.total_entries === 0) {
            console.log("Không tìm thấy task")
            return
        }

        const container = $("#pipeline-step");
        container.empty();
        var count_waiting_task = 0;

        response.task_instances.sort(compare);
        console.log(response.task_instances);

        response.task_instances.forEach((value, index) => {
            if (value.state === "success") {
                count_waiting_task++;
            }
            if (value.state === "failed") {
                clearInterval(interval);
                alertify.error("Model training thất bại");
                return;
            }
            const html = pipelineTemplate
                .replace("{{value}}", value.task_id)
                .replace("{{status}}", statusMapping.get(value.state || "waiting"))
                .replace(
                    "{{status-class}}",
                    statusClass.get(value.state) || "secondary"
                )
                .replace(
                    "{{status-display}}",
                    value.state === "running" || value.state === "queued" ? "" : "d-none"
                );
            container.append(html);
            if (index !== response.task_instances.length - 1) {
                container.append(arrowTemplate);
            }
        });
        console.log(count_waiting_task);
        if (count_waiting_task === response.total_entries) {
            clearInterval(interval);
            // call api get model detail
            getTranningModelDetail(model_id);
            $("#detail-tab").tab("show");
            $("#modelTrainingInProcess").modal("hide");

        }
    });
}

async function getTranningModelDetail(modelId) {
    await $.ajax({
        url: URL_SERVER + "/get_train_model_airflow" + "/" + modelId,
        method: "GET",
        contentType: "application/json; charset=utf-8",
        success: async function (response) {
            const data = JSON.parse(response);
            console.log(data);

            await handelStoreSession("model_submit_detail", JSON.stringify(data));
            await handelStoreSession("dags_run_id", data.dag_run_id);

            if (data.status === "SUCCESS") {
                $("#model_detail_name").text(data.model_name);
                $("#model_detail_mape").text(data.evaluate["mape"] | 0);
                $("#model_detail_rmse").text(data.evaluate["rmse"] | 0);

                // show chart data như trang detail
                await plotChartData(data.plot_data);
            } else {
                alertify.error(data.error);
            }
        },
        error: function (error) {
            console.log(error);
            alertify.error("Không tìm thấy model");
        },
    });
}

function loadPipelineData() {
    console.log("load pipeline data");
    response = loadLogTrainModel();
    console.log("LOG");
    console.log(response);

    count_waiting_task = response?.task_instances.fillter(
        (value) =>
            value.state === undefined ||
            value.state === null ||
            value.state === "running"
    ).length;
    if (count_waiting_task <= 0) {
        clearInterval(interval);
    }
}

$(document).ready(function () {
    alertify.set("notifier", "position", "top-right");
    $("input[type=radio][name=flexRadioDefault]").change(function () {
        var selectedValue = $(this).val();
        if (selectedValue === "DIFF") {
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


        trainModel(event);

        interval = setInterval(loadLogTrainModel, 5000);
    });

    $("#submitModel").on("click", function (event) {
        event.preventDefault();
        console.log("submit model");
        submitModel();
    });


    $("#reloadModel").on("click", function (event) {
        event.preventDefault();
        console.log("submit model");
        let model_id = $("#modelId").text().trim();
        getTranningModelDetail(model_id);
    });

    var el = document.getElementById("curr");
    var r = document.getElementById("size");
    el.innerText = r.valueAsNumber + "%";
    r.addEventListener("change", () => {
        el.innerText = r.valueAsNumber + "%";
    });
});
