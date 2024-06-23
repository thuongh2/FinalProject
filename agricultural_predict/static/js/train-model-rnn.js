var URL_SERVER = "http://agricultural.io.vn:5001"


var interval = null;
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

$("#model_name").change(function () {
  sessionStorage.clear();
});

$("#data_name").change(function () {
  const selectedValue = JSON.parse($(this).val().replace(/'/g, '"'));
  console.log(selectedValue);
  callDrawPlot(selectedValue.data);
  handelStoreSession("data_url", selectedValue.data);
  handelStoreSession("agricultural_name", selectedValue.type);
  handelStoreSession("model_name", $("#model_name").find(":selected").text());
  return;
});

$("#smoothing_data").change(function () {
  plotChart();
  return;
});

$("#smoothing_value").change(function () {
  plotChart();
  return;
});

function handelStoreSession(key, value) {
  sessionStorage.setItem(key, value);
}


function plotStationaryData(data, rootNode) {
  data.forEach((value, index) => {
    console.log(value.x);
    xDim = new Array();
    for (i in value.x) xDim.push(new Date(value.x[i]));
    yDim = value.y;
    mode = value.mode;

    trace = { x: xDim, y: yDim, type: "scatter", mode: "lines" };
    console.log(trace);
    const nodeName = rootNode + index;
    const node = document.createElement("div");
    node.id = nodeName;
    console.log(node);
    document.getElementById(rootNode).appendChild(node);

    var layout = {
      title: "Biểu đồ giá",
    };
    Plotly.newPlot(nodeName, [trace], layout);
  });
}


async function plotChart() {
  sessionStorage.clear();
  await $("#loadChartLoading").removeClass("d-none");
  var model_data = JSON.parse(
    $("#data_name").find(":selected").val().replace(/'/g, '"')
  );
  var smoothing_type = $("#smoothing_data").find(":selected").val();
  var smoothing_value = $("#smoothing_value").val();
  handelStoreSession("data_url", model_data.data);
  handelStoreSession("agricultural_name", model_data.type);
  handelStoreSession("model_name", $("#model_name").find(":selected").text());

  handelStoreSession("smoothing_data", smoothing_type);
  handelStoreSession("smoothing_value", smoothing_value);

  await $.ajax({
    url: URL_SERVER + "/get-data-self-train",
    method: "GET",
    data: {
      model_data: model_data.data,
      smoothing_type: smoothing_type,
      smoothing_value: smoothing_value,
    },
    success: function (response) {
      console.log(response);
      $("#myChart").empty();
      plotStationaryData(response, "myChart");
    },
    error: function (xhr, status, error) {
      console.log(xhr);
      alertify.error("Lỗi khi load dữ liệu!");
    },
  });
  $("#loadChartLoading").addClass("d-none");
}


async function callDrawPlot(selectedValue) {
  sessionStorage.clear();
  
  await plotChart();
  // d3.csv(selectedValue, function (err, rows) {
  //   function unpack(rows, key) {
  //     return rows.map(function (row) {
  //       if (key === "date") {
  //         return new Date(row[key]);
  //       } else {
  //         return parseFloat(row[key]);
  //       }
  //     });
  //   }

  //   Object.keys(rows[0]).forEach(function (row) {
  //     var data = new Array();
  //     if (row === "date") return;
  //     var trace = {
  //       type: "scatter",
  //       mode: "lines",
  //       name: row,
  //       x: unpack(rows, "date"),
  //       y: unpack(rows, row),
  //       line: { color: "#17BECF" },
  //     };

  //     console.log(trace);
  //     const nodeName = "myChart" + row;
  //     const node = document.createElement("div");
  //     node.id = nodeName;

  //     document.getElementById("myChart").appendChild(node);

  //     var layout = {
  //       title: "Biểu đồ giá " + row,
  //     };

  //     Plotly.newPlot(nodeName, [trace], layout);
  //   });
  // });
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

async function trainModel() {
  const model_name = sessionStorage.getItem("model_name");
  if (!model_name) {
    alertify.error("Vui lòng chọn tên model");
  }
  const model_data = sessionStorage.getItem("data_url");
  if (!model_data) {
    alertify.error("Vui lòng chọn dữ liệu");
  }
  const username = $("#username").text().trim();
  const agricultural_name = sessionStorage.getItem("agricultural_name");

  const layers_data = sessionStorage.getItem("layers_data");
  if (!layers_data) {
    alertify.error("Vui lòng thêm layers dữ liệu");
  }

  var formEl = document.getElementById("trainModelForm");
  var formData = new FormData(formEl);
  var argument = {};

  for (var pair of formData.entries()) {
    var name = pair[0];
    var value = pair[1];

    if (value === undefined || value === "") {
      await alertify.error("Vui lòng điền " + name);
      return;
    }
    value = parseInt(value);
    if (name == "size") value = value / 100;
    argument[name] = value;
  }

  const layersDataParsed = JSON.parse(layers_data);
  argument["layers_data"] = layersDataParsed;

  const model_id = $("#modelId").text().trim();

  data = {
    model_name: model_name,
    model_data: model_data,
    username: username,
    agricultural_name: agricultural_name,
    argument: argument,
    model_id: model_id,
  };

  await $("#modelTrainingInProcess").modal("show");
  await $.ajax({
    url: URL_SERVER + "/train-model-rnn-data",
    method: "POST",
    contentType: "application/json; charset=utf-8",
    data: JSON.stringify(data),
    success: function (response) {
      const data = JSON.parse(response);
      console.log(data);

      handelStoreSession("model_submit_detail", JSON.stringify(data));
      handelStoreSession("dags_run_id", data.dag_run_id);

      const container = $("#pipeline-step");
      container.empty();
      container.append(createPipelineSetupStep("running"));
    },
    error: function (error) {
      alertify.error("Thất bại" + error);
    },
  });
}

async function submitModel() {
  data = sessionStorage.getItem("model_submit_detail");
  if (!data) {
    alertify.error("Vui lòng train model trước khi submit");
    return;
  }

  await $.ajax({
    url: URL_SERVER + "/submit-train-model-data",
    method: "POST",
    contentType: "application/json; charset=utf-8",
    data: JSON.stringify(data),
    success: function (response) {
      const data = JSON.parse(response);
      console.log(data);

      if (data) {
        sessionStorage.clear();
        window.location.href = "/detail-model?model_id=" + data;
        return;
      }
      alertify.success("Submit model thành công");
    },
    error: function (error) {
      alert("error" + error);

      alertify.error("Submit model không thành công");
    },
  });
  $("#modelTrainingInProcess").modal("hide");
}

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
  if (
    dag_run_id === "undefined" ||
    dag_run_id === "null" ||
    dag_run_id === ""
  ) {
    console.error("Không tìm thấy dag id");
    return;
  }

  let username = "airflow";
  let password = "airflow";
  let auth = btoa(`${username}:${password}`);
  let model_id = $("#modelId").text().trim();

  var url = URL_SERVER + "/pipeline/{dag_run_id}" + "/" + dag_run_id;
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

  await $.ajax(settings).done(function (response) {
    response = JSON.parse(response);
    console.log(response);
    responseData = response;
    if (response.total_entries === 0) {
      console.log("Không tìm thấy task");
      return;
    }

    const container = $("#pipeline-step");
    container.empty();
    container.append(createPipelineSetupStep("success"));
    container.append(arrowTemplate);
    var count_waiting_task = 0;

    response.task_instances.sort(compare);
    console.log(response.task_instances);

    response.task_instances.forEach((value, index) => {
      const html = createPipelineStep(value);

      container.append(html);
      if (index !== response.task_instances.length - 1) {
        container.append(arrowTemplate);
      }
      if (value.state === "success") {
        count_waiting_task++;
      }
      if (value.state === "failed") {
        clearInterval(interval);
        alertify.error("Model training thất bại");
        return;
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

function createPipelineStep(value) {
  return pipelineTemplate
    .replace("{{value}}", value.task_id)
    .replace("{{status}}", statusMapping.get(value.state || "waiting"))
    .replace("{{status-class}}", statusClass.get(value.state) || "secondary")
    .replace(
      "{{status-display}}",
      value.state === "running" || value.state === "queued" ? "" : "d-none"
    );
}

function createPipelineSetupStep(state) {
  return pipelineTemplate
    .replace("{{value}}", "Setup Pipeline")
    .replace("{{status}}", statusMapping.get(state || "waiting"))
    .replace("{{status-class}}", statusClass.get(state) || "secondary")
    .replace(
      "{{status-display}}",
      state === "running" || state === "queued" ? "" : "d-none"
    );
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

$(document).ready(function () {
  // config for alertify notify
  alertify.set("notifier", "position", "top-right");
  $("#trainModelForm").submit(function (event) {
    event.preventDefault();
    alertify.set("notifier", "position", "top-right");

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

// Thêm/bớt số layer + lưu vào session
document.addEventListener("DOMContentLoaded", function () {
  const addButton = document.getElementById("add-input");
  const removeButton = document.getElementById("remove-input");
  const inputContainer = document.getElementById("input-container");
  let inputCount = 1;

  function saveLayersToSession() {
    const layers = [];
    const inputs = inputContainer.querySelectorAll("input");
    inputs.forEach((input, index) => {
      layers.push({
        id: `LAYER ${index + 1}`,
        units: input.value,
      });
    });
    sessionStorage.setItem("layers_data", JSON.stringify(layers));
  }

  function addInput() {
    inputCount++;
    const newInputDiv = document.createElement("div");
    newInputDiv.classList.add("form-check", "mt-3");
    newInputDiv.innerHTML = `
                    <div>
                        <label for="flexInputDefault${inputCount}"><strong>LAYER ${inputCount}</strong></label>
                        <div class="d-flex align-items-center">
                            <span>Unit:</span>
                            <input class="form-control ml-2" type="text" id="flexInputDefault${inputCount}" name="flexInputDefault${inputCount}">
                        </div>
                    </div>
                `;
    inputContainer.appendChild(newInputDiv);
    newInputDiv
      .querySelector("input")
      .addEventListener("input", saveLayersToSession);
    saveLayersToSession();
  }

  function removeInput() {
    if (inputContainer.childElementCount > 1) {
      inputContainer.removeChild(inputContainer.lastElementChild);
      inputCount--;
      saveLayersToSession();
    }
  }

  function loadLayersFromSession() {
    const storedLayersJson = sessionStorage.getItem("layers_data");
    if (storedLayersJson) {
      const storedLayers = JSON.parse(storedLayersJson);
      inputContainer.innerHTML = "";
      storedLayers.forEach((layer, index) => {
        const newInputDiv = document.createElement("div");
        newInputDiv.classList.add("form-check", "mt-3");
        newInputDiv.innerHTML = `
                            <div>
                                <label for="flexInputDefault${
                                  index + 1
                                }"><strong>${layer.id}</strong></label>
                                <div class="d-flex align-items-center">
                                    <span>Unit:</span>
                                    <input class="form-control ml-2" type="text" id="flexInputDefault${
                                      index + 1
                                    }" name="flexInputDefault${
          index + 1
        }" value="${layer.units}">
                                </div>
                            </div>
                        `;
        inputContainer.appendChild(newInputDiv);
        newInputDiv
          .querySelector("input")
          .addEventListener("input", saveLayersToSession);
      });
      inputCount = storedLayers.length;
    }
  }

  addButton.addEventListener("click", addInput);
  removeButton.addEventListener("click", removeInput);

  loadLayersFromSession();

  inputContainer
    .querySelector("input")
    .addEventListener("input", saveLayersToSession);
});

// Lưu timestep, epoch, batchsize vào session
function saveTimestepToSession(key, value) {
  sessionStorage.setItem(key, value);
}

function saveEpochsToSession(key, value) {
  sessionStorage.setItem(key, value);
}

function saveBatchsizeToSession(key, value) {
  sessionStorage.setItem(key, value);
}

function loadFromSession() {
  const epochs = document.getElementById("epochs");
  const timestep = document.getElementById("timestep");
  const batchsize = document.getElementById("batchsize");

  const epochValue = sessionStorage.getItem("epochs");
  if (epochValue) {
    epochs.value = epochValue;
  }

  const timestepValue = sessionStorage.getItem("timestep");
  if (timestepValue) {
    timestep.value = epochValue;
  }

  const batchSizeValue = sessionStorage.getItem("batchsize");
  if (batchSizeValue) {
    batchsize.value = batchSizeValue;
  }
}

document.addEventListener("DOMContentLoaded", function () {
  loadFromSession();
});
