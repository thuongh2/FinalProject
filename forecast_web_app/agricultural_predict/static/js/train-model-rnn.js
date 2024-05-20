const URL_SERVER = "http://localhost:5000";

$("#model_name").change(function () {
  sessionStorage.clear();
});


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
        line: { color: "#17BECF" },
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
  const username = $("#username").text();
  const agricutural_name = sessionStorage.getItem("agricutural_name");

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

  data = {
    model_name: model_name,
    model_data: model_data,
    username: username,
    agricutural_name: agricutural_name,
    argument: argument,
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

      if (data.status === "SUCCESS") {
        $("#model_detail_name").text(data.model_name);
        $("#model_detail_mape").text(data.evaluate["mape"].toFixed(2) || 0);
        $("#model_detail_rmse").text(data.evaluate["rmse"].toFixed(2) || 0);
        alertify.success("Train model thành công!");
        plotChartData(data.plot_data);
        $("#detail-tab").tab("show");

      } else {
        alertify.error(data.error);
        alertify.error("Train model thất bại!");
      }
    },
    error: function (error) {
      alert("error" + error);

      alertify.error(error);
    },
  });
  await $("#modelTrainingInProcess").modal("hide");
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

      if (data.$oid) {
        window.location.href = "/detail-model?model_id=" + data.$oid;
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

$(document).ready(function () {
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
