var URL_SERVER = "http://agricultural.io.vn:5001";

// Roll price list
document.addEventListener("DOMContentLoaded", function () {
  const table = document.getElementById("priceDataTable");
  const scrollbar = document.getElementById("scrollbar");

  scrollbar.addEventListener("scroll", function () {
    table.style.transform = `translateY(-${scrollbar.scrollTop}px)`;
  });

  function updateScrollbarHeight() {
    const totalHeight = table.clientHeight;
    const visibleHeight = scrollbar.clientHeight;
    const scrollHeight = (visibleHeight / totalHeight) * visibleHeight;
    scrollbar.style.height = `${scrollHeight}px`;
  }

  updateScrollbarHeight();

  window.addEventListener("resize", updateScrollbarHeight);
});

// Predictive processing
$(document).ready(function () {
  var selectedValue = $("#data_name").val();
  callPlotChart(selectedValue);

$("#model_name").change(async function () {
    var modelName = $(this).val();
    var agriculturalType = $('#agriculturalType').text();
    // await showLoading();
    // await clearChart();
    return new Promise((resolve, reject) => {
        console.log("First function is running");
        getDataForLoadChart(modelName, agriculturalType);
    });

});


  $("#data_name").change(function () {
    showLoading();
    clearChart();
    callPlotChart();
  });
  $("#model_time").change(function () {
    showLoading();
    clearChart();
    callPlotChart();
  });
});

async function getDataForLoadChart(modelName, agriculturalType) {
  if (modelName) {
    await $.ajax({
      url: URL_SERVER + "/get-data-from-model/" + modelName,
      type: "GET",
      data: {'agricultural_type': agriculturalType},
      success: function (response) {
        var modelDataSelect = $("#data_name");
        modelDataSelect
            .empty()
            .append('<option value="">Chọn dữ liệu</option>'); // Clear current options
        // Add new options from response
        $.each(response, function (key, value) {
          modelDataSelect.append(
            '<option data-value="' +
              value.type +
              '" value="' +
              value.data +
              '">' +
              value.name +
              "</option>"
          );
        });
      },
      error: function (xhr) {
        $("#model_data").empty().append('<option value="">Chọn dữ liệu</option>');
      },
    });
  } else {
    $("#model_data").empty().append('<option value="">Chọn dữ liệu</option>');
  }
}

document.addEventListener("DOMContentLoaded", function () {
  $("#model_name").val("LSTM");
  $("#model_name").change();
});

function plotChart(data) {
  $("#myDiv").empty();

  var traces = [];
  var minY = Infinity;
  var maxY = -Infinity;

  data.forEach((value, index) => {
    var xDim = new Array();
    for (var i in value.x) xDim.push(new Date(value.x[i]));
    var yDim = value.y;
    var mode = value.mode;
    var name = value.name;
    var lineColor, fillColor, fillOpacity;

    if (index === 0) {
      lineColor = "rgba(0, 110, 255, 0.7)";
      fillColor = "rgba(173, 216, 230, 0.7)";
      fillOpacity = 1;
    } else if (index === 1) {
      lineColor = "rgba(250,128,114,0.8)";
      fillColor = "rgba(240,230,140,0.4)";
      fillOpacity = 1;
    }

    minY = Math.min(minY, Math.min(...yDim));
    maxY = Math.max(maxY, Math.max(...yDim));

    var trace = {
      x: xDim,
      y: yDim,
      type: "scatter",
      mode: mode,
      name: name,
      line: { color: lineColor },
      fill: "tozeroy",
      fillcolor: fillColor,
      visible: true,
      opacity: fillOpacity,
    };
    traces.push(trace);
  });

  var layout = {
    title: {
      text: "BIỂU ĐỒ DỰ ĐOÁN",
      font: {
        family: "Arial",
        size: 20,
        weight: "bold",
      },
    },
    xaxis: {
      type: "date",
      title: "Ngày",
      showgrid: false,
    },
    yaxis: {
      title: "Giá (đồng)",
      showgrid: false,
      range: [minY * 0.98, maxY * 1.02],
    },
  };

  Plotly.newPlot("myDiv", traces, layout);
}

// Xử lý loading
function loadData(state) {
  if (state) {
    $("#loadingIndicator").show();
  } else {
    $("#loadingIndicator").hide();
  }
}

function showLoading() {
  $("#loadingIndicator").show();
}

function hideLoading() {
  $("#loadingIndicator").hide();

}

function clearChart() {
  $("#myDiv").empty();
}

async function callPlotChart(data) {
  await loadData(true);

  const url = URL_SERVER + "/load-chart";
  const modelName = $("#model_name").val();
  const modelData = $("#data_name").val();
  const modelTime = $("#model_time").val();

  const params = {
    model_name: modelName,
    model_data: modelData,
    model_time: modelTime,
  };
  console.log(params)
  const queryString = $.param(params);
  const fullUrl = url + "?" + queryString;

  await $.ajax({
    url: fullUrl,
    type: "GET",
    contentType: "application/json",
    success: function (response) {
      console.log(response);
      hideLoading();
      Plotly.purge("myDiv");
      plotChart(response.plot_data);
      plotActualPrice(response.price_actual);
      plotPredictPrice(response.price_forecast);
    },
    error: function (xhr, status, error) {
      console.log(error);
      hideLoading();
      $("#myDiv").append(`<p class='text-center text-danger font-weight-bold'>Không tìm thấy mô hình</p>`);
    },
  });
}

function plotActualPrice(data) {
  let price = JSON.parse(data);
  $("#priceActualDate").text(formattedDate(price.date));
  $("#priceActual").text(
    parseInt(price.price).toLocaleString("it-IT", {
      style: "currency",
      currency: "VND",
    })
  );
}

function formattedDate(timestamp){
  const date = new Date(timestamp);

  const day = ('0' + date.getDate()).slice(-2);
  const month = ('0' + (date.getMonth() + 1)).slice(-2);
  const year = date.getFullYear();

  return `${day}-${month}-${year}`;
}

function plotPredictPrice(data) {
  let price = JSON.parse(data);
  console.log(price);

  $("#pricePredictDate").text(formattedDate(price.date));

  $("#pricePredict").text(
    parseInt(price.price).toLocaleString("it-IT", {
      style: "currency",
      currency: "VND",
    })
  );
}

$(document).ready(function () {
  $("#callPlotChart").change(function () {
    var selectedValue = $(this).val();
    callPlotChart(selectedValue);
  });
});



// Display price
function updateTableFromCSV(csvFilePath) {
  d3.csv(csvFilePath, function (data) {
    data.reverse();
    var data = data.slice(0, 30);
    d3.select("#priceDataTable tbody").selectAll("*").remove();
    var rows = d3
      .select("#priceDataTable tbody")
      .selectAll("tr")
      .data(data)
      .enter()
      .append("tr");
    rows.each(function (d) {
      console.log(d);
      d3.select(this).append("td").text(d.date);
      d3.select(this).append("td").text(d.price);
    });
  });
}

document.addEventListener("DOMContentLoaded", function () {
  var csvFilePath = $("#data_name").val();
  updateTableFromCSV(csvFilePath);
});
