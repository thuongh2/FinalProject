const URL_SERVER = "http://localhost:5000"

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

  $("#model_name").change(function () {
    modelName = $(this).val();
    showLoading();
    clearChart();
    callPlotChart();
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
    xDim = new Array();
    for (i in value.x) xDim.push(new Date(value.x[i]));
    yDim = value.y;
    mode = value.mode;
    name = value.name;
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

  const queryString = $.param(params);
  const fullUrl = url + "?" + queryString;

  await $.ajax({
    url: fullUrl,
    type: "GET",
    contentType: "application/json",
    success: function (response) {
      hideLoading();
      Plotly.purge("myDiv");
      plotChart(response.plot_data);
    },
    error: function (xhr, status, error) {
      console.log(error);
      hideLoading();
    },
  });
}

$(document).ready(function () {
  $("#callPlotChart").change(function () {
    var selectedValue = $(this).val();
    callPlotChart(selectedValue);
  });
});

// Active card
$(".price-agricutural").click(function () {
  var id = $(this).attr("id");
  localStorage.setItem("activeTab", id);
});

$(document).ready(function () {
  var activeTabId = localStorage.getItem("activeTab");
  if (activeTabId) {
    $(".price-agricutural").removeClass("active-card");
    $("#" + activeTabId).addClass("active-card");
  }
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
      d3.select(this).append("td").text(d.date);
      d3.select(this).append("td").text(d.price);
    });
  });
}

document.addEventListener("DOMContentLoaded", function () {
  var csvFilePath = $("#data_name").val();
  updateTableFromCSV(csvFilePath);
});
