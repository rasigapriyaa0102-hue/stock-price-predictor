// static/js/result_chart.js
document.addEventListener('DOMContentLoaded', function () {
  if (typeof resultData === 'undefined') return;

  const dates = resultData.history_dates;
  const actual = resultData.history_actual;
  const predicted = resultData.history_predicted;

  const traceActual = {
    x: dates,
    y: actual,
    mode: 'lines',
    name: 'Actual',
    line: { width: 2, color: '#19b6b6' }
  };

  // predicted may contain NaN front-padded; treat them as nulls
  const predY = predicted.map(v => (v === null ? null : v));
  const tracePred = {
    x: dates,
    y: predY,
    mode: 'lines',
    name: 'Predicted',
    line: { width: 2, dash: 'dash', color: '#83e0d9' }
  };

  const layout = {
    margin: { t: 10, r: 20, l: 40, b: 40 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(255,255,255,0.02)',
    xaxis: { showgrid: false, tickangle: -45 },
    yaxis: { showgrid: true },
    showlegend: true,
  };

  Plotly.newPlot('chart', [traceActual, tracePred], layout, {responsive:true, displayModeBar:false});
});
