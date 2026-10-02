/**
 * AlgoGPT Live Forecast Demo - Interactive Client
 * Supports dynamic symbol and interval switching (15m, 1h, 1d)
 */

let forecastData = null;
let currentSymbol = 'BANKNIFTY';
let currentInterval = '15m';
let chartInstance = null;
let viewMode = 'interactive'; // 'interactive' or 'static'

document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  setupEventListeners();
  await loadForecastData();
  renderDashboard();
});

function initTheme() {
  const savedTheme = localStorage.getItem('algogpt_theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeButton(savedTheme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  const next = current === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('algogpt_theme', next);
  updateThemeButton(next);
  if (viewMode === 'interactive' && chartInstance) {
    renderInteractiveChart();
  }
}

function updateThemeButton(theme) {
  const btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.textContent = theme === 'dark' ? '☀️ Light' : '🌙 Dark';
  }
}

function setupEventListeners() {
  // Theme toggle
  const themeBtn = document.getElementById('theme-toggle');
  if (themeBtn) themeBtn.addEventListener('click', toggleTheme);

  // Symbol tabs
  const symbolBtns = document.querySelectorAll('.symbol-tab');
  symbolBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      symbolBtns.forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      currentSymbol = e.target.getAttribute('data-symbol');
      renderDashboard();
    });
  });

  // Interval / Timeframe tabs
  const intervalBtns = document.querySelectorAll('.interval-tab');
  intervalBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      intervalBtns.forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      currentInterval = e.target.getAttribute('data-interval');
      renderDashboard();
    });
  });

  // View toggle buttons (Interactive vs Static)
  const viewBtns = document.querySelectorAll('.view-btn');
  viewBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      viewBtns.forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      viewMode = e.target.getAttribute('data-view');
      toggleChartView();
    });
  });
}

async function loadForecastData() {
  try {
    const res = await fetch('data/forecast_data.json');
    if (!res.ok) throw new Error('Network error loading forecast data');
    forecastData = await res.json();
  } catch (err) {
    console.error('Failed to load forecast data:', err);
  }
}

function getActiveData() {
  if (!forecastData || !forecastData[currentSymbol]) return null;
  const symData = forecastData[currentSymbol];
  // Support both nested interval structure and fallback
  if (symData[currentInterval]) {
    return symData[currentInterval];
  }
  return symData['15m'] || symData;
}

function renderDashboard() {
  const data = getActiveData();
  if (!data) return;

  // Update Header & Metadata
  const titleEl = document.getElementById('current-symbol-title');
  if (titleEl) {
    titleEl.textContent = `${data.symbol} (${data.interval}) Forecast Dashboard`;
  }

  const timeEl = document.getElementById('update-time');
  if (timeEl) timeEl.textContent = data.last_updated;

  const intvEl = document.getElementById('data-interval-text');
  if (intvEl) intvEl.textContent = data.interval;

  const exchEl = document.getElementById('data-exchange-text');
  if (exchEl) exchEl.textContent = data.exchange;

  const spotEl = document.getElementById('data-spot-price');
  if (spotEl && data.spot_close) {
    spotEl.textContent = `₹${data.spot_close.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
  }

  // Update Metrics
  const metrics = data.metrics;
  const upsideEl = document.getElementById('upside-prob');
  if (upsideEl) {
    upsideEl.textContent = `${metrics.upside_prob}%`;
    upsideEl.className = 'metric-value ' + (metrics.upside_prob >= 50 ? 'green' : 'orange');
  }

  const volEl = document.getElementById('vol-amp-prob');
  if (volEl) {
    volEl.textContent = `${metrics.vol_amp_prob}%`;
    volEl.className = 'metric-value ' + (metrics.vol_amp_prob > 60 ? 'orange' : 'blue');
  }

  const expReturnEl = document.getElementById('exp-return');
  if (expReturnEl) {
    const retPrefix = metrics.expected_return >= 0 ? '+' : '';
    expReturnEl.textContent = `${retPrefix}${metrics.expected_return}%`;
    expReturnEl.className = 'metric-value ' + (metrics.expected_return >= 0 ? 'green' : 'orange');
  }

  const varEl = document.getElementById('var-95');
  if (varEl) {
    varEl.textContent = `${metrics.var_95}%`;
    varEl.className = 'metric-value ' + (Math.abs(metrics.var_95) > 1.5 ? 'purple' : 'blue');
  }

  // Update Chart Description with Horizon Description
  const chartDescEl = document.querySelector('.chart-desc');
  if (chartDescEl && data.horizon_desc) {
    chartDescEl.innerHTML = `The chart below visualizes <strong>${data.symbol}</strong> historical prices alongside Kronos probabilistic forecasts for the <strong>${data.horizon_desc}</strong>. The orange line represents the mean of <strong>30 Monte Carlo trajectory simulations</strong>, surrounded by <strong>50% and 90% confidence bands</strong> capturing market regime uncertainty.`;
  }

  // Render Predictions Table
  renderTable(data);

  // Render Chart
  if (viewMode === 'interactive') {
    renderInteractiveChart();
  } else {
    updateStaticChart();
  }
}

function updateStaticChart() {
  const chartImg = document.querySelector('.chart-img');
  if (chartImg) {
    chartImg.src = `img/prediction_chart_${currentInterval}.png?t=${Date.now()}`;
    chartImg.onerror = () => { chartImg.src = `img/prediction_chart.png?t=${Date.now()}`; };
  }
}

function toggleChartView() {
  const interactiveWrapper = document.getElementById('interactive-wrapper');
  const staticWrapper = document.getElementById('static-wrapper');

  if (viewMode === 'interactive') {
    if (interactiveWrapper) interactiveWrapper.style.display = 'block';
    if (staticWrapper) staticWrapper.style.display = 'none';
    renderInteractiveChart();
  } else {
    if (interactiveWrapper) interactiveWrapper.style.display = 'none';
    if (staticWrapper) staticWrapper.style.display = 'block';
    updateStaticChart();
  }
}

function renderTable(data) {
  const tbody = document.getElementById('predictions-tbody');
  if (!tbody || !data.predictions) return;
  tbody.innerHTML = '';

  const lastClose = data.historical[data.historical.length - 1].close;

  data.predictions.forEach((p, idx) => {
    const tr = document.createElement('tr');
    const delta = ((p.close - lastClose) / lastClose) * 100;
    const deltaClass = delta >= 0 ? 'positive' : 'negative';
    const deltaStr = (delta >= 0 ? '+' : '') + delta.toFixed(2) + '%';

    tr.innerHTML = `
      <td>Step +${idx + 1} (${p.time})</td>
      <td class="price-cell">₹${p.open.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
      <td class="price-cell">₹${p.high.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
      <td class="price-cell">₹${p.low.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
      <td class="price-cell">₹${p.close.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
      <td class="${deltaClass} price-cell">${deltaStr}</td>
      <td style="color: var(--subtle-text)">₹${p.lower_10.toFixed(1)} — ₹${p.upper_90.toFixed(1)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderInteractiveChart() {
  const data = getActiveData();
  if (!data) return;

  const hist = data.historical.slice(-25); // latest 25 bars
  const pred = data.predictions;

  const canvas = document.getElementById('forecast-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (chartInstance) {
    chartInstance.destroy();
  }

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';
  const textColor = isDark ? '#94a3b8' : '#64748b';

  // Format labels nicely with Indian market close & session awareness
  const formatLabel = (tStr) => {
    if (tStr.includes(' ')) {
      return tStr.split(' ')[1]; // HH:MM
    }
    return tStr.slice(5); // MM-DD for daily
  };

  const labels = [
    ...hist.map((b) => {
      if (b.time.endsWith('15:30')) return `${b.time.split(' ')[1]} Close`;
      return formatLabel(b.time);
    }),
    ...pred.map((p, i) => {
      const isNewDay = i === 0 || p.time.split(' ')[0] !== pred[i-1].time.split(' ')[0];
      const timePart = formatLabel(p.time);
      const datePart = isNewDay && p.time.includes(' ') ? `${p.time.slice(5, 10)} ` : '';
      return `+${i+1} (${datePart}${timePart})`;
    })
  ];

  const histPrices = hist.map(b => b.close);
  const lastPrice = histPrices[histPrices.length - 1];

  // Align prediction arrays with anchor point
  const nullPadding = new Array(hist.length - 1).fill(null);
  const meanForecast = [...nullPadding, lastPrice, ...pred.map(p => p.close)];
  const upper90 = [...nullPadding, lastPrice, ...pred.map(p => p.upper_90)];
  const lower10 = [...nullPadding, lastPrice, ...pred.map(p => p.lower_10)];
  const upper75 = [...nullPadding, lastPrice, ...pred.map(p => p.upper_75)];
  const lower25 = [...nullPadding, lastPrice, ...pred.map(p => p.lower_25)];

  const datasets = [
    {
      label: `Historical (${data.interval})`,
      data: [...histPrices, ...new Array(pred.length).fill(null)],
      borderColor: '#2563eb',
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      pointRadius: 1.5,
      tension: 0.15
    },
    {
      label: 'Kronos Mean Forecast',
      data: meanForecast,
      borderColor: '#ea580c',
      backgroundColor: 'transparent',
      borderWidth: 2.8,
      pointRadius: 2.5,
      pointBackgroundColor: '#ea580c',
      tension: 0.2
    },
    {
      label: '50% Confidence Upper',
      data: upper75,
      borderColor: 'transparent',
      backgroundColor: 'rgba(249, 115, 22, 0.22)',
      fill: '+1',
      pointRadius: 0
    },
    {
      label: '50% Confidence Lower',
      data: lower25,
      borderColor: 'transparent',
      backgroundColor: 'transparent',
      pointRadius: 0
    },
    {
      label: '90% Confidence Upper',
      data: upper90,
      borderColor: 'rgba(249, 115, 22, 0.3)',
      borderDash: [4, 4],
      backgroundColor: 'rgba(249, 115, 22, 0.10)',
      fill: '+1',
      pointRadius: 0
    },
    {
      label: '90% Confidence Lower',
      data: lower10,
      borderColor: 'rgba(249, 115, 22, 0.3)',
      borderDash: [4, 4],
      backgroundColor: 'transparent',
      pointRadius: 0
    }
  ];

  // Add sample Monte Carlo paths
  if (pred[0].samples && pred[0].samples.length >= 3) {
    for (let s = 0; s < 3; s++) {
      const sPath = [...nullPadding, lastPrice, ...pred.map(p => p.samples[s])];
      datasets.push({
        label: `MC Simulation Path #${s+1}`,
        data: sPath,
        borderColor: 'rgba(249, 115, 22, 0.35)',
        borderWidth: 1,
        borderDash: [2, 2],
        pointRadius: 0,
        fill: false
      });
    }
  }

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 350 },
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            color: textColor,
            font: { size: 11, family: 'Inter' },
            filter: (item) => !item.text.includes('Lower')
          }
        },
        tooltip: {
          callbacks: {
            title: function(tooltipItems) {
              if (!tooltipItems.length) return '';
              const idx = tooltipItems[0].dataIndex;
              if (idx < hist.length) {
                return `${hist[idx].time} IST (NSE Historical Market Bar)`;
              } else {
                const pIdx = idx - hist.length;
                return `${pred[pIdx].time} IST (Kronos Forecast Step +${pIdx + 1})`;
              }
            },
            label: function(context) {
              if (context.parsed.y === null) return null;
              return `${context.dataset.label}: ₹${context.parsed.y.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: gridColor },
          ticks: { color: textColor, maxTicksLimit: 14, font: { size: 10 } }
        },
        y: {
          grid: { color: gridColor },
          ticks: {
            color: textColor,
            font: { size: 10 },
            callback: (val) => '₹' + Number(val).toLocaleString('en-IN')
          }
        }
      }
    }
  });
}
