/**
 * AlgoGPT Live Forecast Demo - Interactive Client
 * Fathomless-Inspired Redesign with Immersive Particle System
 * Supports dynamic symbol and interval switching (15m, 1h, 1d)
 */

let forecastData = null;
let currentSymbol = 'BANKNIFTY';
let currentInterval = '15m';
let chartInstance = null;
let viewMode = 'interactive'; // 'interactive' or 'static'

document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  initParticles();
  initScrollAnimations();
  initDepthMeter();
  initNavbar();
  setupEventListeners();
  await loadForecastData();

  // URL parameters support for direct linking & testing
  const urlParams = new URLSearchParams(window.location.search);
  const symParam = urlParams.get('symbol');
  if (symParam && forecastData && forecastData[symParam]) {
    currentSymbol = symParam;
    document.querySelectorAll('.symbol-tab').forEach(b => {
      b.classList.toggle('active', b.getAttribute('data-symbol') === symParam);
    });
  }
  const intvParam = urlParams.get('interval');
  if (intvParam) {
    currentInterval = intvParam;
    document.querySelectorAll('.interval-tab').forEach(b => {
      b.classList.toggle('active', b.getAttribute('data-interval') === intvParam);
    });
  }
  const viewParam = urlParams.get('view');
  if (viewParam === 'static') {
    viewMode = 'static';
    document.querySelectorAll('.view-btn').forEach(b => {
      b.classList.toggle('active', b.getAttribute('data-view') === 'static');
    });
    const interactiveWrapper = document.getElementById('interactive-wrapper');
    const staticWrapper = document.getElementById('static-wrapper');
    if (interactiveWrapper) interactiveWrapper.style.display = 'none';
    if (staticWrapper) staticWrapper.style.display = 'block';
  }

  renderDashboard();
});

/* ==========================================================================
   PARTICLE SYSTEM – Floating Marine Snow / Bioluminescence
   ========================================================================== */
function initParticles() {
  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let particles = [];
  let mouse = { x: -1000, y: -1000 };
  let animId;
  const PARTICLE_COUNT = 80;
  const MAX_DEPTH_PARTICLE_SIZE = 3;

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  document.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  });

  // Create particles
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push(createParticle());
  }

  function createParticle() {
    const scrollFactor = window.scrollY / (document.body.scrollHeight - window.innerHeight || 1);
    return {
      x: Math.random() * (canvas.width || window.innerWidth),
      y: Math.random() * (canvas.height || window.innerHeight),
      size: Math.random() * MAX_DEPTH_PARTICLE_SIZE + 0.5,
      speedX: (Math.random() - 0.5) * 0.3,
      speedY: Math.random() * 0.15 + 0.05, // gentle drift downward (marine snow)
      opacity: Math.random() * 0.5 + 0.1,
      hue: Math.random() > 0.7 ? 180 + Math.random() * 40 : 170 + Math.random() * 20, // cyan-teal range
      pulseSpeed: Math.random() * 0.02 + 0.005,
      pulsePhase: Math.random() * Math.PI * 2,
      bioluminescent: Math.random() > 0.85 // ~15% of particles glow
    };
  }

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scrollProgress = window.scrollY / (document.body.scrollHeight - window.innerHeight || 1);
    // Deepen ambient as user scrolls
    const ambientAlpha = 0.02 + scrollProgress * 0.03;

    particles.forEach(p => {
      // Update position
      p.x += p.speedX;
      p.y += p.speedY;

      // Mouse proximity interaction
      const dx = mouse.x - p.x;
      const dy = mouse.y - p.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 150) {
        const force = (150 - dist) / 150;
        p.x -= dx * force * 0.008;
        p.y -= dy * force * 0.008;
      }

      // Wrap around
      if (p.y > canvas.height + 10) { p.y = -10; p.x = Math.random() * canvas.width; }
      if (p.x > canvas.width + 10) p.x = -10;
      if (p.x < -10) p.x = canvas.width + 10;

      // Pulsing opacity for bioluminescence
      let drawOpacity = p.opacity;
      if (p.bioluminescent) {
        const pulse = Math.sin(Date.now() * p.pulseSpeed + p.pulsePhase) * 0.5 + 0.5;
        drawOpacity = p.opacity + pulse * 0.4;
      }

      // Draw particle with theme awareness
      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = isLight 
        ? `hsla(${p.hue}, 70%, 40%, ${drawOpacity * 0.75})` 
        : `hsla(${p.hue}, 60%, 75%, ${drawOpacity})`;
      ctx.fill();

      // Glow effect for bioluminescent particles
      if (p.bioluminescent && drawOpacity > 0.3) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * 4, 0, Math.PI * 2);
        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 4);
        if (isLight) {
          gradient.addColorStop(0, `hsla(${p.hue}, 80%, 45%, ${drawOpacity * 0.25})`);
          gradient.addColorStop(1, `hsla(${p.hue}, 80%, 45%, 0)`);
        } else {
          gradient.addColorStop(0, `hsla(${p.hue}, 70%, 70%, ${drawOpacity * 0.15})`);
          gradient.addColorStop(1, `hsla(${p.hue}, 70%, 70%, 0)`);
        }
        ctx.fillStyle = gradient;
        ctx.fill();
      }
    });

    animId = requestAnimationFrame(animate);
  }

  animate();
}

/* ==========================================================================
   SCROLL FADE-IN ANIMATIONS
   ========================================================================== */
function initScrollAnimations() {
  const elements = document.querySelectorAll('.fade-in, .fade-in-stagger');
  
  // Instantly activate any element currently in or near viewport
  elements.forEach(el => {
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight * 1.1) {
      el.classList.add('visible');
    }
  });

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.05, rootMargin: '0px 0px -20px 0px' });

  elements.forEach(el => {
    observer.observe(el);
  });
}

/* ==========================================================================
   DEPTH METER & TELEMETRY – Scroll Progress & Oceanic Zones
   ========================================================================== */
function initDepthMeter() {
  const indicator = document.getElementById('depth-indicator');
  const counter = document.getElementById('depth-counter');
  const zoneEl = document.getElementById('depth-zone');
  const hudZoneText = document.getElementById('hud-zone-text');
  if (!indicator && !counter) return;

  function update() {
    const scrollHeight = document.body.scrollHeight - window.innerHeight;
    const progress = scrollHeight > 0 ? (window.scrollY / scrollHeight) : 0;
    const percent = Math.min(Math.max(progress * 100, 0), 100);

    if (indicator) {
      indicator.style.height = `${percent}%`;
    }

    // Depth descending down to -4,500m (Abyssal zone)
    const currentDepth = Math.round(progress * 4500);
    if (counter) {
      counter.textContent = `-${currentDepth.toLocaleString()}m`;
    }

    // Dynamic ocean zone based on descent depth
    let zone = 'EPIPELAGIC';
    let hudText = 'SURFACE • LIVE MODEL';

    if (currentDepth > 3500) {
      zone = 'ABYSSOPELAGIC';
      hudText = 'ABYSS • CORE ENGINE';
    } else if (currentDepth > 1800) {
      zone = 'BATHYPELAGIC';
      hudText = 'MIDNIGHT • METHODOLOGY';
    } else if (currentDepth > 400) {
      zone = 'MESOPELAGIC';
      hudText = 'TWILIGHT • FORECAST CONE';
    }

    if (zoneEl) zoneEl.textContent = zone;
    if (hudZoneText) hudZoneText.textContent = hudText;
  }

  window.addEventListener('scroll', update, { passive: true });
  update();
}

/* ==========================================================================
   FIXED NAVBAR – Scroll-aware opacity, Mobile Drawer & Scroll Spy
   ========================================================================== */
function initNavbar() {
  const nav = document.getElementById('main-nav');
  const sections = document.querySelectorAll('section[id], header');
  const navLinks = document.querySelectorAll('.nav-link');
  const mobileMenuBtn = document.getElementById('mobile-menu-btn');
  const navLinksContainer = document.getElementById('nav-links');

  // Mobile drawer toggle
  if (mobileMenuBtn && navLinksContainer) {
    mobileMenuBtn.addEventListener('click', () => {
      const isOpen = navLinksContainer.classList.toggle('open');
      mobileMenuBtn.classList.toggle('active', isOpen);
      mobileMenuBtn.setAttribute('aria-expanded', isOpen);
    });

    // Auto-close drawer on link click
    navLinks.forEach(link => {
      link.addEventListener('click', () => {
        navLinksContainer.classList.remove('open');
        mobileMenuBtn.classList.remove('active');
        mobileMenuBtn.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // Scroll listener for sticky navbar & scroll-spy
  window.addEventListener('scroll', () => {
    if (nav) {
      if (window.scrollY > 60) {
        nav.classList.add('scrolled');
      } else {
        nav.classList.remove('scrolled');
      }
    }

    // Scroll spy: activate the corresponding nav item
    const scrollPos = window.scrollY + 140;
    let currentId = '';

    sections.forEach(sec => {
      const top = sec.offsetTop;
      const height = sec.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        currentId = sec.getAttribute('id');
      }
    });

    if (currentId) {
      navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === `#${currentId}`) {
          link.classList.add('active');
        } else {
          link.classList.remove('active');
        }
      });
    }
  }, { passive: true });
}

function initTheme() {
  const urlParams = new URLSearchParams(window.location.search);
  const themeParam = urlParams.get('theme');
  // Default to dark (Fathomless deep-sea aesthetic)
  const savedTheme = themeParam || localStorage.getItem('algogpt_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeButton(savedTheme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('algogpt_theme', next);
  updateThemeButton(next);
  
  if (viewMode === 'interactive' && chartInstance) {
    renderInteractiveChart();
  } else if (viewMode === 'static') {
    updateStaticChart();
  }
}

function updateThemeButton(theme) {
  const btn = document.getElementById('theme-toggle');
  if (btn) {
    const isMobile = window.innerWidth <= 480;
    if (isMobile) {
      btn.textContent = theme === 'dark' ? '◐ LIGHT' : '◑ DARK';
      btn.title = theme === 'dark' ? 'Switch to Light Theme' : 'Switch to Dark Theme';
    } else {
      btn.textContent = theme === 'dark' ? '◐ LIGHT' : '◑ DARK';
      btn.title = theme === 'dark' ? 'Switch to Light Theme' : 'Switch to Dark Theme';
    }
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

  // Live Feed Refresh button
  const refreshBtn = document.getElementById('refresh-data-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', async () => {
      refreshBtn.classList.add('refreshing');
      await loadForecastData();
      renderDashboard();
      setTimeout(() => {
        refreshBtn.classList.remove('refreshing');
      }, 500);
    });
  }
}

async function loadForecastData() {
  try {
    const res = await fetch(`data/forecast_data.json?t=${Date.now()}`);
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
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const prefix = isLight ? 'prediction_chart_light' : 'prediction_chart';
    chartImg.src = `img/${prefix}_${currentSymbol}_${currentInterval}.png?t=${Date.now()}`;
    chartImg.onerror = () => {
      chartImg.src = `img/${prefix}_${currentInterval}.png?t=${Date.now()}`;
      chartImg.onerror = () => {
        chartImg.src = `img/${prefix}.png?t=${Date.now()}`;
      };
    };
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
      <td style="color: var(--text-muted)">₹${p.lower_10.toLocaleString('en-IN', {minimumFractionDigits: 2})} — ₹${p.upper_90.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
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

  // Theme-aware palette for the chart
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const gridColor = isLight ? 'rgba(2, 132, 199, 0.08)' : 'rgba(100, 210, 236, 0.06)';
  const textColor = isLight ? '#475569' : '#5a6a82';
  const histColor = isLight ? '#0284c7' : '#64d2ec';
  const tooltipBg = isLight ? 'rgba(255, 255, 255, 0.96)' : 'rgba(5, 12, 24, 0.92)';
  const tooltipTitle = isLight ? '#0a192f' : '#edf0f7';
  const tooltipBody = isLight ? '#334e68' : '#8a99b2';
  const tooltipBorder = isLight ? 'rgba(2, 132, 199, 0.3)' : 'rgba(100, 210, 236, 0.25)';

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
      borderColor: histColor,
      backgroundColor: 'transparent',
      borderWidth: 2,
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
        borderColor: isLight ? 'rgba(234, 88, 12, 0.45)' : 'rgba(249, 115, 22, 0.35)',
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
          backgroundColor: tooltipBg,
          titleColor: tooltipTitle,
          bodyColor: tooltipBody,
          borderColor: tooltipBorder,
          borderWidth: 1,
          padding: 10,
          cornerRadius: 6,
          titleFont: { size: 11, family: 'JetBrains Mono', weight: '600' },
          bodyFont: { size: 11, family: 'JetBrains Mono' },
          boxPadding: 4,
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
