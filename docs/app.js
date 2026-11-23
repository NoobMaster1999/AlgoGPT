'use strict';

const el = id => document.getElementById(id);
const symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY'];
const intervals = {'15m': '15 minutes', '1h': '1 hour', '1d': 'Daily'};
const price = new Intl.NumberFormat('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const dateFormat = new Intl.DateTimeFormat('en-GB', {timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hourCycle: 'h23'});
const numeric = Number.isFinite;
const number = v => numeric(v) ? price.format(v) : '—';
const percent = v => numeric(v) ? Number(v.toFixed(2)) + '%' : '—';
let latestData = null;
let scorecard = null;
let activeData = null;
let symbol = 'NIFTY';
let interval = '15m';
let archiveId = '';
let view = 'interactive';
let generation = 0;
let pageSize = 12;
let lastReview = '';
let refreshPending = false;
let lastRefreshAt = 0;
let inspectedForecast = '';
let inspectedTime = '';
const pollInterval = 5 * 60 * 1000;

function asDate(value) {
  if (typeof value !== 'string') return null;
  let stamp = value;
  const local = /^(\d{4}-\d{2}-\d{2})(?: (\d{2}:\d{2}))?(?: IST(?: .*)?)?$/.exec(value);
  if (local) stamp = local[1] + 'T' + (local[2] || '15:30') + ':00+05:30';
  if (!local && !/(Z|[+-]\d{2}:\d{2})$/i.test(stamp)) return null;
  const date = new Date(stamp);
  return Number.isFinite(+date) ? date : null;
}
function dateText(value, fallback = 'Unavailable') {
  const date = asDate(value);
  return date ? dateFormat.format(date) + ' IST' : fallback;
}
function setText(id, value) { el(id).textContent = value; }
function route() {
  const q = new URLSearchParams(location.search);
  symbol = symbols.includes(q.get('symbol')) ? q.get('symbol') : 'NIFTY';
  interval = Object.hasOwn(intervals, q.get('interval')) ? q.get('interval') : '15m';
  archiveId = q.get('forecast') || '';
  view = q.get('view') === 'static' ? 'static' : 'interactive';
}
function link(id = '', nextSymbol = symbol, nextInterval = interval, nextView = view) {
  const url = new URL(location.href);
  url.searchParams.set('symbol', nextSymbol);
  url.searchParams.set('interval', nextInterval);
  if (id) url.searchParams.set('forecast', id); else url.searchParams.delete('forecast');
  if (nextView === 'static') url.searchParams.set('view', nextView); else url.searchParams.delete('view');
  url.hash = 'live-dashboard';
  return url;
}
function setLink(key, url) {
  const a = el(key);
  a.hidden = !url;
  if (url) a.href = url; else a.removeAttribute('href');
}
async function navigate(url) {
  history.pushState(null, '', url);
  route();
  pageSize = 12;
  await display();
}
function tabs() {
  for (const [group, value] of [['symbol',symbol],['interval',interval],['view',view]]) {
    document.querySelectorAll('button[data-' + group + ']').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset[group] === value));
    });
  }
}
async function readJson(path) {
  const res = await fetch(path, {cache:'no-store'});
  if (!res.ok) throw Error('HTTP ' + res.status);
  const json = await res.json();
  if (!json || typeof json !== 'object' || Array.isArray(json)) throw Error('Invalid document');
  return json;
}
async function refresh({background = false} = {}) {
  if (refreshPending) return;
  refreshPending = true;
  lastRefreshAt = Date.now();
  if (!background) {
    el('refresh-data-btn').disabled = true;
    setText('feed-message','Refreshing published data…');
  }
  try {
    const results = await Promise.allSettled([readJson('data/forecast_data.json'),readJson('data/scorecard.json')]);
    const [feed, score] = results;
    const messages = [];
    const oldForecast = latestData?.[symbol]?.[interval];
    const oldTrack = scorecard;
    if (feed.status === 'fulfilled' && symbols.some(key => feed.value[key])) latestData = feed.value;
    else messages.push(latestData ? 'Refresh failed. Showing the previously loaded forecast feed.' : 'Forecast feed unavailable. Use Refresh data to retry.');
    if (score.status === 'fulfilled' && score.value.schema_version === 1 && score.value.outcomes && score.value.groups) scorecard = score.value;
    else messages.push(scorecard ? 'Showing the previously loaded track record.' : 'Track record unavailable. Use Refresh data to retry.');
    setText('feed-message',messages.join(' '));
    if (background && activeData && (archiveId || JSON.stringify(oldForecast) === JSON.stringify(latestData?.[symbol]?.[interval])) &&
        JSON.stringify(oldTrack?.outcomes) === JSON.stringify(scorecard?.outcomes) && JSON.stringify(oldTrack?.groups) === JSON.stringify(scorecard?.groups)) {
      renderFreshness();
      return;
    }
    await display();
  } finally {
    refreshPending = false;
    el('refresh-data-btn').disabled = false;
  }
}
function usable(d) {
  if (!d || !symbols.includes(d.symbol) || !Object.hasOwn(intervals, d.interval_key)) return false;
  if (!Array.isArray(d.historical) || !d.historical.length || !Array.isArray(d.predictions) || !d.predictions.length) return false;
  return d.historical.every(b => b && typeof b.time === 'string' && numeric(b.close)) &&
    d.predictions.every(b => b && typeof b.time === 'string' && numeric(b.close) && numeric(b.upper_75) && numeric(b.lower_25)) &&
    (d.predictions.every(b => numeric(b.upper_95) && numeric(b.lower_05)) ||
      d.predictions.every(b => numeric(b.upper_90) && numeric(b.lower_10)));
}
function emptyView(title,message) {
  activeData = null;
  setText('current-symbol-title',title);
  setText('detail-message',message);
  setText('freshness','Unavailable');
  el('freshness').dataset.state = '';
  el('forecast-content').hidden = true;
  el('chart-surface').replaceChildren();
}
function renderFreshness() {
  if (!activeData) return;
  const target = asDate(activeData.target_at || activeData.predictions.at(-1).time);
  const due = asDate(activeData.next_update_at);
  const issue = asDate(activeData.issued_at);
  const now = Date.now();
  let status = 'Timing unverified';
  if (target && +target <= now) status = 'Historical';
  else if (archiveId && target) status = 'Open horizon';
  else if (issue && due && target && +issue <= now) status = +due <= now ? 'Update delayed' : 'Current';
  setText('freshness',status);
  el('freshness').dataset.state = status === 'Current' ? 'current' : status === 'Update delayed' ? 'delayed' : '';
}
function sortedEntries() {
  return Object.entries(scorecard?.outcomes || {}).filter(([,o]) => o?.symbol === symbol && o.interval === interval)
    .sort(([idA,a],[idB,b]) => (+(asDate(b.observed_at) || 0) - +(asDate(a.observed_at) || 0)) || idB.localeCompare(idA));
}
function stat(list, title, val, hint) {
  const wrapper = document.createElement('div');
  const term = document.createElement('dt');
  const desc = document.createElement('dd');
  term.textContent = title;
  desc.textContent = val;
  if (hint) {
    const note = document.createElement('small');
    note.textContent = hint;
    desc.append(note);
  }
  wrapper.append(term,desc);
  list.append(wrapper);
}
function addRow(body,cells) {
  const row = body.insertRow();
  for (const item of cells) {
    const cell = row.insertCell();
    if (item instanceof Node) cell.append(item); else cell.textContent = item;
  }
}
function renderScores() {
  const entries = sortedEntries();
  const group = scorecard?.groups?.[symbol]?.[interval];
  setText('score-context',symbol + ' · ' + intervals[interval]);
  el('score-metrics').replaceChildren();
  const score = el('score-metrics');
  if (group) {
    stat(score,'Kronos mean absolute error',number(group.model_mae),'Index points');
    stat(score,'Last-close baseline error',number(group.baseline_mae),'Index points');
    stat(score,'50% band coverage',percent(numeric(group.coverage_50) ? group.coverage_50 * 100 : null),(group.coverage_50_count ?? 0) + ' eligible forecasts');
    stat(score,'90% band coverage',percent(numeric(group.coverage_90) ? group.coverage_90 * 100 : null),(group.coverage_90_count ?? 0) + ' eligible forecasts');
  }
  setText('score-message',group ? (group.completed_count ?? 0) + ' completed · ' + (group.pending_count ?? 0) + ' pending · ' + (group.excluded_count ?? 0) + ' excluded' : scorecard ? 'No evaluated forecasts for this selection yet.' : 'Scorecard unavailable.');
  setText('archive-message',entries.length ? 'Showing ' + Math.min(entries.length,pageSize) + ' of ' + entries.length + ' archived forecasts.' : scorecard ? 'The first published forecast for this series will appear here.' : 'Archive index unavailable.');
  const body = el('archive-body');
  body.replaceChildren();
  el('archive-table').hidden = !entries.length;
  el('show-more').hidden = entries.length <= pageSize;
  for (const [id,o] of entries.slice(0,pageSize)) {
    const a = document.createElement('a');
    a.href = link(id);
    a.textContent = dateText(o.issued_at || o.observed_at,id);
    addRow(body,[a,dateText(o.target_at),o.status,number(o.actual_close),number(o.model_abs_error)]);
  }
}
async function display() {
  const ticket = ++generation;
  tabs();
  renderScores();
  let d;
  if (archiveId) {
    const id = archiveId;
    const cached = activeData?.forecast_id === id ? activeData : null;
    if (!cached) emptyView('Opening archived forecast…','');
    setText('edition','Forecast archive');
    const returnLink = document.createElement('a');
    returnLink.href = link('');
    returnLink.textContent = 'Back to latest forecast';
    try {
      if (!/^[A-Za-z0-9_-]{1,120}$/.test(id)) throw Error('Invalid ID');
      d = cached || await readJson('data/archive/' + encodeURIComponent(id) + '.json');
      if (!usable(d) || d.forecast_id !== id) throw Error('Invalid archive');
    } catch {
      if (ticket !== generation) return;
      emptyView('Archived forecast unavailable','Check the link or retry with Refresh data. ');
      el('detail-message').append(returnLink);
      return;
    }
    if (ticket !== generation) return;
    symbol = d.symbol;
    interval = d.interval_key;
    tabs();
    renderScores();
  } else {
    lastReview = '';
    d = latestData?.[symbol]?.[interval];
    setText('edition','Latest published forecast');
    if (!usable(d) || d.symbol !== symbol || d.interval_key !== interval) {
      emptyView(symbol + ' · ' + intervals[interval],'No usable forecast is available for this selection.');
      return;
    }
  }
  activeData = d;
  setText('detail-message','');
  el('forecast-content').hidden = false;
  setText('current-symbol-title',(symbol === 'NIFTY' ? 'NIFTY 50' : symbol) + ' · ' + intervals[interval]);
  setText('observed',dateText(d.observed_at || d.last_updated));
  setText('issued',dateText(d.issued_at,'Unrecorded'));
  setText('target',dateText(d.target_at || d.predictions.at(-1).time));
  setText('next',archiveId ? 'Archived edition' : dateText(d.next_update_at,'Unscheduled'));
  renderFreshness();
  const outcome = scorecard?.outcomes?.[d.forecast_id];
  setLink('permalink',d.forecast_id ? link(d.forecast_id) : '');
  setLink('latest',archiveId ? link('') : '');
  const prior = sortedEntries().find(([id,o]) => (!archiveId || id !== archiveId) && o.status === 'completed');
  setLink('previous',prior ? link(prior[0]) : '');
  setText('outcome-summary', outcome ? (outcome.status === 'completed' ? 'Observed target close ' + number(outcome.actual_close) + ' · Model error ' + number(outcome.model_abs_error) + ' points · Last-close baseline error ' + number(outcome.baseline_abs_error) + ' points.' : outcome.status + (outcome.reason ? ' · ' + outcome.reason : '')) : '');
  const metrics = el('metrics');
  metrics.replaceChildren();
  const m = d.metrics || {};
  stat(metrics,'Upside probability',percent(m.upside_prob),'Paths ending above spot');
  stat(metrics,'Volatility amplification',percent(m.vol_amp_prob),'Paths above historical volatility');
  stat(metrics,'Expected return',percent(m.expected_return),'Mean across simulated terminal returns');
  stat(metrics,'5th percentile return',percent(m.var_95),'Lower tail of terminal returns');
  const bands = bandFields(d);
  setText('outer-heading',bands.label);
  setText('chart-desc','Last close ' + number(d.spot_close) + ' points · ' + d.predictions.length + ' forecast bars' + (numeric(d.sample_count) ? ' · ' + d.sample_count + ' simulations' : '') + (d.source ? ' · ' + d.source : ''));
  const tbody = el('predictions-tbody');
  tbody.replaceChildren();
  for (const [index,b] of d.predictions.entries()) {
    const change = numeric(d.spot_close) && d.spot_close > 0 ? (b.close / d.spot_close - 1) * 100 : null;
    addRow(tbody,[b.time,number(b.open),number(b.high),number(b.low),number(b.close),percent(change),number(b[bands.low]) + ' – ' + number(b[bands.high]),index === d.predictions.length-1 ? number(outcome?.actual_close) : '—']);
  }
  drawChart(d,outcome,bands);
  if (archiveId && outcome?.status === 'completed' && lastReview !== archiveId) {
    lastReview = archiveId;
    window.dispatchEvent(new CustomEvent('algogpt:outcome-reviewed',{detail:{id:archiveId,status:'completed'}}));
  }
}
function bandFields(d) {
  if (d.predictions.every(b => numeric(b.lower_05) && numeric(b.upper_95))) return {low:'lower_05',high:'upper_95',label:'90% band'};
  return {low:'lower_10',high:'upper_90',label:'Legacy outer bounds'};
}
function svgNode(tag,attrs,content) {
  const node = document.createElementNS('http://www.w3.org/2000/svg',tag);
  for (const [k,v] of Object.entries(attrs)) node.setAttribute(k,String(v));
  if (content !== undefined) node.textContent = content;
  return node;
}
function drawChart(d,outcome,band) {
  const surface = el('chart-surface');
  const refocus = surface.contains(document.activeElement);
  const legend = el('chart-legend');
  surface.replaceChildren();
  legend.replaceChildren();
  const observed = numeric(outcome?.actual_close) ? outcome.actual_close : null;
  for (const [style,label] of [['','Historical close'],['central','Forecast median'],['band','50% band / ' + band.label],['actual','Observed target close']]) {
    if (style === 'actual' && observed === null) continue;
    const item = document.createElement('span');
    const swatch = document.createElement('i');
    swatch.className = style;
    item.append(swatch,document.createTextNode(label));
    legend.append(item);
  }
  const width = Math.round(Math.max(280,Math.min(surface.clientWidth || 900,1100)));
  const height = width < 500 ? 220 : 295;
  const left = width < 500 ? 57 : 80;
  const right = width - 14;
  const top = 20;
  const bottom = height - 36;
  const history = d.historical.slice(-25);
  const bars = history.concat(d.predictions);
  const start = history.length - 1;
  const anchor = history.at(-1).close;
  const values = history.map(b => b.close).concat(d.predictions.flatMap(b => [b[band.low],b[band.high],b.lower_25,b.upper_75,b.close]));
  const hasSamples = d.predictions.every(b => Array.isArray(b.samples) && b.samples.length >= 3 && b.samples.slice(0,3).every(numeric));
  el('sample-toggle').hidden = !hasSamples;
  const showSamples = hasSamples && el('show-samples').checked;
  if (showSamples) values.push(...d.predictions.flatMap(b => b.samples.slice(0,3)));
  if (observed !== null) values.push(observed);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = Math.max((max-min)*.12,Math.abs(min)*.00005,.01);
  const low = min-pad;
  const high = max+pad;
  const x = i => left + (right-left)*i/(bars.length-1);
  const y = v => top + (high-v)/(high-low)*(bottom-top);
  const svg = svgNode('svg',{viewBox:'0 0 ' + width + ' ' + height,role:'img','aria-label':'Historical and forecast index closes. Exact values appear in the forecast table.'});
  const add = (tag,props,content) => { const n = svgNode(tag,props,content); svg.append(n); return n; };
  const label = (xPos,yPos,copy,anchor) => add('text',{x:xPos,y:yPos,fill:'var(--secondary)','font-size':12,'text-anchor':anchor},copy);
  for (let n=0; n<5; n++) {
    const v = low + n/4*(high-low);
    add('line',{x1:left,x2:right,y1:y(v),y2:y(v),stroke:'var(--grid)','stroke-width':1});
    label(left-5,y(v)+4,Math.round(v).toLocaleString('en-IN'),'end');
  }
  const forecastPoints = key => [anchor,...d.predictions.map(b=>b[key])].map((v,i)=>x(start+i)+','+y(v));
  function area(lower,upper,opacity) {
    add('polygon',{points:[...forecastPoints(upper),...forecastPoints(lower).reverse()].join(' '),fill:'var(--band)',opacity});
  }
  area(band.low,band.high,.20);
  area('lower_25','upper_75',.42);
  if (showSamples) {
    for (let i=0;i<3;i++) {
      const path = [anchor,...d.predictions.map(b => b.samples[i])];
      add('polyline',{points:path.map((v,j) => x(start+j)+','+y(v)).join(' '),fill:'none',stroke:'var(--forecast)',opacity:.55,'stroke-dasharray':'3 4'});
    }
  }
  add('polyline',{points:history.map((b,i)=>x(i)+','+y(b.close)).join(' '),fill:'none',stroke:'var(--history)','stroke-width':2.4});
  add('polyline',{points:forecastPoints('close').join(' '),fill:'none',stroke:'var(--forecast)','stroke-width':2.4});
  add('line',{x1:x(start),x2:x(start),y1:top,y2:bottom,stroke:'var(--secondary)','stroke-dasharray':'4 4'});
  if (observed !== null) add('circle',{cx:right,cy:y(observed),r:4.5,fill:'var(--actual)',stroke:'var(--inset)','stroke-width':1.5});
  label(left,bottom+24,bars[0].time.slice(0,10),'start');
  label(right,bottom+24,bars.at(-1).time.slice(0,10),'end');
  surface.append(svg);
  const readout = el('chart-readout');
  if (view === 'static') {
    readout.textContent = 'Static view · Index points';
    return;
  }
  const forecastKey = d.forecast_id || [d.symbol,d.interval_key,d.last_updated].join('/');
  let index = start;
  const cursor = add('line',{x1:x(start),x2:x(start),y1:top,y2:bottom,stroke:'var(--accent)','stroke-width':1});
  const marker = add('circle',{r:4,fill:'var(--accent)'});
  function select(i) {
    index = Math.max(0,Math.min(i,bars.length-1));
    const b = bars[index];
    inspectedForecast = forecastKey;
    inspectedTime = b.time;
    const projected = index > start;
    const description = b.time + ' IST · ' + (projected ? 'Median ' : 'Close ') + number(b.close) +
      (projected ? ' · 50% band ' + number(b.lower_25) + ' – ' + number(b.upper_75) : '') +
      (index === bars.length-1 && observed !== null ? ' · Actual ' + number(observed) : '');
    cursor.setAttribute('x1',x(index));
    cursor.setAttribute('x2',x(index));
    marker.setAttribute('cx',x(index));
    marker.setAttribute('cy',y(b.close));
    svg.setAttribute('aria-valuenow',index+1);
    svg.setAttribute('aria-valuetext',description);
    readout.textContent = description;
  }
  svg.setAttribute('tabindex','0');
  svg.setAttribute('role','slider');
  svg.setAttribute('aria-label','Inspect chart bars using the arrow keys');
  svg.setAttribute('aria-valuemin','1');
  svg.setAttribute('aria-valuemax',bars.length);
  const savedIndex = inspectedForecast === forecastKey ? bars.findIndex(b => b.time === inspectedTime) : -1;
  select(savedIndex < 0 ? start : savedIndex);
  if (refocus) svg.focus({preventScroll:true});
  function selectPointer(e) {
    const rect = svg.getBoundingClientRect();
    if (rect.width) select(Math.round((((e.clientX-rect.left)*width/rect.width)-left)/(right-left)*(bars.length-1)));
  }
  svg.addEventListener('pointermove',selectPointer);
  svg.addEventListener('pointerdown',selectPointer);
  svg.addEventListener('keydown',e=>{
    const targets = {ArrowRight:index+1,ArrowLeft:index-1,ArrowUp:index+1,ArrowDown:index-1,Home:0,End:bars.length-1};
    if (Object.hasOwn(targets,e.key)) {
      e.preventDefault();
      select(targets[e.key]);
    }
  });
}
function changeTheme(theme) {
  document.documentElement.dataset.theme = theme;
  setText('theme-toggle', theme === 'dark' ? 'Light theme' : 'Dark theme');
}
function init() {
  route();
  tabs();
  let saved;
  try { saved = localStorage.getItem('algogpt_theme'); } catch { /* Browser storage may be disabled. */ }
  const fromUrl = new URLSearchParams(location.search).get('theme');
  changeTheme((fromUrl || saved) === 'light' ? 'light' : 'dark');
  el('theme-toggle').addEventListener('click',()=>{
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    changeTheme(next);
    try { localStorage.setItem('algogpt_theme',next); } catch { /* Keep session theme. */ }
  });
  for (const b of document.querySelectorAll('.symbol-tab')) b.addEventListener('click',()=>navigate(link('',b.dataset.symbol)));
  for (const b of document.querySelectorAll('.interval-tab')) b.addEventListener('click',()=>navigate(link('',symbol,b.dataset.interval)));
  for (const b of document.querySelectorAll('.view-btn')) b.addEventListener('click',()=>navigate(link(archiveId,symbol,interval,b.dataset.view)));
  el('show-more').addEventListener('click',()=>{pageSize += 12; renderScores();});
  el('refresh-data-btn').addEventListener('click',refresh);
  el('show-samples').addEventListener('change',()=>{
    if (activeData) drawChart(activeData,scorecard?.outcomes?.[activeData.forecast_id],bandFields(activeData));
  });
  window.addEventListener('popstate',()=>{route(); pageSize=12; display();});
  window.setInterval(renderFreshness,60000);
  const pollIfDue = ()=>{
    if (!document.hidden && Date.now() - lastRefreshAt >= pollInterval) refresh({background:true});
  };
  window.setInterval(pollIfDue,60000);
  document.addEventListener('visibilitychange',pollIfDue);
  let lastWidth = 0;
  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(([entry])=>{
      const width = Math.round(entry.contentRect.width);
      if (width === lastWidth) return;
      lastWidth = width;
      if (activeData && width) drawChart(activeData,scorecard?.outcomes?.[activeData.forecast_id],bandFields(activeData));
    }).observe(el('chart-surface'));
  }
  refresh();
}
document.addEventListener('DOMContentLoaded',init);
