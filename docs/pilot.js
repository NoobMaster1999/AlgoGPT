// Opt-in, device-local counts for the research pilot.
(() => {
  'use strict';
  const panel = document.getElementById('pilot-controls');
  if (!panel) return;
  const key = 'algogpt.research-pilot.v1';
  const dayInIndia = () => new Intl.DateTimeFormat('sv-SE', {
    timeZone: 'Asia/Kolkata', year: 'numeric', month: '2-digit', day: '2-digit'
  }).format(new Date());
  const dayPattern = /^\d{4}-\d{2}-\d{2}$/;
  const entry = document.createElement('p');
  const actions = document.createElement('div');
  const status = document.createElement('p');
  status.setAttribute('role', 'status');
  panel.append(entry, actions, status);
  let state = null;
  let reviewedThisVisit = false;

  function button(label, action) {
    const element = document.createElement('button');
    element.type = 'button';
    element.textContent = label;
    element.addEventListener('click', action);
    actions.append(element);
  }

  function readState() {
    const saved = JSON.parse(localStorage.getItem(key));
    if (!saved?.days || typeof saved.days !== 'object' || Array.isArray(saved.days)) return null;
    const valid = Object.entries(saved.days).every(([day, value]) =>
      dayPattern.test(day) && Number.isFinite(Date.parse(day)) && typeof value === 'boolean');
    return valid ? saved : null;
  }

  function saveDay(joining = false) {
    try {
      state = readState() || (joining ? { days: {} } : null);
      if (!state) return;
      const day = dayInIndia();
      state.days[day] = Boolean(state.days[day] || reviewedThisVisit);
      localStorage.setItem(key, JSON.stringify(state));
    } catch {
      state = null;
      status.textContent = 'Your browser could not save this pilot activity.';
    }
  }

  function exportSummary() {
    const days = Object.keys(state.days).sort();
    const weeks = new Map();
    days.forEach((day, index) => {
      const monday = new Date(day + 'T00:00:00Z');
      monday.setUTCDate(monday.getUTCDate() - (monday.getUTCDay() + 6) % 7);
      const week = monday.toISOString().slice(0, 10);
      const row = weeks.get(week) || { visitDays: 0, outcomeReviewDays: 0, returningReviewer: false };
      row.visitDays++;
      if (state.days[day]) {
        row.outcomeReviewDays++;
        if (index > 0) row.returningReviewer = true;
      }
      weeks.set(week, row);
    });
    const lines = ['week_start_ist,visit_days,outcome_review_days,returning_reviewer'];
    for (const [week, row] of weeks) {
      lines.push([week, row.visitDays, row.outcomeReviewDays, Number(row.returningReviewer)].join(','));
    }
    const url = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], {type: 'text/csv'}));
    const download = document.createElement('a');
    download.href = url;
    download.download = 'algogpt-pilot-weekly.csv';
    download.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = 'Weekly summary exported.';
  }

  function render() {
    actions.replaceChildren();
    entry.textContent = state
      ? 'This browser records days you visit and days you review a completed forecast. You can share the weekly totals yourself.'
      : 'Join the pilot to keep a private count of visits and completed forecast reviews on this device. Sharing a summary is optional.';
    if (state) {
      button('Export weekly summary', exportSummary);
      button('Leave pilot and erase counts', () => {
        try { localStorage.removeItem(key); } catch {
          status.textContent = 'Your browser could not erase the saved counts.';
          return;
        }
        state = null;
        status.textContent = 'Pilot counts erased.';
        render();
      });
    } else {
      button('Join on this device', () => {
        status.textContent = '';
        saveDay(true);
        render();
      });
    }
  }

  saveDay();
  window.addEventListener('storage', event => {
    if (event.key !== key && event.key !== null) return;
    try { state = readState(); } catch { state = null; }
    render();
  });
  window.addEventListener('algogpt:outcome-reviewed', ({ detail }) => {
    if (detail?.status !== 'completed' || !detail.id) return;
    reviewedThisVisit = true;
    if (state) saveDay();
  });
  render();
})();
