// Usage: serve docs over HTTP, install Playwright, then run this file with Node.
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROMIUM_EXECUTABLE || undefined
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const base = process.env.DASHBOARD_URL || 'http://127.0.0.1:8080/';
    await page.goto(base);
    for (const symbol of ['NIFTY', 'BANKNIFTY', 'FINNIFTY']) {
      await page.locator(`button[data-symbol="${symbol}"]`).click();
      for (const interval of ['15m', '1h', '1d']) {
        await page.locator(`button[data-interval="${interval}"]`).click();
        await page.locator('#chart-surface svg').waitFor();
        assert.ok(await page.locator('#predictions-tbody tr').count() > 0);
        await page.locator('#archive-body a').first().waitFor();
      }
    }
    await page.locator('[data-view="static"]').click();
    await page.locator('#chart-surface svg').waitFor();
    await page.locator('#theme-toggle').click();
    const link = await page.locator('#permalink').getAttribute('href');
    assert.ok(link);
    await page.goto(new URL(link, base).href);
    await page.locator('#latest').waitFor();
    await page.locator('#chart-surface svg').waitFor();
    await page.locator('#latest').click();
    await page.locator('#refresh-data-btn').click();
    await page.locator('#chart-surface svg').waitFor();

    await page.locator('#pilot-panel summary').click();
    await page.getByRole('button', {name:'Join on this device'}).click();
    const download = page.waitForEvent('download');
    await page.getByRole('button', {name:'Export weekly summary'}).click();
    assert.equal((await download).suggestedFilename(), 'algogpt-pilot-weekly.csv');
    await page.getByRole('button', {name:'Leave pilot and erase counts'}).click();
    assert.equal(await page.evaluate(() => localStorage.getItem('algogpt.research-pilot.v1')), null);

    // Exercise the completed-outcome journey even on a newly seeded archive.
    const feed = await (await page.request.get(new URL('data/forecast_data.json', base).href)).json();
    const snapshot = feed.NIFTY['15m'];
    const id = snapshot.forecast_id;
    const actual = snapshot.predictions.at(-1).close;
    const fixture = {
      schema_version: 1,
      groups: {NIFTY: {'15m': {issued_count: 1, completed_count: 1, model_mae: 0}}},
      outcomes: {[id]: {symbol:'NIFTY', interval:'15m', status:'completed',
        observed_at:snapshot.observed_at, target_at:snapshot.target_at, actual_close:actual,
        model_abs_error:0, baseline_abs_error:Math.abs(snapshot.spot_close - actual)}}
    };
    await page.route('**/scorecard.json*', route => route.fulfill({json:fixture}));
    await page.goto(base);
    await page.locator('#previous').click();
    await page.waitForFunction(() => document.querySelector('#outcome-summary').textContent.includes('Observed target close'));
    assert.equal(await page.locator('#predictions-tbody tr:last-child td:last-child').textContent(),
      new Intl.NumberFormat('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2}).format(actual));
    await page.unroute('**/scorecard.json*');

    await page.setViewportSize({width: 360, height: 800});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.route('**/scorecard.json*', route => route.fulfill({status:503, body:''}));
    await page.reload();
    await page.locator('#chart-surface svg').waitFor();
    await page.waitForFunction(() => document.querySelector('#score-message').textContent.trim().length > 0);
    assert.deepEqual(errors, []);
    console.log('Browser checks passed for nine series, archive links, pilot, mobile and feed failure.');
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
