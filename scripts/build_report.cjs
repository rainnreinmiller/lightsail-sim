// Render report/REPORT.md to report/report.html (pandoc, MathML) and report/report.pdf (Chromium).
//   node scripts/build_report.cjs        (needs pandoc and the playwright package)
const { execFileSync } = require('node:child_process');
const path = require('node:path');
const { chromium } = require('playwright');

const dir = path.join(__dirname, '..', 'report');
execFileSync('pandoc', ['REPORT.md', '-s', '--mathml', '--no-highlight', '--css', 'report.css', '--embed-resources',
  '--resource-path', '.', '-o', 'report.html'], { cwd: dir, stdio: 'inherit' });

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.join(dir, 'report.html'), { waitUntil: 'load' });
  await page.pdf({ path: path.join(dir, 'report.pdf'), format: 'Letter', printBackground: true,
    margin: { top: '0.7in', bottom: '0.7in', left: '0.75in', right: '0.75in' },
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#888"><span class="pageNumber"></span> / <span class="totalPages"></span></div>' });
  await browser.close();
  console.log('wrote report/report.html and report/report.pdf');
})();
