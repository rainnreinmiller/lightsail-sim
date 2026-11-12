// Node CLI used by tests/test_js_crosscheck.py: evaluates the JS engine on parameter sets
// supplied as JSON and prints the results as JSON.
//   node web/js/crosscheck.mjs cases.json
import { readFileSync } from 'node:fs';
import * as ph from './physics.js';

const input = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const fix = (v) => (v === null ? Infinity : v);   // JSON has no Infinity
if (input.mode === 'assimilate') {
  const asm = await import('./assimilation.js');
  const design = { ...ph.DEFAULTS, ...input.design, illuminationStop: Infinity };
  const table = asm.sharedTable();
  const A = new asm.Assimilator(design, ph.DEFAULT_SPEC, { table });
  const res = { prior: A.prior, mu: A.mu, cov: A.cov, steps: [] };
  for (const chunk of input.chunks) {
    A.update(input.records.slice(chunk[0], chunk[1]));
    res.steps.push(A.summary());
  }
  res.table = input.tablePoints.map(([t, a0, L0]) => table.beta(t, a0, L0));
  process.stdout.write(JSON.stringify(res));
  process.exit(0);
}
const cases = input;
const out = cases.map((c) => {
  const p = { ...ph.DEFAULTS, ...c.params, illuminationStop: fix(c.params.illuminationStop) };
  const d = ph.derive(p);
  const res = { derived: d, models: {} };
  for (const m of ph.MODELS) {
    const s = ph.missionSummary(p, m);
    const pts = c.points.map((x) => {
      const [t, tau] = ph.timeToDistance(x, d.a0, d.L0, d.xStop, m);
      return { beta: ph.betaOfX(x, d.a0, d.L0, d.xStop, m), t, tau,
               accel: ph.properAcceleration(x, d.a0, d.L0, d.xStop, m) };
    });
    const tr = ph.buildTrajectory(p, m, 600);
    const traj = c.points.map((x) => {
      // time on the cumulative trajectory grid at the grid point nearest to x
      let best = 0;
      for (let i = 0; i < tr.x.length; i++) if (Math.abs(Math.log(tr.x[i] / x)) < Math.abs(Math.log(tr.x[best] / x))) best = i;
      return { x: tr.x[best], t: tr.t[best], tau: tr.tau[best] };
    });
    res.models[m] = { summary: s, points: pts, traj,
                      elasticities: ph.elasticities(p, 'betaFinal', m),
                      elasticitiesT: ph.elasticities(p, 'tTarget', m) };
  }
  return res;
});
process.stdout.write(JSON.stringify(out, (k, v) => (v === Infinity ? null : v)));
