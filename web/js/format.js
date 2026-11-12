// Shared number, distance and time formatting for the UI.
import { AU, LY, YEAR, DAY } from './physics.js';

const SUPD = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
export const sup = (n) => String(n).split('').map((c) => SUPD[c]).join('');
export function num(v, n = 3) {
  if (!Number.isFinite(v)) return v === Infinity ? '∞' : '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-3) {
    const e = Math.floor(Math.log10(a));
    return `${(v / 10 ** e).toFixed(n - 1)}×10${sup(e)}`;
  }
  return Number(v.toPrecision(n)).toLocaleString('en-US', { maximumFractionDigits: 8 });
}
export function fmtDist(m) {
  if (!Number.isFinite(m)) return '∞';
  if (m < 1e3) return `${num(m)} m`;
  if (m < 1e9) return `${num(m / 1e3)} km`;
  if (m < 0.5 * AU) return `${num(m / 1e9)} million km`;
  if (m < 0.03 * LY) return `${num(m / AU)} AU`;
  return `${num(m / LY)} ly`;
}
export function fmtTime(s) {
  if (!Number.isFinite(s)) return '∞';
  if (s < 1e-3) return `${num(s * 1e6)} µs`;
  if (s < 1) return `${num(s * 1e3)} ms`;
  if (s < 120) return `${num(s)} s`;
  if (s < 2 * 3600) return `${num(s / 60)} min`;
  if (s < 2 * DAY) return `${num(s / 3600)} h`;
  if (s < YEAR) return `${num(s / DAY)} d`;
  return `${num(s / YEAR)} yr`;
}
export const pctc = (b) => `${num(100 * b, b < 0.001 ? 2 : 3)}% c`;

export const TIME_TICKS = [[1e-3, '1 ms'], [0.01, '10 ms'], [0.1, '0.1 s'], [1, '1 s'], [10, '10 s'], [60, '1 min'],
  [3600, '1 h'], [DAY, '1 d'], [30 * DAY, '1 mo'], [YEAR, '1 yr'], [10 * YEAR, '10 yr'], [100 * YEAR, '100 yr'],
  [1000 * YEAR, '1000 yr']];
export const DIST_TICKS = [[1, '1 m'], [1e3, '1 km'], [1e6, '1000 km'], [1e9, '10⁶ km'], [AU, '1 AU'], [100 * AU, '100 AU'],
  [1e4 * AU, '10⁴ AU'], [LY, '1 ly'], [10 * LY, '10 ly']];
