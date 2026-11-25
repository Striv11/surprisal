// Surprise timeline: pure anomaly detection (unit-tested) plus canvas drawing.

export const MIN_SPIKE_NATS = 1.5;
export const SPIKE_FACTOR = 3;

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Indices whose surprise is an anomaly: above `minNats` AND more than
 * `factor` times the median of everything before it. The absolute floor stops
 * tiny fluctuations in a well-learned stream from being flagged; the relative
 * test stops a noisy, unlearned stream from being flagged everywhere.
 */
export function findAnomalies(
  surprises: readonly number[],
  minNats = MIN_SPIKE_NATS,
  factor = SPIKE_FACTOR,
): number[] {
  const out: number[] = [];
  for (let i = 0; i < surprises.length; i++) {
    const s = surprises[i];
    if (s <= minNats) continue;
    if (i === 0 || s > factor * median(surprises.slice(0, i))) out.push(i);
  }
  return out;
}

export interface Colors { line: string; spike: string; grid: string; text: string }

/** Draw the surprise series; anomalies get a red marker and a faint column. */
export function drawTimeline(
  canvas: HTMLCanvasElement,
  surprises: readonly number[],
  colors: Colors,
): void {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const g = canvas.getContext("2d");
  if (!g) return;
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const pad = { l: 34, r: 8, t: 8, b: 16 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const yMax = Math.max(3, ...surprises);
  const x = (i: number) => pad.l + (surprises.length <= 1 ? pw / 2 : (i / (surprises.length - 1)) * pw);
  const y = (v: number) => pad.t + ph - (v / yMax) * ph;

  g.font = "11px system-ui, sans-serif";
  g.fillStyle = colors.text;
  g.strokeStyle = colors.grid;
  g.lineWidth = 1;
  for (const v of [0, yMax / 2, yMax]) {
    g.beginPath(); g.moveTo(pad.l, y(v)); g.lineTo(w - pad.r, y(v)); g.stroke();
    g.fillText(v.toFixed(1), 2, y(v) + 4);
  }
  g.fillText("nats", 2, h - 2);
  if (surprises.length === 0) return;

  const spikes = findAnomalies(surprises);
  g.fillStyle = colors.spike + "33";
  for (const i of spikes) g.fillRect(x(i) - 4, pad.t, 8, ph);

  g.strokeStyle = colors.line;
  g.lineWidth = 2;
  g.beginPath();
  surprises.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))));
  g.stroke();
  g.fillStyle = colors.spike;
  for (const i of spikes) { g.beginPath(); g.arc(x(i), y(surprises[i]), 4, 0, Math.PI * 2); g.fill(); }
}
