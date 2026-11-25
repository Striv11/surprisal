import { Engine } from "./engine.ts";
import { drawTimeline, findAnomalies } from "./timeline.ts";
import { NOTES, PRESETS, argmax, parsePattern } from "./pattern.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const MAX_HISTORY = 64;
let dreamSeed = 1;

let engine: Engine;
const history: { note: number; surprise: number }[] = [];

function render(): void {
  const p = engine.predict();
  const top = argmax(p);
  $("bars").replaceChildren(
    ...NOTES.map((n, i) => {
      const row = document.createElement("div");
      row.className = "bar" + (i === top ? " top" : "");
      const pct = (p[i] * 100).toFixed(0);
      row.innerHTML = `<span>${n}</span><span class="track"><span class="fill" style="width:${pct}%"></span></span><span class="pct">${pct}%</span>`;
      return row;
    }),
  );
  $("empty").hidden = history.length > 0;
  const anomalies = new Set(findAnomalies(history.map((h) => h.surprise)));
  redrawTimeline();
  $("history").replaceChildren(
    ...history.map((h, i) => {
      const li = document.createElement("li");
      if (anomalies.has(i)) li.className = "spike";
      li.title = `surprise ${h.surprise.toFixed(2)} nats`;
      const small = document.createElement("small");
      small.textContent = h.surprise.toFixed(1);
      li.append(NOTES[h.note], small);
      return li;
    }),
  );
}

function redrawTimeline(): void {
  const css = getComputedStyle(document.documentElement);
  const v = (n: string) => css.getPropertyValue(n).trim();
  drawTimeline($("timeline") as HTMLCanvasElement, history.map((h) => h.surprise), {
    line: v("--accent"), spike: v("--hot"), grid: "rgba(128,128,128,.35)", text: v("--muted"),
  });
}

function feed(note: number): void {
  const surprise = engine.observe(note);
  history.push({ note, surprise });
  if (history.length > MAX_HISTORY) history.shift();
}

function readPattern(): number[] | null {
  const { symbols, rejected } = parsePattern(($("text") as HTMLInputElement).value);
  $("error").textContent = rejected.length ? `Ignored unknown characters: ${rejected.join(" ")}` : "";
  if (symbols.length === 0) {
    $("error").textContent ||= "Enter at least one note (A–G or 1–7).";
    return null;
  }
  return symbols;
}

async function main(): Promise<void> {
  const piano = $("piano");
  NOTES.forEach((n, i) => {
    const b = document.createElement("button");
    b.textContent = n;
    b.setAttribute("aria-label", `Note ${n}`);
    b.addEventListener("click", () => { feed(i); render(); });
    piano.append(b);
  });
  for (const [name, text] of Object.entries(PRESETS)) {
    const b = document.createElement("button");
    b.textContent = name;
    b.addEventListener("click", () => { ($("text") as HTMLInputElement).value = text; });
    $("presets").append(b);
  }

  try {
    engine = await Engine.load(fetch(`${import.meta.env.BASE_URL}surprisal.wasm`));
  } catch (err) {
    $("status").textContent = `Could not load the model: ${(err as Error).message}`;
    return;
  }
  $("status").textContent = "Model ready (runs entirely in your browser).";

  $("text-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const s = readPattern();
    if (!s) return;
    s.forEach(feed);
    render();
  });
  $("train").addEventListener("click", () => {
    const s = readPattern();
    if (!s) return;
    for (let i = 0; i < 20; i++) { engine.reset(); s.forEach((n) => engine.observe(n)); }
    engine.reset();
    $("status").textContent = "Trained 20 passes. Now feed the pattern and then break it.";
    history.length = 0;
    render();
  });
  $("reset").addEventListener("click", () => { engine.reset(); history.length = 0; render(); });
  $("dream").addEventListener("click", () => {
    const notes = engine.generate(16, dreamSeed++);
    $("dreamed").textContent = notes.map((n) => NOTES[n]).join(" ");
  });
  window.addEventListener("resize", redrawTimeline);
  render();
}

main();
