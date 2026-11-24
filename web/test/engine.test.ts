import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { Engine } from "../src/engine.ts";
import { parsePattern, argmax } from "../src/pattern.ts";

const wasmPath = new URL("../../target/wasm32-unknown-unknown/release/surprisal.wasm", import.meta.url);
const skip = existsSync(wasmPath) ? false : "build the wasm first (cargo build --release --lib --target wasm32-unknown-unknown)";

async function trained(): Promise<{ e: Engine; melody: number[] }> {
  const e = await Engine.load(readFileSync(wasmPath));
  const melody = parsePattern("C E G E C E G B").symbols;
  for (let i = 0; i < 200; i++) {
    e.reset();
    for (const s of melody) e.observe(s);
  }
  e.reset();
  return { e, melody };
}

test("learns a melody, predicts the next note, spikes on a wrong one", { skip }, async () => {
  const { e, melody } = await trained();
  const s = melody.map((m) => e.observe(m));
  const typical = Math.max(...s.slice(1));
  assert.ok(typical < 0.3, `typical surprise ${typical}`);
  // After C E G E the model expects C.
  e.reset();
  for (const m of melody.slice(0, 4)) e.observe(m);
  const p = e.predict();
  assert.equal(argmax(p), melody[4]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-5);
  const wrong = e.observe(3);
  assert.ok(wrong > 5 * typical, `wrong ${wrong} vs typical ${typical}`);
  e.free();
});

test("dream mostly follows transitions seen in training", { skip }, async () => {
  const { e, melody } = await trained();
  for (const m of melody) e.observe(m);
  const seen = new Set(melody.map((m, i) => `${m}>${melody[(i + 1) % melody.length]}`));
  const d = e.generate(64, 5);
  assert.equal(d.length, 64);
  const ok = d.slice(1).filter((m, i) => seen.has(`${d[i]}>${m}`)).length;
  assert.ok(ok / 63 > 0.7, `only ${ok}/63 dreamed transitions were learned ones`);
  e.free();
});

test("bad input is rejected with clear errors", { skip }, async () => {
  const e = await Engine.load(readFileSync(wasmPath));
  assert.throws(() => e.observe(7), RangeError);
  assert.deepEqual(e.generate(0), []);
  e.free();
  assert.throws(() => e.predict(), /freed/);
});
