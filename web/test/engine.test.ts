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

test("two-level engine beats single-level on a period-16 pattern", { skip }, async () => {
  const wasm = readFileSync(wasmPath);
  const half = [0, 1, 2, 3, 0, 1, 2, 3];
  const period = [...half, ...half.slice(0, 7), 4];
  const tune = Array.from({ length: 8 }, () => period).flat();
  const run = async (top: number): Promise<number> => {
    const e = await Engine.load(wasm, 5, 24, 1, top);
    // (leak and top-down rate come from sp_create2's defaults)
    for (let i = 0; i < 300; i++) {
      e.reset();
      for (const s of tune) e.observe(s);
    }
    e.reset();
    const mean = tune.reduce((a, s) => a + e.observe(s), 0) / tune.length;
    e.free();
    return mean;
  };
  const one = await run(0);
  const two = await run(32);
  assert.ok(two < one * 0.6, `one ${one}, two ${two}`);
});

test("save/restore round-trips exactly and rejects garbage", { skip }, async () => {
  const { e, melody } = await trained();
  for (const m of melody.slice(0, 3)) e.observe(m);
  const json = e.save();
  const r = e.restore(json);
  assert.equal(r.save(), json);
  assert.deepEqual([...r.predict()], [...e.predict()]);
  assert.equal(r.vocab, e.vocab);
  for (const m of [4, 1, 2]) assert.equal(r.observe(m), e.observe(m));
  assert.throws(() => e.restore("{}"), /valid surprisal/);
  assert.throws(() => e.restore(json.slice(0, 100)), /valid surprisal/);
  r.free();
  e.free();
});
